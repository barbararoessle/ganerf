# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code adapted from https://github.com/nerfstudio-project/nerfstudio/

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Literal, Optional, Type
import os
import shutil

import torch
import cv2
import cleanfid.fid as clean_fid
from torch.cuda.amp.grad_scaler import GradScaler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)

from nerfstudio.utils import profiler
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils.writer import to8b

from .ganerf_datamanager import GanerfDataManagerConfig
from .ganerf import GanerfModelConfig

@dataclass
class GanerfPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GanerfPipeline)
    """target class to instantiate"""
    datamanager: GanerfDataManagerConfig = GanerfDataManagerConfig()
    """specifies the datamanager config"""
    model: GanerfModelConfig = GanerfModelConfig()
    """specifies the model config"""


class GanerfPipeline(VanillaPipeline):
    def __init__(
        self,
        config: VanillaPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config, device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank, grad_scaler=grad_scaler
        )

    @profiler.time_function
    def get_prediction(self, step: int):
        ray_bundles, batch = self.datamanager.next_train(step)
        model_outputs = {
            "patch_size": self.datamanager.config.patch_size,
            "num_patches": self.datamanager.config.num_patches,
        }
        for key, ray_bundle in ray_bundles.items():
            if "patches" in key:
                # do not optimize poses with patch based losses
                ray_bundle.origins = ray_bundle.origins.detach()
                ray_bundle.directions = ray_bundle.directions.detach()
            model_output = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
            model_outputs.update({key + "_" + k: v for k, v in model_output.items()})
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        if self.config.datamanager.camera_optimizer is not None:
            camera_opt_param_group = self.config.datamanager.camera_optimizer.param_group
            if camera_opt_param_group in self.datamanager.get_param_groups():
                # Report the camera optimization metrics
                metrics_dict["camera_opt_translation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, :3].norm()
                )
                metrics_dict["camera_opt_rotation"] = (
                    self.datamanager.get_param_groups()[camera_opt_param_group][0].data[:, 3:].norm()
                )

        return model_outputs, metrics_dict, batch

    @profiler.time_function
    def get_discriminator_train_loss_dict(self, step: int, model_outputs, batch, metrics_dict):
        loss_dict = self.model.get_discriminator_loss_dict(step, model_outputs, batch, metrics_dict)
        return loss_dict

    @profiler.time_function
    def get_nerf_train_loss_dict(self, step: int, model_outputs, batch, metrics_dict):
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    @profiler.time_function
    def get_average_eval_image_metrics(self, step: Optional[int] = None, write_path=None, eval_on_trainset=False):
        """Iterate over all the images in the eval dataset and get the average.
        Returns:
            metrics_dict: dictionary of metrics
        """
        with_kid = True
        self.eval()
        with torch.no_grad():
            metrics_dict_list = []
            if write_path is not None:
                image_dir = os.path.join(write_path, "{}_images_{}".format("train" if eval_on_trainset else "test", step))
                os.makedirs(image_dir, exist_ok=True)
                if with_kid:
                    for _, batch in self.datamanager.fixed_indices_train_dataloader:
                        train_rgb_8b = to8b(batch["image"].to(self.device))

                        # write train rgb to compute kid
                        train_real_dir = os.path.join(image_dir, "train_real")
                        os.makedirs(train_real_dir, exist_ok=True)
                        h, w, _ = train_rgb_8b.shape
                        # devide large images into crops to maintain meaningfull kid computation (on smaller 299x299 resolution)
                        n_crops = 1
                        if h > 1000:
                            n_crops = 4
                        h_crop_dim = h // n_crops
                        w_crop_dim = w // n_crops
                        image_8b_crops = (
                            train_rgb_8b.unfold(0, h_crop_dim, h_crop_dim)
                            .unfold(1, w_crop_dim, w_crop_dim)
                            .permute(0, 1, 3, 4, 2)
                            .reshape(-1, h_crop_dim, w_crop_dim, 3)
                            .cpu()
                            .numpy()
                        )
                        for n, image_8b in enumerate(image_8b_crops):
                            image_file = str(batch["image_idx"]) + "_" + str(n) + ".png"
                            cv2.imwrite(
                                os.path.join(train_real_dir, image_file), cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR)
                            )
            dl = self.datamanager.fixed_indices_train_dataloader if eval_on_trainset else self.datamanager.fixed_indices_eval_dataloader
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                MofNCompleteColumn(),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    "[green]Evaluating images...", total=len(dl)
                )
                for camera_ray_bundle, batch in dl:
                    # time this the following line
                    inner_start = time()
                    height, width = camera_ray_bundle.shape
                    num_rays = height * width
                    outputs = self.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
                    metrics_dict, images_dict = self.model.get_image_metrics_and_images(outputs, batch)
                    # write images
                    if write_path is not None:
                        for image_name, image in images_dict.items():
                            if image_name in [
                                "img",
                                "depth",
                                "rgb",
                                "real",
                            ]:
                                image_8b = to8b(image)
                                test_image_type_dir = os.path.join(image_dir, image_name)
                                os.makedirs(test_image_type_dir, exist_ok=True)
                                image_file = str(batch["image_idx"]) + ".png"
                                cv2.imwrite(
                                    os.path.join(test_image_type_dir, image_file),
                                    cv2.cvtColor(image_8b.cpu().numpy(), cv2.COLOR_RGB2BGR),
                                )
                                # write crops for kid computation
                                if with_kid:
                                    test_rendered_dir = os.path.join(image_dir, "test_rendered")
                                    os.makedirs(test_rendered_dir, exist_ok=True)
                                    if image_name == "rgb":
                                        h, w, _ = image_8b.shape
                                        image_8b_crops = (
                                            image_8b.unfold(0, h_crop_dim, h_crop_dim)
                                            .unfold(1, w_crop_dim, w_crop_dim)
                                            .permute(0, 1, 3, 4, 2)
                                            .reshape(-1, h_crop_dim, w_crop_dim, 3)
                                            .cpu()
                                            .numpy()
                                        )
                                        for n, image_8b in enumerate(image_8b_crops):
                                            image_file = str(batch["image_idx"]) + "_" + str(n) + ".png"
                                            cv2.imwrite(
                                                os.path.join(test_rendered_dir, image_file),
                                                cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR),
                                            )
                    assert "num_rays_per_sec" not in metrics_dict
                    metrics_dict["num_rays_per_sec"] = num_rays / (time() - inner_start)
                    fps_str = f"fps_at_{height}x{width}"
                    assert fps_str not in metrics_dict
                    metrics_dict[fps_str] = metrics_dict["num_rays_per_sec"] / (height * width)
                    metrics_dict_list.append(metrics_dict)
                    progress.advance(task)
            # average the metrics list
            metrics_dict = {}
            for key in metrics_dict_list[0].keys():
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )
            self.train()
            if with_kid:
                metrics_dict["kid"] = clean_fid.compute_kid(
                    test_rendered_dir,
                    train_real_dir,
                )
                shutil.rmtree(train_real_dir)
                shutil.rmtree(test_rendered_dir)
        return metrics_dict