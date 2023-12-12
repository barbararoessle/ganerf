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

import functools
import os
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, cast

from rich import box, style
from rich.panel import Panel
from rich.table import Table
import torch
from torch.cuda.amp.grad_scaler import GradScaler

from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.engine.callbacks import TrainingCallbackLocation
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import check_eval_enabled
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.engine.optimizers import OptimizerConfig

from utils.writer import write_output_json

TRAIN_INTERATION_OUTPUT = Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]

@dataclass
class RMSpropOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with RMSprop"""
    _target: Type = torch.optim.RMSprop

@dataclass
class GanerfTrainerConfig(TrainerConfig):
    _target: Type = field(default_factory=lambda: GanerfTrainer)
    save_only_latest_checkpoint: bool = False


class GanerfTrainer(Trainer):

    def __init__(self, config: TrainerConfig, local_rank: int = 0, world_size: int = 1) -> None:
        super().__init__(
            config=config, local_rank=local_rank, world_size=world_size
        )

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        # don't want to call save_dataparser_transform if pipeline's datamanager does not have a dataparser
        if isinstance(self.pipeline.datamanager, VanillaDataManager):
            self.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(
                self.base_dir / "dataparser_transforms.json"
            )

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.max_num_iterations + 1
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                while self.training_state == "paused":
                    time.sleep(0.01)
                with self.train_lock:
                    with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:
                        self.pipeline.train()

                        # training callbacks before the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                            )

                        # time the forward pass
                        loss, loss_dict, metrics_dict = self.train_iteration(step)

                        # training callbacks after the training iteration
                        for callback in self.callbacks:
                            callback.run_callback_at_location(
                                step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION
                            )

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.world_size
                        * self.pipeline.datamanager.get_train_rays_per_batch()
                        / max(0.001, train_t.duration),
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)
                    # The actual memory allocated by Pytorch. This is likely less than the amount
                    # shown in nvidia-smi since some unused memory can be held by the caching
                    # allocator and some context needs to be created on GPU. See Memory management
                    # (https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management)
                    # for more details about GPU memory management.
                    writer.put_scalar(
                        name="GPU Memory (MB)", scalar=torch.cuda.max_memory_allocated() / (1024**2), step=step
                    )

                # Do not perform evaluation if there are no validation images
                if self.pipeline.datamanager.eval_dataset:
                    self.eval_iteration(step)

                if step_check(step, self.config.steps_per_save):
                    self.save_checkpoint(step)

                writer.write_out_storage()

        # save checkpoint at the end of training
        self.save_checkpoint(step)

        # write out any remaining events (e.g., total train time)
        writer.write_out_storage()

        table = Table(
            title=None,
            show_header=False,
            box=box.MINIMAL,
            title_style=style.Style(bold=True),
        )
        table.add_row("Config File", str(self.config.get_base_dir() / "config.yml"))
        table.add_row("Checkpoint Directory", str(self.checkpoint_dir))
        CONSOLE.print(Panel(table, title="[bold][green]:tada: Training Finished :tada:[/bold]", expand=False))

        # after train end callbacks
        for callback in self.callbacks:
            callback.run_callback_at_location(step=step, location=TrainingCallbackLocation.AFTER_TRAIN)

        if not self.config.viewer.quit_on_train_completion:
            self._train_complete_viewer()

    def is_train_d(self):
        return self.pipeline.model.config.adv_loss_mult > 0.0

    @profiler.time_function
    def backward_and_update_d(self, d_loss):
        d_loss.backward()
        if has_finite_gradients(self.pipeline.model.discriminator):
            self.optimizers.optimizer_step("discriminator")

    @profiler.time_function
    def backward_and_update_nerf(self, step, loss):
        self.grad_scaler.scale(loss).backward()  # type: ignore
        if has_optimizer_for(self.optimizers, "proposal_networks"):
            optimizer_scaler_step(self.optimizers, "proposal_networks", self.grad_scaler)
        optimizer_scaler_step(self.optimizers, "fields", self.grad_scaler)
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"
        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            # run forward pass
            output, metrics_dict, batch = self.pipeline.get_prediction(step=step)
            # compute loss for discriminator
            if self.is_train_d():
                d_loss_dict = self.pipeline.get_discriminator_train_loss_dict(step, output, batch, metrics_dict)
                d_loss = functools.reduce(torch.add, d_loss_dict.values())
        # update discriminator
        if self.is_train_d():
            self.backward_and_update_d(d_loss)

        with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
            # compute loss for nerf
            output, loss_dict, metrics_dict = self.pipeline.get_nerf_train_loss_dict(step, output, batch, metrics_dict)
            loss = functools.reduce(torch.add, loss_dict.values())
        # update nerf
        self.backward_and_update_nerf(step, loss)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        # Merging loss and metrics dict into a single output.
        if self.is_train_d():
            loss_dict.update(d_loss_dict)
        
        return loss, loss_dict, metrics_dict  # type: ignore


    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step: int) -> None:
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.steps_per_eval_batch):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.steps_per_eval_image):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.steps_per_eval_all_images):
            self.run_eval_on_all_images(step)

    def run_eval_on_all_images(self, step):
        metrics_dict = self.pipeline.get_average_eval_image_metrics(step=step, write_path=self.config.get_base_dir())
        output_path = os.path.join(self.config.get_base_dir(), f"output_{step}.json")
        write_output_json(
            self.config.experiment_name,
            self.config.method_name,
            str(self.config.get_checkpoint_dir()),
            metrics_dict,
            output_path,
        )
        for k, v in metrics_dict.items():
            CONSOLE.print(f"{k}: {v}")
        writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)

def has_finite_gradients(net, filter=""):
    for n, params in net.named_parameters():
        if filter in n:
            if params.grad is not None and not params.grad.isfinite().all():
                return False
    return True

def optimizer_scaler_step(optimizers, param_group_name, grad_scaler: GradScaler) -> None:
    max_norm = optimizers.config[param_group_name]["optimizer"].max_norm
    if max_norm is not None:
        grad_scaler.unscale_(optimizers.optimizers[param_group_name])
        torch.nn.utils.clip_grad_norm_(optimizers.parameters[param_group_name], max_norm)
    if any(any(p.grad is not None for p in g["params"]) for g in optimizers.optimizers[param_group_name].param_groups):
        grad_scaler.step(optimizers.optimizers[param_group_name])

def has_optimizer_for(optimizers, network_name):
    return network_name in optimizers.optimizers