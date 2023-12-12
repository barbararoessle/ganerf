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
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.model_components.ray_samplers import ProposalNetworkSampler, UniformSampler
from nerfstudio.model_components.renderers import AccumulationRenderer, DepthRenderer, NormalsRenderer, RGBRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.utils import colormaps
from nerfstudio.models.nerfacto import NerfactoModelConfig, NerfactoModel

from utils.vgg_loss import VggLoss
from generator.stylegan2 import Discriminator

@dataclass
class GanerfModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: GanerfModel)
    vgg_loss_mult: float = 0.0003
    """Vgg loss multiplier."""
    adv_loss_mult: float = 0.0003
    """Adversarial loss multiplier."""
    gp_loss_mult: float = 0.2
    """Gradient penalty loss multiplier."""

class GanerfModel(NerfactoModel):
    """GANeRF model"""

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))

        # Discriminator
        if self.config.adv_loss_mult > 0.0:
            self.discriminator = Discriminator(
                block_kwargs={},
                mapping_kwargs={},
                epilogue_kwargs={"mbstd_group_size": 4},
                channel_base=16384,
                channel_max=512,
                num_fp16_res=4,
                conv_clamp=256,
                img_channels=3,
                c_dim=0,
                img_resolution=64,
            )

        # Fields
        self.field = NerfactoField(
            self.scene_box.aabb,
            hidden_dim=self.config.hidden_dim,
            num_levels=self.config.num_levels,
            max_res=self.config.max_res,
            base_res=self.config.base_res,
            features_per_level=self.config.features_per_level,
            log2_hashmap_size=self.config.log2_hashmap_size,
            hidden_dim_color=self.config.hidden_dim_color,
            hidden_dim_transient=self.config.hidden_dim_transient,
            spatial_distortion=scene_contraction,
            num_images=self.num_train_data,
            use_pred_normals=self.config.predict_normals,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        self.density_fns = []
        num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        self.proposal_networks = torch.nn.ModuleList()
        if self.config.use_same_proposal_network:
            assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.config.proposal_net_args_list[0]
            network = HashMLPDensityField(
                self.scene_box.aabb,
                spatial_distortion=scene_contraction,
                **prop_net_args,
                implementation=self.config.implementation,
            )
            self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        else:
            for i in range(num_prop_nets):
                prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
                network = HashMLPDensityField(
                    self.scene_box.aabb,
                    spatial_distortion=scene_contraction,
                    **prop_net_args,
                    implementation=self.config.implementation,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # Samplers
        def update_schedule(step):
            return np.clip(
                np.interp(step, [0, self.config.proposal_warmup], [0, self.config.proposal_update_every]),
                1,
                self.config.proposal_update_every,
            )

        # Change proposal network initial sampler if uniform
        initial_sampler = None  # None is for piecewise as default (see ProposalNetworkSampler)
        if self.config.proposal_initial_sampler == "uniform":
            initial_sampler = UniformSampler(single_jitter=self.config.use_single_jitter)

        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=self.config.num_nerf_samples_per_ray,
            num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
            num_proposal_network_iterations=self.config.num_proposal_iterations,
            single_jitter=self.config.use_single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler,
        )

        # Collider
        self.collider = NearFarCollider(near_plane=self.config.near_plane, far_plane=self.config.far_plane)

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="median")
        self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.vgg_loss = VggLoss(device=torch.device("cuda"))

        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        if self.config.adv_loss_mult > 0.0:
            param_groups["discriminator"] = list(self.discriminator.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        compute_on = "random_rays"
        gt_rgb = batch[compute_on + "_image"].to(self.device) # RGB or RGBA image
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)  # Blend if RGBA
        metrics_dict["psnr"] = self.psnr(outputs[compute_on + "_rgb"], gt_rgb)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
            )
        return metrics_dict

    def get_patches(self, outputs, batch, patch_type="patches", fake_only=False, posneg1=False):
        real_patch_key = patch_type + "_image"
        fake_patch_key = patch_type + "_rgb"
        bs = outputs["num_patches"]
        patch_size = outputs["patch_size"]
        if not fake_only:
            real_patches = batch[real_patch_key].to(self.device).reshape(bs, patch_size, patch_size, 3)
            real_patches = real_patches.permute(0, 3, 1, 2).contiguous()
            if posneg1:
                real_patches = real_patches * 2.0 - 1.0
        else:
            real_patches = None
        fake_patches = outputs[fake_patch_key].reshape(bs, patch_size, patch_size, 3)
        fake_patches = fake_patches.permute(0, 3, 1, 2).contiguous()
        if posneg1:
            fake_patches = fake_patches * 2.0 - 1.0
        return real_patches, fake_patches

    def run_discriminator(self, x):
        bs, c, h, w = x.shape
        if self.training:
            input = x.contiguous()
        else:
            input = x
        # reshape
        d_img_resolution = self.discriminator.img_resolution
        if d_img_resolution < h:
            input = (
                input.unfold(2, d_img_resolution, d_img_resolution)
                .unfold(3, d_img_resolution, d_img_resolution)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(-1, c, d_img_resolution, d_img_resolution)
            )
        # call
        output = self.discriminator(input)
        # reshape
        if d_img_resolution < h:
            output = output.reshape(bs, 1, -1)
        return output

    def compute_adv_loss(self, fake_patch):
        pred_fake = self.run_discriminator(fake_patch)
        adv_loss = torch.nn.functional.softplus(-pred_fake).mean()
        return adv_loss

    def compute_simple_gradient_penalty(self, x):
        x.requires_grad_(True)
        pred_real = self.run_discriminator(x)
        gradients = torch.autograd.grad(outputs=[pred_real.sum()], inputs=[x], create_graph=True, only_inputs=True)[0]
        r1_penalty = gradients.square().sum([1, 2, 3]).mean()
        return r1_penalty / 2

    def compute_d_loss(self, real_patch, fake_patch):
        pred_real = self.run_discriminator(real_patch)
        pred_fake = self.run_discriminator(fake_patch.detach())
        loss_d_real = torch.nn.functional.softplus(-pred_real).mean()
        loss_d_fake = torch.nn.functional.softplus(pred_fake).mean()
        gradient_penalty = self.config.gp_loss_mult * self.compute_simple_gradient_penalty(real_patch)
        loss_d = loss_d_real + loss_d_fake
        return loss_d, gradient_penalty

    def get_discriminator_loss_dict(self, step, outputs, batch, metrics_dict=None):
        loss_dict = {}
        if self.config.gp_loss_mult > 0:
            real_patches, fake_patches = self.get_patches(outputs, batch, posneg1=True)
            # discriminator loss
            set_requires_grad(self.discriminator, True)
            loss_dict["d_loss"], loss_dict["gp"] = self.compute_d_loss(real_patches, fake_patches)
        return loss_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = {}
        # losses on random rays
        compute_on = "random_rays"
        gt_rgb = batch[compute_on + "_image"].to(self.device)
        pred_rgb, gt_rgb = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs[compute_on + "_rgb"],
            pred_accumulation=outputs[compute_on + "_accumulation"],
            gt_image=gt_rgb,
        )
        loss_dict["rgb_loss"] = self.rgb_loss(gt_rgb, pred_rgb)
        if self.config.vgg_loss_mult > 0:
            real_patches, fake_patches = self.get_patches(outputs, batch)
            include_first = outputs["num_patches"]
            real_patches = real_patches[:include_first]
            fake_patches = fake_patches[:include_first]
            loss_dict["vgg_loss"] = self.config.vgg_loss_mult * self.vgg_loss(fake_patches, real_patches)
        # adversarial loss
        if self.config.adv_loss_mult > 0:
            # compute the loss for each scale
            set_requires_grad(self.discriminator, False)
            _, fake_patches = self.get_patches(outputs, batch, fake_only=True, posneg1=True)
            loss_dict["adv_loss"] = self.config.adv_loss_mult * self.compute_adv_loss(fake_patches)
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs[compute_on + "_weights_list"], outputs[compute_on + "_ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * metrics_dict["distortion"]
            if self.config.predict_normals:
                # orientation loss for computed normals
                loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
                    outputs[compute_on + "_rendered_orientation_loss"]
                )

                # ground truth supervision for normals
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs[compute_on + "_rendered_pred_normal_loss"]
                )
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        gt_rgb = batch["image"].to(self.device)
        predicted_rgb = outputs["rgb"]  # Blended with background (black if random background)
        gt_rgb = self.renderer_rgb.blend_background(gt_rgb)
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb_1chw = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb_1chw, predicted_rgb)
        ssim = self.ssim(gt_rgb_1chw, predicted_rgb)
        lpips = self.lpips(gt_rgb_1chw, predicted_rgb.clamp(0.0, 1.0))

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {
            "rgb": outputs["rgb"],
            "real": gt_rgb, 
            "img": combined_rgb, 
            "accumulation": 
            combined_acc, 
            "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict

def set_requires_grad(net, requires_grad=False):
    for param in net.parameters():
        param.requires_grad = requires_grad