from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig, LoggingConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification

from .ganerf_trainer import GanerfTrainerConfig, RMSpropOptimizerConfig
from .ganerf_pipeline import GanerfPipelineConfig
from .ganerf_datamanager import GanerfDataManagerConfig
from .ganerf import GanerfModelConfig
from .scannetpp_dataparser import ScannetppDataParserConfig

ganerf_method = MethodSpecification(
    config=GanerfTrainerConfig(
    method_name="ganerf",
    steps_per_eval_batch=10000000,
    steps_per_eval_image=10000000,
    steps_per_save=25000,
    max_num_iterations=200000,
    logging=LoggingConfig(steps_per_log=100),
    mixed_precision=False,
    use_grad_scaler=True,
    pipeline=GanerfPipelineConfig(
        datamanager=GanerfDataManagerConfig(
            dataparser=ScannetppDataParserConfig(),
            train_num_rays_per_batch=4096,
            eval_num_rays_per_batch=4096,
            patch_size=256,
            camera_optimizer=CameraOptimizerConfig(
                mode="off",
                optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
            ),
        ),
        model=GanerfModelConfig(
            eval_num_rays_per_chunk=2 << 15,
        ),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.0001, max_steps=200000),
        },
        "discriminator": {
            "optimizer": RMSpropOptimizerConfig(lr=1e-3),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=2 << 15),
    vis="tensorboard",
),
    description="Base config for GANeRF 1st training stage",
)