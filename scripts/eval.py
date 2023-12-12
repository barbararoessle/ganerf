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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import tyro

from nerfstudio.utils.rich_utils import CONSOLE

from utils.eval_utils import eval_setup
from utils.writer import write_output_json


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    load_step: Optional[int] = None
    eval_on_trainset: bool = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, step = eval_setup(self.load_config, load_step=self.load_step)
        metrics_dict = pipeline.get_average_eval_image_metrics(step=self.load_step, write_path=config.get_base_dir(), eval_on_trainset=self.eval_on_trainset)
        # Get the output and define the names to save to
        print(metrics_dict)
        if not self.eval_on_trainset:
            output_path = os.path.join(config.get_base_dir(), "output_{}.json".format(self.load_step))
            write_output_json(config.experiment_name, config.method_name, str(checkpoint_path), metrics_dict, output_path)
            CONSOLE.print(f"Saved results to: {output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
