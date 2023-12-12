import os
import shutil
import argparse
from pathlib import Path

from scripts.eval import ComputePSNR

parser = argparse.ArgumentParser()
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--scene', type=str, default="")
parser.add_argument('--exp_name', type=str, default="")
parser.add_argument('--iteration', type=int, default=200000)
args = parser.parse_args()

exp_dir = os.path.join(args.output_dir, args.scene, "ganerf", args.exp_name)
dataset_dir = os.path.join(exp_dir, "generator_dataset_{}".format(args.iteration))

# render train & test views (test views should already be there)
ComputePSNR(load_config=Path(os.path.join(exp_dir, "config.yml")), load_step=args.iteration, eval_on_trainset=True).main()
ComputePSNR(load_config=Path(os.path.join(exp_dir, "config.yml")), load_step=args.iteration).main()

# copy train & test renderings and ground truth images to form the dataset for generator training
dest_train_real_dir = os.path.join(dataset_dir, "train", "real")
dest_train_nerf_dir = os.path.join(dataset_dir, "train", "nerf")
dest_test_real_dir = os.path.join(dataset_dir, "test", "real")
dest_test_nerf_dir = os.path.join(dataset_dir, "test", "nerf")
dest_dirs = [dest_train_real_dir, dest_train_nerf_dir, dest_test_real_dir, dest_test_nerf_dir]

train_result_dir = os.path.join(exp_dir, "train_images_{}".format(args.iteration))
test_result_dir = os.path.join(exp_dir, "test_images_{}".format(args.iteration))
src_train_real_dir = os.path.join(train_result_dir, "real")
src_train_nerf_dir = os.path.join(train_result_dir, "rgb")
src_test_real_dir = os.path.join(test_result_dir, "real")
src_test_nerf_dir = os.path.join(test_result_dir, "rgb")
src_dirs = [src_train_real_dir, src_train_nerf_dir, src_test_real_dir, src_test_nerf_dir]

for src_dir, dest_dir in zip(src_dirs, dest_dirs):
    shutil.copytree(src_dir, dest_dir)

# clean up
shutil.rmtree(train_result_dir)