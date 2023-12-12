import json
import os
import argparse

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="")
parser.add_argument('--output_dir', type=str, default="")
parser.add_argument('--scene', type=str, default="")
args = parser.parse_args()

if len(args.scene) > 0:
    scenes = [args.scene,]
else:
    scenes = sorted(os.listdir(args.input_dir))
image_size = 768  # final image height downsized from (1752, 1168)

# process scenes
for scene in scenes:
    print("Info: convert {}".format(scene))
    in_scene_dir = os.path.join(args.input_dir, "data", scene)
    trafo_json = os.path.join(in_scene_dir, "dslr", "nerfstudio", "transforms.json")
    out_path = os.path.join(args.output_dir, scene)
    meta_json_path = os.path.join(out_path, "transforms.json")
    out_undistorted_images_path = os.path.join(out_path, "undistorted_images")

    # read trafo file
    with open(trafo_json, "r") as tf:
        trafo_dict = json.load(tf)

    # read intrinsics
    fisheye = {
        "w": trafo_dict["w"],
        "h": trafo_dict["h"],
        "fx": trafo_dict["fl_x"],
        "fy": trafo_dict["fl_y"],
        "cx": trafo_dict["cx"],
        "cy": trafo_dict["cy"],
        "k1": trafo_dict["k1"],
        "k2": trafo_dict["k2"],
        "k3": trafo_dict["k3"],
        "k4": trafo_dict["k4"],
    }
    cam2worlds = dict()
    undistorted_rgb_paths = []
    test_undistorted_rgb_paths = []
    os.makedirs(out_undistorted_images_path, exist_ok=True)
    for frame_list, file_list in zip(
        [trafo_dict["frames"], trafo_dict["test_frames"]], [undistorted_rgb_paths, test_undistorted_rgb_paths]
    ):
        for frame in frame_list:
            # read rgb
            rgb_filename = os.path.basename(frame["file_path"])
            in_rgb_path = os.path.join(in_scene_dir, "dslr", "resized_images", rgb_filename)
            out_rgb_path = os.path.join(out_undistorted_images_path, rgb_filename)
            bgr = cv2.imread(in_rgb_path).astype(float)

            cam2worlds[rgb_filename] = frame["transform_matrix"]

            # undistort fisheye
            K = np.eye(3)
            K[0, 0] = fisheye["fx"]
            K[1, 1] = fisheye["fy"]
            K[0, 2] = fisheye["cx"]
            K[1, 2] = fisheye["cy"]
            D = np.array(
                [
                    fisheye["k1"],
                    fisheye["k2"],
                    fisheye["k3"],
                    fisheye["k4"],
                ]
            )
            bgr_undistorted = cv2.fisheye.undistortImage(bgr, K, D=D, Knew=K)

            # resize
            resize_size = (int(float(fisheye["w"]) * float(image_size) / float(fisheye["h"])), image_size)
            bgr_resized = (
                transforms.functional.to_tensor(Image.fromarray(bgr_undistorted.astype(np.uint8)).resize(resize_size))
                .permute(1, 2, 0)
                .numpy()
            )

            # write image
            cv2.imwrite(out_rgb_path, (bgr_resized * 255.0).astype(np.uint8))
            file_list.append(out_rgb_path)

    # resize intrinsics
    fact_x = float(resize_size[0]) / float(fisheye["w"])
    fact_y = float(resize_size[1]) / float(fisheye["h"])
    K[0, 0] *= fact_x
    K[1, 1] *= fact_y
    K[0, 2] *= fact_x
    K[1, 2] *= fact_y

    undistorted_rgb_paths = sorted(undistorted_rgb_paths)
    test_undistorted_rgb_paths = sorted(test_undistorted_rgb_paths)

    # write json file
    meta = dict()
    meta["camera_model"] = "OPENCV"
    meta["h"] = resize_size[1]
    meta["w"] = resize_size[0]
    meta["fl_x"] = K[0, 0]
    meta["fl_y"] = K[1, 1]
    meta["cx"] = K[0, 2]
    meta["cy"] = K[1, 2]

    frames = []
    test_frames = []
    for file_list, frame_list in zip([undistorted_rgb_paths, test_undistorted_rgb_paths], [frames, test_frames]):
        for rgb_path in file_list:
            frame = dict()
            rgb_filename = os.path.basename(rgb_path)
            stem = os.path.splitext(rgb_filename)[0]
            rgb_path = os.path.join(out_undistorted_images_path, rgb_filename)
            frame["file_path"] = rgb_path
            cam2world = cam2worlds[rgb_filename]
            frame["transform_matrix"] = cam2world
            frame_list.append(frame)
    meta["frames"] = frames
    meta["test_frames"] = test_frames

    with open(meta_json_path, "w") as mf:
        json.dump(meta, mf, indent=4)
