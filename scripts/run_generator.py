import json
import os

import mediapy as media
import numpy as np
import torch
from generator.generator import Generator
from generator.paired_image_dataset import SimpleTestDataset
from utils.writer import dict_to_namespace

def load_generator_ckpt(model, file_name, exp_dir):
    file_path = os.path.join(exp_dir, file_name)
    print("Loading checkpoint from {}".format(file_path))
    state_dict = torch.load(file_path)
    model.load_state_dict(state_dict["G_ema"])
    return state_dict["epoch"]

def run_generator_on_video(data_dir, exp_dir, fps, checkpoint="003000_model.ckpt", aspect_ratio_9_16=False):
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    config_json = os.path.join(exp_dir, "config.json")
    with open(config_json, "r") as cf:
        cfg_dict = json.load(cf)
    opt = dict_to_namespace(cfg_dict)
    
    dataset = SimpleTestDataset(data_dir)
    unnormalize = dataset.unnormalize
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)
    n_samples = len(dataset)
    print("{} images".format(n_samples))

    # setup model
    synthesis_kwargs = {"channel_base": 32768, "channel_max": 512, "num_fp16_res": 4, "conv_clamp": 256}
    model = (
        Generator(
            img_resolution=opt.patch_size,
            img_channels=3,
            start_res=opt.start_res,
            rgb_end_res=opt.rgb_end_res,
            synthesis_kwargs=synthesis_kwargs,
        )
        .eval()
        .requires_grad_(False)
        .cuda()
    )

    epoch = load_generator_ckpt(model, checkpoint, exp_dir)

    with torch.no_grad():
        images = []
        for i, data in enumerate(loader):
            A = data["A"].cuda()
            B = data["B"].cuda()
            # run fully convolutionally
            grid_z = torch.randn([1, model.z_dim], device=A.device).repeat(A.shape[0], 1).split(opt.batch_gpu)
            grid_c = torch.zeros((A.shape[0], 0), device=A.device).split(opt.batch_gpu)
            gen_img_c = A.split(opt.batch_gpu)
            fake = torch.cat([model(z=z, c=c, rgb_input=b).detach() for z, c, b in zip(grid_z, grid_c, gen_img_c)])
            fake = unnormalize(fake.squeeze(0).detach().cpu()).permute(1, 2, 0).clamp(0.0, 1.0)
            if aspect_ratio_9_16 is not None:
                h, w, _ = fake.shape
                exact_width = float(h) / 9. * 16.
                cropped_width = int(exact_width) // 2 * 2  # make divisible by 2
                crop_left = int((w - cropped_width) / 2)
                crop_right = w - crop_left - cropped_width
                fake = fake[:, crop_left:-crop_right]
            images.append(fake.numpy())

    video_file = os.path.join(data_dir + ".mp4")
    media.write_video(video_file, images, fps=fps)
    print("Info: ran on generator checkpoint from epoch {}".format(epoch))
