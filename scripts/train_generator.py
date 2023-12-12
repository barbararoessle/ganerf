import argparse
import copy
import json
import os
import shutil

import cleanfid.fid as clean_fid
import cv2
import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from generator.paired_image_dataset import PairedImageDataset
from generator.generator import Generator
from generator.stylegan2 import Discriminator
from generator.generator_loss import GeneratorLoss
from generator import dnnlib
from generator.torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from utils.writer import namespace_to_dict, dict_to_namespace, print_network_info, write_output_json
from utils.tracker import MeanTracker

def load_generator_ckpt(G, G_opt, D, D_opt, G_ema, file_name, exp_dir):
    file_path = os.path.join(exp_dir, file_name)
    print("Loading checkpoint from {}".format(file_path))
    state_dict = torch.load(file_path)
    if G is not None:
        G.load_state_dict(state_dict["G"])
    if G_opt is not None:
        G_opt.load_state_dict(state_dict["optimizer_G"])
    if D is not None:
        D.load_state_dict(state_dict["D"])
    if D_opt is not None:
        D_opt.load_state_dict(state_dict["optimizer_D"])
    if G_ema is not None:
        G_ema.load_state_dict(state_dict["G_ema"])
    return state_dict["epoch"] + 1, state_dict["min_val_loss"]

def save_generator_ckpt(epoch, G, G_opt, D, D_opt, G_ema, val_loss, file_name, exp_dir):
    torch.save(
        {
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "G_ema": G_ema.state_dict(),
            "optimizer_G": G_opt.state_dict(),
            "optimizer_D": D_opt.state_dict(),
            "min_val_loss": val_loss,
        },
        os.path.join(exp_dir, file_name),
    )

def compute_metrics( fake, real, unnormalize, psnr, ssim, lpips):
    generator_rgb = unnormalize(fake.detach()).clamp(0.0, 1.0)
    target = unnormalize(real.detach()).clamp(0.0, 1.0)
    metrics = dict()
    metrics["generator_psnr"] = float(
        np.mean([psnr(img.unsqueeze(0), tar.unsqueeze(0)).cpu() for img, tar in zip(generator_rgb, target)])
    )
    metrics["generator_ssim"] = float(
        np.mean(
            [ssim(img.unsqueeze(0), tar.unsqueeze(0), data_range=1.0).cpu() for img, tar in zip(generator_rgb, target)]
        )
    )
    metrics["generator_lpips"] = float(
        np.mean([lpips(img.unsqueeze(0), tar.unsqueeze(0)).detach().cpu() for img, tar in zip(generator_rgb, target)])
    )
    return metrics

to8b = lambda x: (255 * torch.clamp(x, min=0, max=1)).to(torch.uint8)

def main():
    parser = argparse.ArgumentParser(
        description="Training generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output_dir", default="", help="output directory")
    parser.add_argument("--exp_name", type=str, default="", help="experiment name")
    parser.add_argument("--scene", default="", help="scene")
    parser.add_argument("--patch_size", type=int, default=256, help="patch size")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument(
        "--batch_gpu", type=int, default=8, help="batch gpu - batch is split up and gradients are accumulated"
    )
    parser.add_argument("--l1_weight", type=float, default=3.0, help="l1 weight")
    parser.add_argument("--vgg_weight", type=float, default=1.0, help="vgg weight")
    parser.add_argument("--gan_weight", type=float, default=1.0, help="gan weight")
    parser.add_argument("--r1_gamma", type=float, default=3.0, help="r1 weight")
    parser.add_argument("--start_res", type=int, default=4, help="image resolution to start upsampling from")
    parser.add_argument("--rgb_end_res", type=int, default=128, help="last image resolution to concatenate input rgb")
    parser.add_argument(
        "--hflip", default=True, type=lambda x: (str(x).lower() in ["true", "1", "yes"]), help="flip horizontal"
    )
    parser.add_argument("--n_epochs", type=int, default=3000, help="number epochs")
    parser.add_argument("--ema_nimg", type=int, default=10000, help="number images to average generator over")
    parser.add_argument("--validate_freq", type=int, default=100, help="frequency of validating")
    parser.add_argument("--print_freq", type=int, default=10, help="frequency of print on console")
    parser.add_argument("--decay_rate", type=float, default=0.9992, help="decay rate")
    parser.add_argument("--start_decay", type=int, default=100000, help="start learning rate decay in epochs")
    parser.add_argument("--end_decay", type=int, default=5000, help="end learning rate decay in epochs")

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = False  # Allow PyTorch to internally use tf32 for matmul
    torch.backends.cudnn.allow_tf32 = False  # Allow PyTorch to internally use tf32 for convolutions
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.
    device = torch.device("cuda")

    opt = parser.parse_args()

    exp_dir = os.path.join(opt.output_dir, opt.scene, "ganerf", opt.exp_name, "generator")

    config_json = os.path.join(exp_dir, "config.json")
    ckpt_file = os.path.join(exp_dir, "best.ckpt")
    continue_train = os.path.exists(ckpt_file)
    if continue_train:
        with open(config_json, "r") as cf:
            cfg_dict = json.load(cf)
        opt = dict_to_namespace(cfg_dict)
    else:
        os.makedirs(exp_dir, exist_ok=True)
        with open(config_json, "w") as cf:
            json.dump(namespace_to_dict(opt), cf, indent=4)
    print(opt)

    train_dataset = PairedImageDataset(opt, split="train")
    full_image_size = eval_image_size = train_dataset.full_image_size
    unnormalize = train_dataset.unnormalize
    val_dataset = PairedImageDataset(opt, split="test")
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    train_loader_full_images = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    val_loader_full_images = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0
    )
    n_train_samples = len(train_dataset)
    n_val_samples = len(val_dataset)
    print("{} training images, {} val images".format(n_train_samples, n_val_samples))

    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()  # for range [0, 1]
    ssim = structural_similarity_index_measure
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).cuda()  # for range [0, 1]

    # setup model
    synthesis_kwargs = {"channel_base": 32768, "channel_max": 512, "num_fp16_res": 4, "conv_clamp": 256}
    G = (
        Generator(
            img_resolution=opt.patch_size,
            img_channels=3,
            start_res=opt.start_res,
            rgb_end_res=opt.rgb_end_res,
            synthesis_kwargs=synthesis_kwargs,
        )
        .train()
        .requires_grad_(False)
        .cuda()
    )
    D = (
        Discriminator(
            block_kwargs={},
            mapping_kwargs={},
            epilogue_kwargs={"mbstd_group_size": 4},
            channel_base=32768,
            channel_max=512,
            num_fp16_res=4,
            conv_clamp=256,
            img_channels=3,
            c_dim=0,
            img_resolution=128,
        )
        .train()
        .requires_grad_(False)
        .cuda()
    )
    print_network_info(G)
    print_network_info(D)

    G_ema = copy.deepcopy(G).eval()
    ddp_modules = dict()
    for name, module in [("G_synthesis", G.synthesis), ("D", D), (None, G_ema)]:
        if name is not None:
            ddp_modules[name] = module

    # setup training phases and optimizers
    loss = GeneratorLoss(
        device=device,
        **ddp_modules,
        unnormalize=unnormalize,
        l1_weight=opt.l1_weight,
        vgg_weight=opt.vgg_weight,
        gan_weight=opt.gan_weight,
        r1_gamma=opt.r1_gamma,
    )
    phases = []
    G_opt_kwargs = {"lr": 0.002, "betas": [0, 0.99], "eps": 1e-08}
    D_opt_kwargs = {"lr": 0.002, "betas": [0, 0.99], "eps": 1e-08}
    G_reg_interval = 4
    D_reg_interval = 16
    for name, module, opt_kwargs, reg_interval in [
        ("G", G, G_opt_kwargs, G_reg_interval),
        ("D", D, D_opt_kwargs, D_reg_interval),
    ]:
        if reg_interval is None:
            opti = torch.optim.Adam(params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "both", module=module, opt=opti, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta**mb_ratio for beta in opt_kwargs.betas]
            opti = torch.optim.Adam(module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name + "main", module=module, opt=opti, interval=1)]
            phases += [dnnlib.EasyDict(name=name + "reg", module=module, opt=opti, interval=reg_interval)]

    print("Initialized training phases {}".format([p.name for p in phases]))

    G_opt = phases[0].opt
    D_opt = phases[1].opt if G_reg_interval is None else phases[2].opt
    G_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=G_opt, gamma=opt.decay_rate)
    D_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=D_opt, gamma=opt.decay_rate)

    epoch_count = 1
    min_lpips = 1e6
    if continue_train:
        epoch_count, min_lpips = load_generator_ckpt(G, G_opt, D, D_opt, G_ema, "best.ckpt", exp_dir)

    tb = SummaryWriter(log_dir=os.path.join(exp_dir, "tb"))

    batch_idx = 0  # the total number of training iterations
    for epoch in range(epoch_count, opt.n_epochs + 1):
        train_metrics = MeanTracker()
        for _, data in enumerate(train_loader):  # inner loop within one epoch

            # get data
            phase_real_img = (data["B"].to(device)).split(opt.batch_gpu)  # same for all phases
            phase_real_c = torch.zeros((opt.batch_size, 0), device=device).split(opt.batch_gpu)  # same for all phases
            phase_gen_img_c = data["A"].to(device).split(opt.batch_gpu)  # same for all phases
            all_gen_z = torch.randn([len(phases) * opt.batch_size, G.z_dim], device=device)
            all_gen_z = [
                phase_gen_z.split(opt.batch_gpu) for phase_gen_z in all_gen_z.split(opt.batch_size)
            ]  # different for phases
            all_gen_c = torch.zeros((opt.batch_size * len(phases), 0), device=device)
            all_gen_c = [
                phase_gen_c.split(opt.batch_gpu) for phase_gen_c in all_gen_c.split(opt.batch_size)
            ]  # different for phases

            # run training phases
            for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
                if batch_idx % phase.interval != 0:
                    continue
                if opt.gan_weight < 0.0 and phase.name in ["Dboth", "Dmain", "Dreg"]:
                    continue
                # initialize gradient accumulation
                phase.opt.zero_grad(set_to_none=True)
                phase.module.requires_grad_(True)

                # accumulate gradients over multiple rounds
                for real_img, real_c, gen_img_c, gen_z, gen_c in zip(
                    phase_real_img, phase_real_c, phase_gen_img_c, phase_gen_z, phase_gen_c
                ):
                    gain = phase.interval
                    loss.accumulate_gradients(
                        phase=phase.name,
                        real_img=real_img,
                        real_c=real_c,
                        gen_img_c=gen_img_c,
                        gen_z=gen_z,
                        gen_c=gen_c,
                        gain=gain,
                    )

                # update weights
                phase.module.requires_grad_(False)
                for param in phase.module.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                phase.opt.step()
            # update G_ema
            ema_beta = 0.5 ** (opt.batch_size / max(opt.ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

            # track train losses
            train_metrics.add(loss.curr_metrics)
            loss.reset_metrics()

            # update state
            batch_idx += 1

        if epoch > opt.start_decay and epoch < opt.end_decay:
            G_lr_scheduler.step()
            D_lr_scheduler.step()

        if epoch % opt.print_freq == 0:
            mean_train_metrics = train_metrics.as_dict()
            loss_string = ", ".join(["{} {:.4f}".format(k, v) for k, v in mean_train_metrics.items()])
            print("Epoch {:>3d}, {}".format(epoch, loss_string))
            for k, v in mean_train_metrics.items():
                tb.add_scalars(k, {"train": v}, epoch)
            g_lr = G_opt.param_groups[0]["lr"]
            d_lr = D_opt.param_groups[0]["lr"]
            tb.add_scalar("g_lr", g_lr, epoch)
            tb.add_scalar("d_lr", d_lr, epoch)

        if epoch % opt.validate_freq == 0:
            val_metrics = MeanTracker()
            with torch.no_grad():
                train_dataset.activate_augmentations(False)
                train_dataset.load_full_image = True
                val_dataset.activate_augmentations(False)
                val_dataset.load_full_image = True
                epoch_dir = os.path.join(exp_dir, "test_images_{}".format(epoch))
                train_real_dir = os.path.join(epoch_dir, "train_real")
                os.makedirs(train_real_dir, exist_ok=True)
                test_gen_dir = os.path.join(epoch_dir, "test_gen")
                os.makedirs(test_gen_dir, exist_ok=True)
                n_crops = 1
                if eval_image_size[0] > 1000:
                    n_crops = 4
                h_crop_dim = eval_image_size[0] // n_crops
                w_crop_dim = eval_image_size[1] // n_crops
                for i, val_data in enumerate(train_loader_full_images):
                    # write train rgb to compute kid
                    B = val_data["B"]
                    # resize B if needed
                    if eval_image_size[0] != full_image_size[0] or eval_image_size[1] != full_image_size[1]:
                        B = torchvision.transforms.functional.resize(B, size=eval_image_size)
                    # devide large images into crops to maintain meaningfull kid computation (on smaller 299x299 resolution)
                    unnorm_B = to8b(unnormalize(B)[0].permute(1, 2, 0))  # batch size is 1
                    image_8b_crops = (
                        unnorm_B.unfold(0, h_crop_dim, h_crop_dim)
                        .unfold(1, w_crop_dim, w_crop_dim)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(-1, h_crop_dim, w_crop_dim, 3)
                        .numpy()
                    )
                    for n, image_8b in enumerate(image_8b_crops):
                        image_file = str(i) + "_" + str(n) + ".png"
                        cv2.imwrite(
                            os.path.join(train_real_dir, image_file), cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR)
                        )

                for i, val_data in enumerate(val_loader_full_images):
                    A = val_data["A"].to(device)
                    grid_z = torch.randn([1, G.z_dim], device=device).repeat(A.shape[0], 1).split(opt.batch_gpu)
                    grid_c = torch.zeros((A.shape[0], 0), device=device).split(opt.batch_gpu)
                    gen_img_c = A.split(opt.batch_gpu)
                    fake = torch.cat(
                        [
                            G_ema(z=z, c=c, rgb_input=b, noise_mode="random").detach()
                            for z, c, b in zip(grid_z, grid_c, gen_img_c)
                        ]
                    )

                    B = val_data["B"].to(device)
                    # resize A, B, val_data, fake if needed
                    if eval_image_size[0] != full_image_size[0] or eval_image_size[1] != full_image_size[1]:
                        A = torchvision.transforms.functional.resize(A, size=eval_image_size)
                        B = torchvision.transforms.functional.resize(B, size=eval_image_size)
                        fake = torchvision.transforms.functional.resize(fake, size=eval_image_size)
                        val_data["A"] = A.cpu()
                        val_data["B"] = B.cpu()

                    val_color_metrics = compute_metrics(fake, B, unnormalize, psnr, ssim, lpips)
                    val_metrics.add(val_color_metrics, weight=A.shape[0])
                    # write crops for kid computation
                    unnorm_fake = to8b(unnormalize(fake)[0].permute(1, 2, 0))  # batch size is 1
                    image_8b_crops = (
                        unnorm_fake.unfold(0, h_crop_dim, h_crop_dim)
                        .unfold(1, w_crop_dim, w_crop_dim)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(-1, h_crop_dim, w_crop_dim, 3)
                        .cpu()
                        .numpy()
                    )
                    for n, image_8b in enumerate(image_8b_crops):
                        image_file = str(i) + "_" + str(n) + ".png"
                        cv2.imwrite(os.path.join(test_gen_dir, image_file), cv2.cvtColor(image_8b, cv2.COLOR_RGB2BGR))
                    # write file
                    file_path = os.path.join(epoch_dir, "{}.png".format(i))
                    cv2.imwrite(file_path, cv2.cvtColor(unnorm_fake.cpu().numpy(), cv2.COLOR_RGB2BGR))
                train_dataset.activate_augmentations(True)
                train_dataset.load_full_image = False
                val_dataset.activate_augmentations(True)
                val_dataset.load_full_image = False
                mean_val_metrics = val_metrics.as_dict()

                mean_val_metrics["kid"] = clean_fid.compute_kid(
                    test_gen_dir,
                    train_real_dir,
                )
                shutil.rmtree(train_real_dir)
                shutil.rmtree(test_gen_dir)

                loss_string = ", ".join(["{} {:.4f}".format(k, v) for k, v in mean_val_metrics.items()])
                print("Validate full images epoch {:>3d}, {}".format(epoch, loss_string))
                for k, v in mean_val_metrics.items():
                    tb.add_scalars(k + "_full_images", {"val": v}, epoch)
                json_file = os.path.join(exp_dir, "output_{}.json".format(epoch))
                write_output_json(opt.exp_name, "ganerf", os.path.join(exp_dir, "{:0>6}_model.ckpt".format(epoch)), mean_val_metrics, json_file)
                if mean_val_metrics["generator_lpips"] < min_lpips:
                    min_lpips = mean_val_metrics["generator_lpips"]
                    save_generator_ckpt(epoch, G, G_opt, D, D_opt, G_ema, min_lpips, "best.ckpt", exp_dir)

        if (epoch % 1000) == 0:
            save_generator_ckpt(epoch, G, G_opt, D, D_opt, G_ema, min_lpips, "{:0>6}_model.ckpt".format(epoch), exp_dir)
    save_generator_ckpt(epoch, G, G_opt, D, D_opt, G_ema, min_lpips, "{:0>6}_model.ckpt".format(epoch), exp_dir)

if __name__ == "__main__":
    main()
