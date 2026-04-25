# sample_marmousi.py
# Author: Shijun Cheng (adapted for Marmousi II sampling)
# Description: Script to perform depth-progressive sampling on the Marmousi II
#              velocity model using either DDPM or DDIM, with optional well and
#              structural conditioning. Outputs mean/std of predictions and saves results.

import argparse
import os
import time
import numpy as np
import scipy.io as sio
import torch as th
import torch.nn.functional as F
from code.datasets import (
    normalizer_vel, denormalizer_vel,
    normalizer_depth, normalizer_well_loc,
    ricker_wavelet, convolve_wavelet,
)
from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main():
    # Parse command-line arguments
    args = create_argparser().parse_args()
    device = th.device('cuda')
    logger.configure()

    # Step index of the trained model to use
    train_step = 400000

    # Set output directory based on DDIM or DDPM
    if not args.use_ddim:
        dir_output = f'./output/ddpm/step{train_step}/'
    else:
        dir_output = f'./output/{args.timestep_respacing}/step{train_step}/'
    os.makedirs(dir_output, exist_ok=True)

    # Build model and diffusion process
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **params, use_wellguide=args.use_wellguide
    )
    # Load pretrained weights
    model.load_state_dict(
        th.load(f'{args.model_path}{train_step:06d}.pt', map_location=device)
    )
    model.to(device).eval()

    # Choose sampling function: DDPM or DDIM
    sample_fn = (
        diffusion.p_sample_loop
        if not args.use_ddim
        else diffusion.ddim_sample_loop
    )

    # Mean squared error criterion for accuracy
    criterion = th.nn.MSELoss()

    # Patch stride sizes for depth (z) and lateral (x)
    step_size_z = args.depth_size // 2
    step_size_x = args.width_size - args.width_size // 8

    # Precompute Gaussian weights for depth blending
    sigma_z = args.depth_size // 8
    z = np.arange(args.depth_size) - args.depth_size // 2
    g_z = np.exp(-(z**2) / (2 * sigma_z**2))
    g_z = th.tensor(g_z, dtype=th.float32, device=device).view(args.depth_size, 1)
    gaussian1d_z = g_z.repeat(args.out_channels, 1, args.width_size)

    # Precompute Gaussian weights for lateral blending
    sigma_x = args.width_size // 4
    x = np.arange(args.width_size) - args.width_size // 2
    g_x = np.exp(-(x**2) / (2 * sigma_x**2))
    gaussian1d_x_ori = th.tensor(g_x, dtype=th.float32, device=device).view(1, args.width_size)

    # Load Marmousi II test data
    data = sio.loadmat('../dataset/test/Marmousi.mat')
    vp = normalizer_vel(data['vel'])  # normalize velocity
    ref = data['ref']                 # structural reflectivity
    nz, nx = vp.shape

    # Convert to torch tensors: shape [1,1,nz,nx]
    vp = th.tensor(vp, dtype=th.float32, device=device).unsqueeze(0).unsqueeze(0)
    struc = th.tensor(ref, dtype=th.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Repeat lateral Gaussian for full depth
    gaussian1d_x = gaussian1d_x_ori.repeat(nz, 1)

    # Compute starting indices for patches in z and x
    indices_z = list(range(step_size_z, nz - args.depth_size + 1, step_size_z))
    if indices_z[-1] != nz - args.depth_size:
        indices_z.append(nz - args.depth_size)
    indices_x = list(range(0, nx - args.width_size + 1, step_size_x))
    if indices_x[-1] != nx - args.width_size:
        indices_x.append(nx - args.width_size)

    # Test different numbers of wells: 1, 3, 6
    for wn in [1, 3, 6]:
        start_time = time.time()
        print(f'Sampling start for Marmousi with {wn} wells usewell {args.use_well} useref {args.use_ref} ...')

        # Define global well positions in x
        if wn == 1:
            global_well_x = [300]
        elif wn == 3:
            global_well_x = [300, 750, 2200]
        else:
            global_well_x = [300, 750, 1220, 1700, 2200, 2520]

        # Initialize accumulators for full model
        total_pred_mean = th.zeros((nz, nx), device=device)
        total_pred_std = th.zeros_like(total_pred_mean)
        total_weight = th.zeros_like(total_pred_mean)

        # Loop over lateral patches
        for ix in indices_x:
            # Per-patch accumulators
            acc_mean = th.zeros((args.out_channels, nz, args.width_size), device=device)
            acc_std = th.zeros_like(acc_mean)
            acc_weight = th.zeros_like(acc_mean)

            # Top shallow patch: z=[0:depth_size]
            vp_top = vp[:, :, :args.depth_size, ix:ix+args.width_size]
            # Blend shallow prior into accumulators
            acc_mean[:, :args.depth_size] += gaussian1d_z * denormalizer_vel(vp_top[0])
            acc_weight[:, :args.depth_size] += gaussian1d_z
            vp_top = vp_top.repeat(args.batch_size, 1, 1, 1)

            # Determine local well index within patch or -1 if none
            in_wells = [x for x in global_well_x if ix <= x < ix+args.width_size]
            if len(in_wells) > 1:
                raise ValueError('Patch contains multiple wells; increase width_size')
            if len(in_wells) == 1 and args.use_well:
                lw = in_wells[0] - ix
                wl_norm = normalizer_well_loc(np.array([lw], np.float32), dmax=args.width_size)
                wlt = th.tensor(wl_norm, device=device).repeat(args.batch_size, 1)
            else:
                lw, wlt = -1, None

            # Depth-progressive sampling for this patch
            for iz in indices_z:
                print(f'Sampling patch at z={iz}, x={ix}')
                # Extract bottom patch
                vp_bot = vp[:, :, iz:iz+args.depth_size, ix:ix+args.width_size]
                struc_bot = struc[:, :, iz:iz+args.depth_size, ix:ix+args.width_size] \
                            if args.use_ref else th.zeros_like(vp_bot)

                # Prepare depth encodings for shallow & deep
                depth_top = th.arange(iz-step_size_z, iz-step_size_z+args.depth_size, device=device)
                depth_bottom = depth_top + step_size_z
                depth_top = normalizer_depth(depth_top).view(1,1,-1,1).repeat(args.batch_size,1,1,args.width_size)
                depth_bottom = normalizer_depth(depth_bottom).view(1,1,-1,1).repeat(args.batch_size,1,1,args.width_size)

                # Build conditioning tensor [batch,3,depth,width]
                cond_top = th.cat([vp_top, depth_top, depth_bottom], dim=1).float()
                # Build well-bottom tensor if present
                if args.use_well and lw >= 0:
                    well_bot = vp_bot[:, :, :, lw:lw+1]
                    well_bot = well_bot.repeat(args.batch_size, 1, 1, args.width_size)
                else:
                    well_bot = None

                # Run sampling for this window
                sample, _, _, loss_before, loss_after = sample_fn(
                    model, cond_top, struc_bot, well_bot, wlt,
                    lw, (args.batch_size, args.out_channels, args.depth_size, args.width_size),
                    scale_factor=(args.scale_factor if args.use_wellguide else None),
                    clip_denoised=args.clip_denoised,
                )
                vp_top = sample.clone()  # update shallow prior for next window

                # Accumulate mean and std for this window
                denorm = denormalizer_vel(sample.squeeze())
                acc_mean[:, iz:iz+args.depth_size] += gaussian1d_z * denorm.mean(dim=0)
                acc_std[:, iz:iz+args.depth_size]  += gaussian1d_z * denorm.std(dim=0)
                acc_weight[:, iz:iz+args.depth_size] += gaussian1d_z

            # Normalize per-patch accumulators
            patch_mean = acc_mean / acc_weight
            patch_std  = acc_std  / acc_weight
            # Blend laterally into full model
            total_pred_mean[:, ix:ix+args.width_size] += gaussian1d_x_ori * patch_mean.squeeze()
            total_pred_std[:, ix:ix+args.width_size]  += gaussian1d_x_ori * patch_std.squeeze()
            total_weight[:, ix:ix+args.width_size]    += gaussian1d_x_ori

        # Final normalization across full model
        total_pred_mean /= total_weight
        total_pred_std  /= total_weight

        # Compute accuracy
        with th.no_grad():
            accs = criterion(total_pred_mean, denormalizer_vel(vp.squeeze()))

        # Save results to .mat file
        filename = (
            f"{dir_output}Marmousi_wn{wn}_batch{args.batch_size}"
            f"_usewell{args.use_well}_useref{args.use_ref}"
            f"_wellguide{args.use_wellguide}" +
            (f"_scale{args.scale_factor}" if args.use_wellguide else "") +
            ".mat"
        )
        sio.savemat(filename, {
            'pred_mean': total_pred_mean.cpu().numpy(),
            'pred_std':  total_pred_std.cpu().numpy(),
            'accs':      accs.item(),
            'time':      time.time() - start_time,
            'loss_before': np.array(loss_before, dtype=np.float32),
            'loss_after':  np.array(loss_after, dtype=np.float32),
        })
        print(f'Marmousi inference time: {time.time() - start_time:.2f}s')

    logger.log("sampling complete")


def gaussian_weight(x, y, sigma):
    """Compute 2D Gaussian weight given x,y offsets and sigma."""
    return th.exp(-((x**2 + y**2)/(2*sigma**2))).float()


def gaussian_1d(length, sigma):
    """Compute 1D Gaussian kernel of given length and sigma."""
    coord = th.arange(length).float() - (length - 1)/2
    return th.exp(-coord**2/(2*sigma**2))


def create_argparser():
    # Default arguments for sampling
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        batch_size=50,
        model_path="../trained_model/ema_0.999_",
        depth_size=32,
        width_size=320,
        dt=1e-3,
        use_well=True,
        use_ref=True,
        use_wellguide=False,
        scale_factor=20,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
