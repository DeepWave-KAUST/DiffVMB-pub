# sample_indistribution.py
# Author: Shijun Cheng (adapted for in-distribution model sampling)
# Description: Script to perform depth-progressive sampling on in-distribution
#              velocity models (SEAM Arid, SEG/EAGE, Overthrust) using DDPM/DDIM
#              with optional well and structural conditioning. Outputs mean/std
#              and saves results for each model.

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
    # Parse CLI arguments
    args = create_argparser().parse_args()
    device = th.device('cuda')
    logger.configure()

    # Load specific training checkpoint step
    train_step = 400000
    # Prepare output directory depending on DDIM or DDPM
    if not args.use_ddim:
        dir_output = f'./output_singlewell/ddpm/step{train_step}/'
    else:
        dir_output = f'./output_singlewell/{args.timestep_respacing}/step{train_step}/'
    os.makedirs(dir_output, exist_ok=True)

    # Instantiate model and diffusion process
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **params,
        use_wellguide=args.use_wellguide,
    )
    # Load pretrained weights
    model.load_state_dict(
        th.load(f'{args.model_path}{train_step:06d}.pt', map_location=device)
    )
    model.to(device).eval()

    # Choose sampler: DDPM or DDIM
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # Loss metric for accuracy evaluation
    criterion = th.nn.MSELoss()

    # Define patch strides in depth and lateral directions
    step_size_z = args.depth_size // 2
    step_size_x = args.width_size - args.width_size // 8

    # Prepare Gaussian blending weights in depth
    sigma_z = args.depth_size // 8
    z = np.arange(args.depth_size) - args.depth_size // 2
    g_z = np.exp(-(z**2)/(2*sigma_z**2))
    g_z = th.tensor(g_z, dtype=th.float32, device=device).view(args.depth_size, 1)
    gaussian1d_z = g_z.repeat(args.out_channels, 1, args.width_size)

    # Prepare Gaussian blending weights in lateral
    sigma_x = args.width_size // 8
    x = np.arange(args.width_size) - args.width_size // 2
    g_x = np.exp(-(x**2)/(2*sigma_x**2))
    gaussian1d_x_ori = th.tensor(g_x, dtype=th.float32, device=device).view(1, args.width_size)

    # List of in-distribution test models
    model_list = ['SEAMArid', 'SEGEAGE', 'Overthrust']
    for md in model_list:
        start_time = time.time()
        print(f'Sampling start for {md} with well {args.use_well} ref {args.use_ref}')

        # Load velocity and reflectivity from .mat file
        data = sio.loadmat(f'../dataset/test/{md}.mat')
        vp = normalizer_vel(data['vel'])  # normalize velocities
        ref = data['ref']                  # structural reflectivity
        nz, nx = vp.shape

        # Convert to tensors with shape [1,1,nz,nx]
        vp = th.tensor(vp, dtype=th.float32, device=device).unsqueeze(0).unsqueeze(0)
        struc = th.tensor(ref, dtype=th.float32, device=device).unsqueeze(0).unsqueeze(0)

        # Repeat lateral Gaussian for full depth
        gaussian1d_x = gaussian1d_x_ori.repeat(nz, 1)

        # Compute patch start indices in z and x directions
        indices_z = list(range(step_size_z, nz - args.depth_size + 1, step_size_z))
        if indices_z[-1] != nz - args.depth_size:
            indices_z.append(nz - args.depth_size)
        indices_x = list(range(0, nx - args.width_size + 1, step_size_x))
        if indices_x[-1] != nx - args.width_size:
            indices_x.append(nx - args.width_size)

        # Determine local well index per lateral patch
        local_well_indices = [
            (args.global_well_x - ix) if ix <= args.global_well_x < ix + args.width_size else -1
            for ix in indices_x
        ]

        # Initialize full-model accumulators
        total_pred_mean = th.zeros((nz, nx), device=device)
        total_pred_std = th.zeros_like(total_pred_mean)
        total_weight = th.zeros_like(total_pred_mean)

        # Loop over lateral patches
        for idx, ix in enumerate(indices_x):
            # Per-patch accumulators
            acc_mean = th.zeros((args.out_channels, nz, args.width_size), device=device)
            acc_std  = th.zeros_like(acc_mean)
            acc_weight = th.zeros_like(acc_mean)

            # Top shallow patch: z=[0:depth_size]
            vp_top = vp[:, :, :args.depth_size, ix:ix+args.width_size]
            # Blend shallow prior
            acc_mean[:, :args.depth_size] += gaussian1d_z * denormalizer_vel(vp_top[0])
            acc_weight[:, :args.depth_size] += gaussian1d_z
            vp_top = vp_top.repeat(args.batch_size, 1, 1, 1)

            # Get well-local index
            lw = local_well_indices[idx]
            if lw >= 0 and args.use_well:
                wl_norm = normalizer_well_loc(np.array([lw], np.float32), dmax=args.width_size)
                wlt = th.tensor(wl_norm, dtype=th.float32, device=device).repeat(args.batch_size, 1)
            else:
                wlt = None

            # Depth-progressive generation
            for iz in indices_z:
                print(f'Sampling patch at z={iz}, x={ix}')
                # Extract bottom patch at depth iz
                vp_bot = vp[:, :, iz:iz+args.depth_size, ix:ix+args.width_size]
                # Structural patch if enabled
                struc_bot = struc[:, :, iz:iz+args.depth_size, ix:ix+args.width_size] \
                            if args.use_ref else th.zeros_like(vp_bot)

                # Prepare depth encodings for top and bottom
                depth_top = th.arange(iz-step_size_z, iz-step_size_z+args.depth_size, device=device)
                depth_bottom = depth_top + step_size_z
                depth_top = normalizer_depth(depth_top).view(1,1,-1,1).repeat(args.batch_size,1,1,args.width_size)
                depth_bottom = normalizer_depth(depth_bottom).view(1,1,-1,1).repeat(args.batch_size,1,1,args.width_size)

                # Build cond_top tensor: [batch,3,depth,width]
                cond_top = th.cat([vp_top, depth_top, depth_bottom], dim=1).float()

                # Build well-bottom tensor if present
                if args.use_well and lw >= 0:
                    well_bot = vp_bot[:, :, :, lw:lw+1].repeat(args.batch_size,1,1,args.width_size)
                else:
                    well_bot = None

                if not args.use_ref:
                    struc_bot = None

                # Sample using the chosen sampling function
                sample, _, _, loss_before, loss_after = sample_fn(
                    model, cond_top, struc_bot, well_bot, wlt, lw,
                    (args.batch_size, args.out_channels, args.depth_size, args.width_size),
                    scale_factor=(args.scale_factor if args.use_wellguide else None),
                    clip_denoised=args.clip_denoised,
                )
                # Update vp_top for next depth
                vp_top = sample.clone()

                # Accumulate mean and std for this depth window
                denorm = denormalizer_vel(sample.squeeze())
                acc_mean[:, iz:iz+args.depth_size] += gaussian1d_z * denorm.mean(dim=0)
                acc_std[:, iz:iz+args.depth_size]  += gaussian1d_z * denorm.std(dim=0)
                acc_weight[:, iz:iz+args.depth_size] += gaussian1d_z

            # Normalize per-patch results
            patch_mean = acc_mean / acc_weight
            patch_std  = acc_std  / acc_weight
            # Blend laterally into full volume
            total_pred_mean[:, ix:ix+args.width_size] += gaussian1d_x_ori * patch_mean.squeeze()
            total_pred_std[:,  ix:ix+args.width_size] += gaussian1d_x_ori * patch_std.squeeze()
            total_weight[:,    ix:ix+args.width_size] += gaussian1d_x_ori

        # Final normalization for full model
        total_pred_mean /= total_weight
        total_pred_std  /= total_weight

        # Compute accuracy
        with th.no_grad():
            accs = criterion(total_pred_mean, denormalizer_vel(vp.squeeze()))

        # Save to .mat
        filename = (
            f"{dir_output}{md}_batch{args.batch_size}_usewell{args.use_well}"
            f"_useref{args.use_ref}_wellguide{args.use_wellguide}.mat"
        )
        sio.savemat(filename, {
            'pred_mean': total_pred_mean.cpu().numpy(),
            'pred_std':  total_pred_std.cpu().numpy(),
            'accs':      accs.item(),
            'time':      time.time() - start_time,
            'loss_before': np.array(loss_before, dtype=np.float32),
            'loss_after':  np.array(loss_after,  dtype=np.float32),
        })
        print(f'{md} inference time: {time.time()-start_time:.2f}s')

    logger.log("sampling complete")


def gaussian_weight(x, y, sigma):
    """Compute 2D Gaussian weight given x,y offsets."""
    return th.exp(-((x**2 + y**2)/(2*sigma**2))).float()


def gaussian_1d(length, sigma):
    """Compute 1D Gaussian kernel of given length and sigma."""
    coord = th.arange(length).float() - (length - 1)/2
    return th.exp(-coord**2/(2*sigma**2))


def create_argparser():
    # Default CLI arguments for in-distribution sampling
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
        global_well_x=150,  # Fixed well position for single-well tests
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
