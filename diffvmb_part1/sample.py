# sample.py
# Author: Shijun Cheng (sjcheng.academic@gmail.com)
#
# Description:
#   Sampling script for Part I of the depth-progressive velocity model building
#   manuscript.
#
#   This script implements the inference pipeline for conditional velocity model
#   generation, proceeding from shallow to deep in a depth-progressive manner.
#   Structural constraints are provided via a reflectivity model derived from
#   the true velocity, which serves as an idealized proxy for migration-based
#   structural images available in practice.
#
#   Evaluation is performed on four benchmark models:
#     In-distribution models:
#       - SEAM Arid   -- complex near-surface with strong lateral variation
#       - SEG/EAGE    -- salt-body model with high-velocity contrasts
#       - Overthrust  -- thrust-belt model with folded reflectors
#     Out-of-distribution model:
#       - Marmousi    -- geologically complex model used to assess generalization
#
#   For each model, the script supports four conditioning configurations by
#   toggling well-log constraints (use_well) and reflectivity constraints
#   (use_ref), and optionally applies gradient-based well guidance during
#   reverse diffusion. Results (ensemble mean, uncertainty map, and individual
#   samples) are saved as .mat files for further analysis.
#
# Usage:
#   python sample.py [--model_path PATH] [--batch_size N] [--use_ddim] ...
#   Run with --help to see all available arguments and their defaults.
#   NOTE: when use_ddim=True, set --timestep_respacing to "ddim{N}" (e.g.
#   "ddim10") to specify the number of DDIM denoising steps.

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from code.datasets import (
    normalizer_vel,
    denormalizer_vel,
    normalizer_depth,
    normalizer_well_loc,
    ricker_wavelet,
    convolve_wavelet,
)
import scipy.io as sio
from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


def main(use_well, use_ref):
    """
    Run depth-progressive diffusion sampling on a set of benchmark velocity
    models and save the predictions to .mat files.

    :param use_well:  if True, condition the reverse diffusion on a sparse
                      well-log velocity constraint and apply gradient-based
                      well guidance at each sampling step.
    :param use_ref:   if True, condition the reverse diffusion on the
                      reflectivity (structural) section at each depth window.
    """
    args = create_argparser().parse_args()

    device = th.device('cuda')

    logger.configure()

    # ── Model and diffusion setup ──────────────────────────────────────────────
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**params)

    # Load pre-trained weights and switch to evaluation mode.
    model.load_state_dict(th.load(f'{args.model_path}', map_location=device))
    model.to(device=device)
    model.eval()

    # Select the sampling function: full DDPM chain or accelerated DDIM.
    # NOTE: when use_ddim=True, the timestep respacing string must be set in
    # script_util.py (or passed via --timestep_respacing on the command line).
    # Use the format "ddim{N}" where N is the number of DDIM steps after
    # respacing, e.g. "ddim10" for 10 denoising steps. This value is also
    # used as the output sub-folder name under ./output/.
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    # ── Output directory ───────────────────────────────────────────────────────
    # Use a dedicated sub-folder for DDPM vs. DDIM (named by respacing factor).
    if not args.use_ddim:
        dir_output = './output/ddpm/'
    else:
        dir_output = f'./output/{args.timestep_respacing}/'
    os.makedirs(dir_output, exist_ok=True)

    # L1 loss used to evaluate the quality of the final prediction.
    criterion = th.nn.L1Loss()

    # ── Gaussian blending weights along the depth axis ─────────────────────────
    # Overlapping depth windows are blended with a 1-D Gaussian weight to avoid
    # hard boundary artifacts where adjacent windows meet.
    step_size_z = args.depth_size // 2          # overlap stride between consecutive windows
    sigma_z     = args.depth_size // 4          # standard deviation of the Gaussian taper
    z = np.arange(args.depth_size) - args.depth_size // 2  # centered depth indices
    gaussian1d_z_ori = np.exp(-(z ** 2) / (2 * sigma_z ** 2))  # shape (depth_size,)
    gaussian1d_z_ori = (
        th.tensor(gaussian1d_z_ori, dtype=th.float32)
        .to(device)
        .view(args.depth_size, 1)   # column vector for broadcasting over nx
    )

    # ── Benchmark models ───────────────────────────────────────────────────────
    model_list = ['SEAMArid', 'Overthrust', 'SEGEAGE', 'Marmousi']

    for idx, md in enumerate(model_list):
        print(
            f'Sampling start for model {md} with batch size {args.batch_size} '
            f'usewell {use_well} useref {use_ref}'
        )

        # ── Load ground-truth velocity and reflectivity ────────────────────────
        dict_ = sio.loadmat(f'../dataset/part1/test/{md}_vel.mat')
        vp  = dict_['vel']   # P-wave velocity model, shape (nz, nx)
        ref = dict_['ref']   # reflectivity section,  shape (nz, nx)

        # Normalize velocity to [-1, 1] (consistent with training normalization).
        vp = normalizer_vel(vp, dmin=args.vmin, dmax=args.vmax)
        nz, nx = vp.shape

        # Convert to torch tensors with shape (1, 1, nz, nx) on the GPU.
        vp    = th.tensor(vp,  dtype=th.float32).unsqueeze(0).unsqueeze(1).to(device)
        struc = th.tensor(ref, dtype=th.float32).unsqueeze(0).unsqueeze(1).to(device)

        # Replicate along the batch dimension for ensemble sampling.
        vp    = vp.repeat(args.batch_size,    1, 1, 1)   # (B, 1, nz, nx)
        struc = struc.repeat(args.batch_size, 1, 1, 1)   # (B, 1, nz, nx)

        # ── Well-log configuration ─────────────────────────────────────────────
        # Fix a single borehole at trace index 50 (arbitrary; can be changed).
        well_loc_index = 50
        well_loc_np = np.expand_dims(
            np.array(well_loc_index, dtype=np.float32), axis=0
        )
        # Normalize the well's lateral position to [-1, 1].
        well_loc_np = normalizer_well_loc(well_loc_np, dmax=nx)

        # ── Accumulation buffers for depth-progressive blending ────────────────
        # Lateral Gaussian weight tiled to full model width.
        gaussian1d_z = gaussian1d_z_ori.repeat(1, nx)   # (depth_size, nx)

        # Buffers for the ensemble-mean and ensemble-std maps (physical units).
        accumulated_mean   = th.zeros((nz, nx), dtype=th.float32).to(device)
        accumulated_std    = th.zeros((nz, nx), dtype=th.float32).to(device)
        accumulated_weight = th.zeros((nz, nx), dtype=th.float32).to(device)

        # Batch-level buffers for storing all individual samples (normalized).
        gaussian1d_z_batch       = gaussian1d_z_ori.repeat(args.batch_size, 1, nx)  # (B, depth_size, nx)
        accumulated_sample       = th.zeros((args.batch_size, nz, nx), dtype=th.float32).to(device)
        accumulated_weight_batch = th.zeros((args.batch_size, nz, nx), dtype=th.float32).to(device)

        # ── Initialise with the known shallow (top) window ─────────────────────
        # The first depth_size rows are assumed observed; use them as the
        # starting conditioning for the depth-progressive loop.
        vp_top = vp[:, :, :args.depth_size]   # (B, 1, depth_size, nx)

        # Accumulate the top window into the mean and per-sample buffers.
        accumulated_mean[:args.depth_size, :]   += gaussian1d_z * denormalizer_vel(vp_top[0].squeeze(), dmin=args.vmin, dmax=args.vmax)
        accumulated_weight[:args.depth_size, :] += gaussian1d_z

        accumulated_sample[:, :args.depth_size, :]       += gaussian1d_z_batch * vp_top.squeeze()
        accumulated_weight_batch[:, :args.depth_size, :] += gaussian1d_z_batch

        # ── Prepare well-location tensor ───────────────────────────────────────
        well_loc_tensor = (
            th.tensor(well_loc_np, dtype=th.float32).to(device).repeat(args.batch_size, 1)
            if use_well else None
        )
        use_wellguide = use_well   # enable gradient-based well guidance iff well data is used

        # ── Depth window index list ────────────────────────────────────────────
        # Generate starting row indices for each sliding window, stepping by
        # step_size_z. Append the last valid index if it is not already included.
        indices_z = list(range(step_size_z, nz - args.depth_size + 1, step_size_z))
        if indices_z[-1] != nz - args.depth_size:
            indices_z.append(nz - args.depth_size)

        # ── Depth-progressive sampling loop ────────────────────────────────────
        for iz in indices_z:
            print(f'sampling depth grid index {iz}')

            # Ground-truth bottom window (used only for well conditioning).
            vp_bottom = vp[:, :, iz:iz + args.depth_size]   # (B, 1, depth_size, nx)

            # ── Build conditioning tensor cond_top ─────────────────────────────
            # cond_top has 3 channels:
            #   ch 0 – vp_top:   shallow velocity context (previous window output)
            #   ch 1 – depth encoding of the top window rows
            #   ch 2 – depth encoding of the bottom window rows
            depth_top = th.arange(
                iz - step_size_z, args.depth_size + iz - step_size_z, device=device
            ).unsqueeze(1)                       # (depth_size, 1)
            depth_top    = depth_top.repeat(1, 1, 1, nx)      # (depth_size, 1, 1, nx)
            depth_bottom = depth_top + step_size_z
            depth_top    = normalizer_depth(depth_top,    dmax=nz)
            depth_bottom = normalizer_depth(depth_bottom, dmax=nz)

            depth_top    = depth_top.repeat(args.batch_size,    1, 1, 1)
            depth_bottom = depth_bottom.repeat(args.batch_size, 1, 1, 1)

            cond_top = th.cat([vp_top, depth_top, depth_bottom], axis=1).float()   # (B, 3, depth_size, nx)

            # ── Well-log condition ─────────────────────────────────────────────
            if use_well:
                # Option 1 (active): sparse column mask — only the borehole
                # column carries the true velocity; all other columns are zero.
                mask = th.zeros_like(vp_bottom)
                mask[:, :, :, well_loc_index:well_loc_index + 1] = 1
                well_bottom = mask * vp_bottom   # (B, 1, depth_size, nx)

                # Option 2 (alternative): broadcast the single borehole trace
                # to all lateral positions so every trace sees the well.
                # well_bottom = vp_bottom[:, :, :, well_loc_index:well_loc_index+1]
                # well_bottom = well_bottom.repeat(1, 1, 1, nx)
            else:
                well_bottom = None

            # ── Reflectivity (structure) condition ────────────────────────────
            struc_bottom = struc[:, :, iz:iz + args.depth_size] if use_ref else None

            # ── Run diffusion sampling ─────────────────────────────────────────
            # sample: (B, 1, depth_size, nx) normalized predicted velocity patch
            sample, _, _, loss_before, loss_after = sample_fn(
                model,
                cond_top, struc_bottom, well_bottom, well_loc_tensor, well_loc_index,
                (args.batch_size, args.out_channels, args.depth_size, nx),
                scale_factor=args.scale_factor if use_wellguide else None,
                clip_denoised=args.clip_denoised,
            )

            # Feed the current prediction forward as the top-window context
            # for the next (deeper) depth window.
            vp_top = sample.clone()

            # ── Gaussian-weighted accumulation ─────────────────────────────────
            # Blend the current window into the full-model buffers using the
            # 1-D Gaussian taper so overlapping regions are smoothly merged.
            accumulated_mean[iz:iz + args.depth_size]   += gaussian1d_z * denormalizer_vel(sample.squeeze(), dmin=args.vmin, dmax=args.vmax).mean(dim=0)
            accumulated_std[iz:iz + args.depth_size]    += gaussian1d_z * denormalizer_vel(sample.squeeze(), dmin=args.vmin, dmax=args.vmax).std(dim=0)
            accumulated_weight[iz:iz + args.depth_size] += gaussian1d_z

            accumulated_sample[:, iz:iz + args.depth_size]       += gaussian1d_z_batch * sample.squeeze()
            accumulated_weight_batch[:, iz:iz + args.depth_size] += gaussian1d_z_batch

        # ── Aggregate results ──────────────────────────────────────────────────
        # Divide by accumulated weights to obtain the blended ensemble statistics.
        final_pred_mean = accumulated_mean / accumulated_weight           # (nz, nx), physical units
        final_pred_std  = accumulated_std  / accumulated_weight           # (nz, nx), physical units

        # Per-sample full-model predictions in physical units.
        final_sample = denormalizer_vel(accumulated_sample / accumulated_weight_batch, dmin=args.vmin, dmax=args.vmax)  # (B, nz, nx)

        # L1 accuracy of the ensemble mean vs. the ground-truth velocity.
        with th.no_grad():
            accs = criterion(final_pred_mean, denormalizer_vel(vp[0].squeeze(), dmin=args.vmin, dmax=args.vmax))

        # ── Save outputs ───────────────────────────────────────────────────────
        # Include scale_factor in the filename only when well guidance is active,
        # since it has no effect otherwise.
        if use_wellguide:
            file_name = (
                f'{dir_output}{md}_batch{args.batch_size}_usewell{use_well}'
                f'_useref{use_ref}_wellguide{use_wellguide}_scale{args.scale_factor}_out.mat'
            )
        else:
            file_name = (
                f'{dir_output}{md}_batch{args.batch_size}_usewell{use_well}'
                f'_useref{use_ref}_wellguide{use_wellguide}_out.mat'
            )

        sio.savemat(
            file_name,
            {
                'pred_mean':   final_pred_mean.squeeze().cpu().numpy(),  # ensemble mean velocity map
                'pred_std':    final_pred_std.squeeze().cpu().numpy(),   # ensemble std (uncertainty) map
                'sample':      final_sample.squeeze().cpu().numpy(),     # all B individual samples
                'accs':        accs.item(),                              # L1 error vs. ground truth
                'loss_before': np.array(loss_before, dtype=np.float32),  # per-step well MSE before guidance
                'loss_after':  np.array(loss_after,  dtype=np.float32),  # per-step well MSE after guidance
            }
        )

    logger.log("sampling complete")


def create_argparser():
    """
    Build the argument parser with default hyper-parameters for inference.
    These defaults can be overridden from the command line.
    """
    defaults = dict(
        clip_denoised=True,          # clip x_0 predictions to [-1, 1] during sampling
        use_ddim=True,               # use DDIM instead of full DDPM chain
        batch_size=50,               # number of independent samples per depth window
        model_path="../trained_model/model_part1.pt",
        depth_size=32,               # height of each depth window in samples
        scale_factor=10,             # well-guidance gradient step scale
        vmax=5000,                       # Maximum velocity for normalization (m/s)
        vmin=1400,                       # Minimum velocity for normalization (m/s)
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Run inference for all four combinations of well and reflectivity conditioning.
    use_well_list = [True, False]
    use_ref_list  = [True, False]
    for use_well in use_well_list:
        for use_ref in use_ref_list:
            main(use_well, use_ref)