# train.py
# Author: Shijun Cheng (sjcheng.academic@gmail.com)
#
# Description:
#   Training script for Part I of the depth-progressive diffusion manuscript.
#
#   In this part, structural constraints are derived from a reflectivity model
#   computed directly from the true velocity model, serving as an idealized
#   proxy for migration-based structural images. This setting isolates the
#   contribution of the generative framework itself before introducing the
#   complexities of field-data constraints addressed in Part II.
#
#   To enable joint unconditional and conditional generation within a single
#   model (classifier-free guidance), well-log constraints and structural
#   (reflectivity) constraints are each randomly dropped with independent
#   probabilities (wellcond_drop and refcond_drop) during training. This
#   allows the model to learn both the unconditional velocity distribution
#   and its conditional variants from a single training run, without
#   requiring separate models.
#
#   Training data consists of 2-D cross-sections extracted from a large
#   collection of industrial velocity models, providing broad coverage
#   of geologically realistic velocity structures.
#
#   This script is adapted from the OpenAI IDDPM framework and extended with
#   custom multi-condition inputs (shallow velocity context, depth positional
#   encoding, well logs, and reflectivity).
#
# Usage:
#   python train.py [--data_dir PATH] [--batch_size N] [--lr LR] ...
#   Run with --help to see all available arguments and their defaults.

import argparse
import torch as th

from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import os

def main():
    """
    Main entry point for training the diffusion model.
    Parses arguments, sets up model, data loaders, and starts the training loop.
    """
    # Parse command-line arguments
    args = create_argparser().parse_args()

    # Initialize logging
    logger.configure()

    # Select compute device (use GPU if available)
    device = th.device('cuda') if th.cuda.is_available() else th.device('cpu')
    logger.log(f"Using device: {device}")

    logger.log("Creating model and diffusion process...")
    # Prepare model and diffusion hyperparameters from args
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    # Instantiate the neural network and diffusion scheduler
    model, diffusion = create_model_and_diffusion(
        **params,
    )

    # Directory for saving checkpoints
    os.makedirs(args.dir_cp, exist_ok=True)

    # Load pretrained weights if provided
    # logger.log(f"Loading pretrained model...")
    # pretrained_dict = th.load('../trained_model/ema_0.999_400000.pt', map_location=device)
    # model.load_state_dict(pretrained_dict)

    # Move model to the selected device
    model.to(device)

    # Create schedule sampler for timesteps in diffusion
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("Setting up data loader...")
    # Load seismic velocity dataset with shallow/deep patches and conditioning
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        depth_size=args.depth_size,
        vmax=args.vmax,
        vmin=args.vmin,
        device=device,
        class_cond=args.class_cond,
    )

    logger.log("Starting training loop...")
    # Initialize and run the training loop
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        dir_cp=args.dir_cp,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        wellcond_drop=args.wellcond_drop,
        refcond_drop=args.refcond_drop,
    ).run_loop()


def create_argparser():
    """
    Defines and returns the command-line argument parser with default values.
    """
    # Default hyperparameters and paths
    defaults = dict(
        data_dir="../dataset/part1/train/",    # Path to training data
        schedule_sampler="uniform",      # Type of timestep sampler (uniform or loss-aware)
        lr=1e-4,                         # Learning rate
        weight_decay=0.0,                # Weight decay for optimizer
        lr_anneal_steps=0,               # Steps for learning rate annealing
        batch_size=48,                   # Global batch size
        ema_rate="0.999",                # Exponential moving average rate
        log_interval=100,                # Steps between logging to console
        save_interval=10000,             # Steps between saving checkpoints
        resume_checkpoint="",            # Path to pretrained checkpoint
        dir_cp='./checkpoints_part1/',   # Path to save checkpoint
        use_fp16=False,                  # Enable mixed-precision training
        fp16_scale_growth=1e-3,          # Scaling factor growth for mixed precision
        depth_size=32,                   # Patch depth dimension
        wellcond_drop=0.05,              # Drop probability for well constraints
        refcond_drop=0.05,               # Drop probability for structural constraints
        vmax=5000,                       # Maximum velocity for normalization (m/s)
        vmin=1400,                       # Minimum velocity for normalization (m/s)
    )
    # Append model-specific defaults (e.g., architecture, diffusion steps)
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Execute training when run as a script
    main()
