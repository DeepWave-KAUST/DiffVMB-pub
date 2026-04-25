# train.py
# Author: Shijun Cheng (adapted from OpenAI IDDPM)
# Description: Training script for depth-progressive diffusion model for seismic velocity model building.
#              Leverages OpenAI's IDDPM framework with custom conditioning on well logs and structural constraints.

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
        data_dir="../../../dataset/realmodel4/train/",    # Path to training data
        schedule_sampler="uniform",      # Type of timestep sampler (uniform or loss-aware)
        lr=1e-4,                         # Learning rate
        weight_decay=0.0,                # Weight decay for optimizer
        lr_anneal_steps=0,               # Steps for learning rate annealing
        batch_size=48,                   # Global batch size
        ema_rate="0.999",                # Exponential moving average rate
        log_interval=100,                # Steps between logging to console
        save_interval=10000,             # Steps between saving checkpoints
        resume_checkpoint="",            # Path to pretrained checkpoint
        use_fp16=False,                  # Enable mixed-precision training
        fp16_scale_growth=1e-3,          # Scaling factor growth for mixed precision
        depth_size=32,                   # Patch depth dimension
        wellcond_drop=0.05,              # Drop probability for well constraints
        refcond_drop=0.05,               # Drop probability for structural constraints
    )
    # Append model-specific defaults (e.g., architecture, diffusion steps)
    defaults.update(model_and_diffusion_defaults())

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    # Execute training when run as a script
    main()
