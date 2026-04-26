# script_util.py
# Author: Shijun Cheng (adapted from OpenAI IDDPM)
# Description: Utility functions for configuring and instantiating the diffusion model,
#              including default hyperparameters, model creation, diffusion scheduler,
#              and argument parsing helpers.

import argparse
import inspect
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

# Number of classes for class-conditional models (unused for unconditional)
NUM_CLASSES = 1000

def model_and_diffusion_defaults():
    """
    Return the default configuration dict for model and diffusion hyperparameters.
    These defaults cover channel sizes, network depth, attention settings,
    diffusion steps, and related flags.
    """
    return dict(
        in_channels=1,                        # number of input channels (1 for single-component velocity)
        num_channels=64,                      # base channel width of the U-Net
        out_channels=1,                       # number of output channels (matches in_channels when learn_sigma=False)
        channel_mult=(1, 2, 4, 8, 16),        # per-resolution channel multipliers for the U-Net encoder/decoder
        num_res_blocks=2,                     # number of residual blocks per resolution level
        num_heads=4,                          # number of attention heads in each multi-head self-attention layer
        num_heads_upsample=-1,                # attention heads in the decoder; -1 inherits num_heads
        attention_resolutions=(8, 16),        # spatial resolutions (in samples) at which self-attention is applied
        dropout=0.0,                          # dropout probability; 0.0 disables dropout
        learn_sigma=False,                    # if True, the network also predicts the diffusion variance
        sigma_small=False,                    # if True, use the smaller fixed variance schedule (β̃_t instead of β_t)
        class_cond=False,                     # if True, condition the model on class labels
        diffusion_steps=1000,                 # total number of forward diffusion timesteps T
        noise_schedule="cosine",              # beta schedule type: "cosine" or "linear"
        timestep_respacing="",                # timestep respacing string:
                                              #   - set to "" during training (no respacing, all T steps used)
                                              #   - set to "ddim{N}" during DDIM inference, where N is the
                                              #     number of denoising steps after respacing (e.g. "ddim10"
                                              #     for 10 steps); also used as the output sub-folder name
        use_kl=False,                         # if True, optimize the variational lower-bound (KL loss) instead of MSE
        predict_xstart=True,                  # if True, the network predicts x_0 directly; otherwise predicts noise ε
        rescale_timesteps=True,               # if True, rescale integer timestep indices to [0, 1000] range
        rescale_learned_sigmas=False,         # if True, apply rescaling to the learned sigma outputs
        use_checkpoint=False,                 # if True, use gradient checkpointing to reduce GPU memory usage
        use_scale_shift_norm=True,            # if True, use AdaGN (scale-and-shift group norm) for time conditioning
    )

def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
):
    """
    Instantiate the UNetModel and the corresponding SpacedDiffusion process.
    Combines model architecture defaults with diffusion scheduler parameters.
    """
    # Create the neural network model
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
    )
    # Create the diffusion scheduler and noise process
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
):
    """
    Helper to instantiate the U-Net architecture with given hyperparameters.
    """
    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Configure and return a SpacedDiffusion instance based on provided schedules.
    - betas: noise schedule array
    - loss_type: one of MSE, RESCALED_MSE, or RESCALED_KL
    - timestep spacing: allows DDIM-style accelerated sampling
    """
    # Obtain beta schedule (variance) for forward diffusion
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # Determine the loss function type
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE

    # If no respacing provided, use full sequence
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL
            ) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add each key/value in default_dict to the argparse parser as --key.
    Automatically infers argument type and handles booleans.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of attributes from args into a dict for easy function calls.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Convert common string inputs to boolean for argparse.
    Accepted true values: yes, true, t, y, 1
    Accepted false values: no, false, f, n, 0
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
