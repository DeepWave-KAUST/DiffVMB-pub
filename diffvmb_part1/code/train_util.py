# train_util.py
# Author: Shijun Cheng (adapted from OpenAI IDDPM)
# Description: Training loop and utilities for depth-progressive diffusion model.
#              Handles distributed setup, mixed-precision, EMA, checkpointing, and logging.

import copy
import functools
import os
import time
import random

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# Initial loss scaling for mixed-precision training
INITIAL_LOG_LOSS_SCALE = 20.0

class TrainLoop:
    """
    Core training loop for diffusion model.

    - Manages data iteration, forward/backward, optimization, FP16, EMA, and checkpointing.
    - Supports optional dropping of well/structural conditioning data.
    """
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        dir_cp,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        wellcond_drop=0,
        refcond_drop=0,
    ):
        # Store hyperparameters and modules
        self.model = model
        self.device = next(model.parameters()).device
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        # Allow multiple EMA rates
        self.ema_rate = (
            [ema_rate] if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.dir_cp = dir_cp
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        # Sampler for selecting diffusion timesteps
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        # Probability of dropping conditioning during training
        self.wellcond_drop = wellcond_drop
        self.refcond_drop = refcond_drop

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        # Prepare model parameters and FP16 master copy if needed
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Load checkpoints and initialize optimizer/EMA
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        self.opt = AdamW(
            self.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )
        # If resuming, restore optimizer and EMA states
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Initialize EMA parameters as copies of model weights
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in self.ema_rate
            ]

    def _load_and_sync_parameters(self):
        """
        Load model checkpoint if specified and set resume_step.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"Loading model from {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(resume_checkpoint, map_location=self.device)
            )

    def _load_ema_parameters(self, rate):
        """
        Load EMA parameters from checkpoint corresponding to a given rate.
        """
        ema_params = copy.deepcopy(self.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"Loading EMA from {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location=self.device)
            ema_params = self._state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        """
        Restore optimizer state from a checkpoint if available.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06d}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"Loading optimizer state from {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location=self.device)
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Convert model to FP16 and create master FP32 parameter copy.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        Main training loop: iterate over batches, run steps, log, and save checkpoints.
        """
        while (not self.lr_anneal_steps or
               self.step + self.resume_step < self.lr_anneal_steps):
            batch = next(self.data)
            self.run_step(*batch)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Early exit for integration tests
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save final checkpoint if needed
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_vp, batch_cond_top, batch_struc, batch_well, batch_well_loc, cond):
        """
        Execute one training iteration: forward/backward + optimization + logging.
        """
        self.forward_backward(batch_vp, batch_cond_top, batch_struc,
                              batch_well, batch_well_loc, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()
        self.log_step()

    def forward_backward(self, batch_vp, batch_cond_top,
                         batch_struc, batch_well, batch_well_loc, cond):
        """
        Compute loss for a batch and perform backward pass. Supports random dropping
        of well/structural conditions to improve robustness.
        """
        zero_grad(self.model_params)
        # Move inputs to device
        batch_vp = batch_vp.to(self.device)
        batch_cond_top = batch_cond_top.to(self.device)
        batch_struc = batch_struc.to(self.device)
        batch_well = batch_well.to(self.device)
        batch_well_loc = batch_well_loc.to(self.device)

        # Randomly drop conditioning channels
        if random.random() < self.wellcond_drop:
            batch_well = None
            batch_well_loc = None
        if random.random() < self.refcond_drop:
            batch_struc = None

        # Sample diffusion timesteps and weights
        t, weights = self.schedule_sampler.sample(
            batch_vp.shape[0], self.device
        )
        # Define loss function partial
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.model,
            batch_vp,
            batch_cond_top,
            batch_struc,
            batch_well,
            batch_well_loc,
            t,
            model_kwargs=cond,
        )
        losses = compute_losses()
        # Update sampler for loss-aware sampling
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        # Weighted average loss
        loss = (losses["loss"] * weights).mean()
        log_loss_dict(
            self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )
        # Backpropagate with optional FP16 scaling
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize_fp16(self):
        """
        FP16 optimization: unscale grads, update master, apply EMA, re-sync.
        """
        # Detect NaNs and adjust loss scale
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return
        # Transfer gradients and step optimizer
        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update EMAs
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Standard FP32 optimization and EMA update.
        """
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """
        Compute and log the L2 norm of gradients across master params.
        """
        sqsum = 0.0
        for p in self.master_params:
            if p.grad is None:
                continue
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Linearly anneal learning rate if lr_anneal_steps is set.
        """
        if not self.lr_anneal_steps:
            return
        frac_done = 0.5 * (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """
        Log training step, sample count, and FP16 loss scale.
        """
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save model EMA checkpoints at specified intervals.
        """
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"Saving EMA@{rate} checkpoint...")
            filename = (f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                        if rate else
                        f"model{(self.step+self.resume_step):06d}.pt")
            with bf.BlobFile(bf.join(self.dir_cp, filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert master_params back to model state_dict format for saving.
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _) in enumerate(self.model.named_parameters()):
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Load state_dict into master_params list for FP16 resumption.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        return make_master_params(params) if self.use_fp16 else params

# Utility functions for checkpoint naming and discovery

def parse_resume_step_from_filename(filename):
    # Extract step number from 'modelNNNNNN.pt'
    split = filename.split("model")
    if len(split) < 2:
        return 0
    num = split[-1].split(".")[0]
    try:
        return int(num)
    except ValueError:
        return 0

def parse_dataname_from_filename(filename):
    # Custom parser (unused) based on 'gaussian5<name>' pattern
    split = filename.split("gaussian5")
    if len(split) < 2:
        return 0
    return split[-1].split(".")[0]

def get_blob_logdir():
    # Directory for blob logging, fallback to local logger dir
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def find_resume_checkpoint():
    # Override to auto-discover blob checkpoints; defaults to None
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    # Locate EMA checkpoint file next to main checkpoint
    if main_checkpoint is None: return None
    filename = f"ema_{rate}_{step:06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    return path if bf.exists(path) else None


def log_loss_dict(diffusion, ts, losses):
    """
    Log loss values and their quartiles across diffusion timesteps.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
