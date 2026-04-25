# unet.py
# Author: Shijun Cheng (adapted from OpenAI IDDPM Unet implementation)
# Description: Defines the U-Net based diffusion model architecture with
#              support for timestep embeddings, positional encodings,
#              conditioning on shallow priors, structural and well data,
#              and multiple attention mechanisms.

import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)


class PositionalEncod(nn.Module):
    """
    Positional encoding for well location embeddings.
    Projects scalar location into sinusoidal features.
    """
    def __init__(self, PosEnc=2, device='cuda'):
        super().__init__()
        self.PEnc = PosEnc
        # Precompute k*pi frequencies for sine/cosine
        self.k_pi_sx = (th.tensor(np.pi)*(2**th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device); 
        self.k_pi_sx = self.k_pi_sx.T

    def forward(self, input):
        # input: [N, 1] tensor of well locations
        tmpsx = th.cat([th.sin(self.k_pi_sx*input[:,0]), th.cos( self.k_pi_sx*input[:,0])], axis=0)
        # Concatenate original input and encoded features
        return th.cat([input, tmpsx.T], dim=-1)


class TimeEmbedding(nn.Module):
    """
    Positional embedding of diffusion timesteps.
    Converts scalar timesteps into sinusoidal embeddings of dimension `dim`.
    """
    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0, "Time embedding dimension must be even"
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        # x: [N] tensor of timesteps
        half = self.dim // 2
        # Compute sin/cos frequencies
        emb = math.log(10000) / (half - 1)
        emb = th.exp(th.arange(half, device=x.device) * -emb)
        # Outer product to get [N, half]
        emb = th.outer(x * self.scale, emb)
        # Concatenate sin and cos
        return th.cat([emb.sin(), emb.cos()], dim=-1)


class TimestepBlock(nn.Module):
    """
    Abstract block interface that supports timestep and condition embeddings.
    """
    @abstractmethod
    def forward(self, x, time_emb, cond_emb, ref, well, well_loc_emb):
        pass


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    Sequential container that passes timestep and conditioning embeddings
    to child layers implementing TimestepBlock.
    """
    def forward(self, x, time_emb, cond_emb, ref, well, well_loc_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb, cond_emb, ref, well, well_loc_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Upsampling by factor 2 with optional convolutional smoothing.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        # Nearest-neighbor upsample
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Downsampling by factor 2 with optional convolutional stride.
    """
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        stride = 2
        if use_conv:
            # Strided conv for learnable downsampling
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            # Average pooling
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block with optional conditioning on:
    - global timestep embedding
    - shallow patch embedding
    - structural constraint embedding
    - well data and well location embedding
    Supports scale-shift normalization (FiLM).
    """
    def __init__(
        self,
        channels,
        time_emb_channels,
        dropout,
        out_channels=None,
        cond_channels=None,
        ref_channels=None,
        well_channels=None,
        wellloc_emb_dim=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_checkpoint = use_checkpoint
        # Input normalization and conv
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        # Timestep embedding transform
        hidden = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
        self.time_emb_layers = nn.Sequential(
            SiLU(),
            linear(time_emb_channels, hidden),
        )
        # Output layers
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
        )
        # Skip connection, conv or identity
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)
        # Embedding layers for conditioning inputs
        if cond_channels is not None:
            self.condtop_emb_conv = nn.Sequential(
                normalization(cond_channels), SiLU(),
                conv_nd(dims, cond_channels, self.out_channels, 3, padding=1),
            )
        if ref_channels is not None:
            self.ref_emb_conv = nn.Sequential(
                normalization(ref_channels), SiLU(),
                conv_nd(dims, ref_channels, self.out_channels, 3, padding=1),
            )
        if well_channels is not None:
            self.well_emb_conv = nn.Sequential(
                normalization(well_channels), SiLU(),
                conv_nd(dims, well_channels, self.out_channels, 3, padding=1),
            )
            # Well location embedding transform
            hidden = 2 * self.out_channels if use_scale_shift_norm else self.out_channels
            self.wellloc_emb_layers = nn.Sequential(
                SiLU(), linear(wellloc_emb_dim, hidden),
            )
            # Fuse well embeddings into features
            self.well_fuse = nn.Sequential(
                normalization(self.out_channels), SiLU(), nn.Dropout(dropout),
                zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)),
            )
        # Final merge conv for all condition channels
        if cond_channels is not None:
            self.allcond_fuse = nn.Sequential(
                normalization(2 * self.out_channels), SiLU(),
                conv_nd(dims, 2 * self.out_channels, self.out_channels, 3, padding=1),
            )

    def forward(self, x, time_emb, cond_emb, ref_emb=None, well_emb=None, well_loc_emb=None):
        # Optionally use checkpointing to save memory
        return checkpoint(self._forward,
                          (x, time_emb, cond_emb, ref_emb, well_emb, well_loc_emb),
                          self.parameters(), self.use_checkpoint)

    def _forward(self, x, time_emb, cond_emb, ref_emb, well_emb, well_loc_emb):
        # Main residual block logic
        h = self.in_layers(x)
        # Process timestep embedding
        te = self.time_emb_layers(time_emb).type(h.dtype)
        # Broadcast to spatial dims
        while te.dim() < h.dim(): te = te[..., None]
        # Apply FiLM or additive conditioning
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(te, 2, dim=1)
            h = self.out_layers[0](h) * (1 + scale) + shift
            h = self.out_layers[1:](h)
        else:
            h = h + te
            h = self.out_layers(h)
        # Add skip connection
        out = self.skip_connection(x) + h
        # Fuse shallow/structural/well conditions
        B, C, H, W = out.shape
        cond_ds = F.adaptive_avg_pool2d(cond_emb, (H, W))
        cond_ds = self.condtop_emb_conv(cond_ds)
        # Add well info if present
        if well_emb is not None:
            well_ds = F.adaptive_avg_pool2d(well_emb, (H, W))
            well_ds = self.well_emb_conv(well_ds)
            wl = self.wellloc_emb_layers(well_loc_emb).type(h.dtype)
            while wl.dim() < out.dim(): wl = wl[..., None]
            if self.use_scale_shift_norm:
                scale, shift = th.chunk(wl, 2, dim=1)
                well_ds = self.well_fuse[0](well_ds) * (1 + scale) + shift
                well_ds = self.well_fuse[1:](well_ds)
            else:
                well_ds = well_ds + wl
                well_ds = self.well_fuse(well_ds)
            cond_ds = cond_ds + well_ds
        # Add structural embed if present
        if ref_emb is not None:
            ref_ds = F.adaptive_avg_pool2d(ref_emb, (H, W))
            ref_ds = self.ref_emb_conv(ref_ds)
            cond_ds = cond_ds + ref_ds
        # Merge outputs and conditions
        fused = th.cat([out, cond_ds], dim=1)
        return self.allcond_fuse(fused)


class AttentionBlock(nn.Module):
    """
    Standard multi-head spatial attention block.
    Allows each position to attend to all others.
    """
    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.num_heads = num_heads
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attn = QKVAttention()
        # Output projection initialized to zero
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))
        self.use_checkpoint = use_checkpoint

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        B, C, *S = x.shape
        x_flat = x.reshape(B, C, -1)
        qkv = self.qkv(self.norm(x_flat))
        # Reshape for multi-head
        qkv = qkv.reshape(B * self.num_heads, -1, qkv.shape[2])
        h = self.attn(qkv)
        h = h.reshape(B, C, -1)
        h = self.proj_out(h)
        return (x_flat + h).reshape(B, C, *S)


class QKVAttention(nn.Module):
    """
    Computes Q-K-V attention: softmax(QK^T)V.
    """
    def forward(self, qkv):
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        w = th.einsum('bct,bcs->bts', q * scale, k * scale)
        w = th.softmax(w.float(), dim=-1).type(w.dtype)
        return th.einsum('bts,bcs->bct', w, v)


class CrossEfficientAttention(nn.Module):
    """
    Cross-attention with linear complexity
    between x (query) and cproj (key/value source).
    """
    def __init__(self, dims, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        assert channels % num_heads == 0
        self.scale = 1 / math.sqrt(math.sqrt(channels))
        self.to_q = conv_nd(dims, channels, channels, 1)
        self.to_kv = conv_nd(dims, channels, channels*2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

    def forward(self, x, cproj):
        return checkpoint(self._forward, (x, cproj), self.parameters(), self.use_checkpoint)

    def _forward(self, x, cproj):
        B, C, H, W = x.shape
        N = H * W
        Q = self.to_q(x).reshape(B, self.num_heads, C//self.num_heads, N)
        KV = self.to_kv(cproj).reshape(B, 2, self.num_heads, C//self.num_heads, N)
        K, V = KV[:,0], KV[:,1]
        # Softmax across dims
        q = F.softmax(Q * self.scale, dim=2)
        k = F.softmax(K, dim=3)
        # Efficient attention core
        context = th.bmm(k.reshape(B, -1, N), V.reshape(B, -1, N).transpose(1,2))
        out = th.bmm(context, q.reshape(B, -1, N))
        out = out.reshape(B, C, H, W)
        return self.proj_out(out)


class LinearAttention(nn.Module):
    """
    Memory-efficient linearized attention: uses feature map phi(x)=elu(x)+1 to approximate softmax.
    """
    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, Q, K, V):
        # Q, K: [B, N, C], V: [B, N, C]
        phi = lambda x: F.elu(x) + 1
        Q_phi = phi(Q)  # [B, N, C]
        K_phi = phi(K)  # [B, N, C]
        # compute KV^T: [B, C, C]
        KV = torch.einsum('bnc,bnd->bcd', K_phi, V)
        # normalization: compute Z = (Q_phi * sum K_phi)^-1
        denom = torch.einsum('bnc,bnc->bn', Q_phi, K_phi.sum(dim=1, keepdim=True).expand_as(Q_phi))
        Z = 1.0 / (denom + self.eps)
        # attend: [B, N, C] = Q_phi @ KV
        out = torch.einsum('bnc,bcd->bnd', Q_phi, KV)
        # apply normalization
        out = out * Z.unsqueeze(-1)
        return out

class UNetModel(nn.Module):
    """
    Full U-Net combining:
      - Timestep embeddings
      - Shallow-prior conditioning
      - Structural & well data conditioning
      - Downsampling & upsampling paths
      - Attention at specified resolutions
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        time_emb_scale=1.0,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # Pad input so H,W divisible by this
        self.padder_size = 2 ** len(channel_mult)

        # 1) Timestep embedding MLP: expands scalar t -> rich embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # 2) Input conditioning:
        #    shallow-prior patch (cond_embed), structural image (ref_embed), well data (well_embed)
        self.cond_embed = nn.Sequential(
            conv_nd(dims, 3, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        self.ref_embed = nn.Sequential(
            conv_nd(dims, 1, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        self.well_embed = nn.Sequential(
            conv_nd(dims, 1, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        # Positional encoding for well location
        wellloc_embed_dim = model_channels * 4
        self.wellloc_embed = nn.Sequential(
            PositionalEncod(model_channels // 2),
            linear(model_channels + 1, wellloc_embed_dim),
            SiLU(),
            linear(wellloc_embed_dim, wellloc_embed_dim),
        )

        # Optional class label embedding
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 3) Initial conv: concatenate input and shallow prior
        self.inp = conv_nd(dims, in_channels + model_channels, model_channels, 3, padding=1)

        # 4) Downsampling path: stacks of ResBlocks (+Attention) and Downsample
        self.downs = nn.ModuleList([])
        encoder_channels = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        cond_channels=model_channels,
                        ref_channels=model_channels,
                        well_channels=model_channels,
                        wellloc_emb_dim=wellloc_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.downs.append(TimestepEmbedSequential(*layers))
                encoder_channels.append(ch)
            if level != len(channel_mult) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                encoder_channels.append(ch)
                ds *= 2

        # 5) Bottleneck (middle) blocks
        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                ref_channels=model_channels,
                well_channels=model_channels,
                wellloc_emb_dim=wellloc_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                ref_channels=model_channels,
                well_channels=model_channels,
                wellloc_emb_dim=wellloc_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # 6) Upsampling path: mirror of downsampling with ResBlocks, Attention, Upsample
        self.ups = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + encoder_channels.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        cond_channels=model_channels,
                        ref_channels=model_channels,
                        well_channels=model_channels,
                        wellloc_emb_dim=wellloc_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(*layers))

        # 7) Final normalization + activation + conv to produce output
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.downs.apply(convert_module_to_f16)
        self.middle.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.downs.apply(convert_module_to_f32)
        self.middle.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.downs.parameters()).dtype

    def forward(self, inp, cond_top, ref, well, well_loc, timesteps, y=None):
        """
        Forward pass of U-Net:
        1) Pad inputs to valid size
        2) Embed shallow prior, structural image, well data, and timesteps
        3) Optional label embedding for class-conditional
        4) Initial conv and downsampling path with skip connections
        5) Bottleneck processing
        6) Upsampling path combining skips
        7) Final conv to output predicted noise or x0
        """
        # Record original spatial size for cropping
        b, c, h, w = inp.shape
        # 1) Ensure divisible by padder_size
        inp = self.check_image_size(inp)
        cond_top = self.check_image_size(cond_top)
        # 2) Compute shallow prior embedding
        cond_top = self.cond_embed(cond_top)
        # Concatenate input and cond
        x = th.cat([inp, cond_top], dim=1)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # 3) Timestep embedding
        time_emb = self.time_embed(timesteps)

        # 4) Well & structural embeddings if provided
        if well is not None:
            well = self.well_embed(well)
            well_loc = self.wellloc_embed(well_loc)
        if ref is not None:
            ref = self.ref_embed(ref)
        # 5) Class label embedding
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            time_emb = time_emb + self.label_emb(y)

        # 6) U-Net encoding
        skips = []
        x = x.type(self.inner_dtype)
        x = self.inp(x)
        skips.append(x)

        for module in self.downs:
            x = module(x, time_emb, cond_top, ref, well, well_loc)
            skips.append(x)
        # 7) Bottleneck
        x = self.middle(x, time_emb, cond_top, ref, well, well_loc)
        # 8) Decoding
        for module in self.ups:
            cat_in = th.cat([x, skips.pop()], dim=1)
            x = module(cat_in, time_emb, cond_top, ref, well, well_loc)
        # 9) Final conv and crop to original size
        x = x.type(inp.dtype)
        x = self.out(x)
        return x[:, :, :h, :w]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x
