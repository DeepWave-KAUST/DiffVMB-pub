import numpy as np
import torch as th
from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps:  the number of diffusion steps in the original
                           process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper (e.g. "ddim10" for 10 denoising steps).
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # DDIM striding: find an integer stride such that stepping through
            # [0, num_timesteps) with that stride yields exactly the requested
            # number of steps.
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        # Parse comma-separated section counts from the string.
        section_counts = [int(x) for x in section_counts.split(",")]

    # Divide the full timestep range into len(section_counts) equal-ish sections.
    # Any remainder steps are distributed one-per-section from the front.
    size_per = num_timesteps // len(section_counts)
    extra    = num_timesteps % len(section_counts)

    start_idx = 0
    all_steps = []

    for i, section_count in enumerate(section_counts):
        # Actual size of this section (may be one larger if it receives a remainder step).
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )

        # Compute the fractional stride needed to space section_count steps
        # evenly across this section.
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)

        # Walk through the section and collect the selected timestep indices.
        cur_idx    = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride

        all_steps += taken_steps
        start_idx  += size

    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that skips steps from an underlying base diffusion
    process, enabling accelerated sampling (e.g. DDIM) while reusing the
    same trained model.

    The key idea is to derive a new, shorter beta schedule from the subset of
    selected timesteps: for each retained timestep i, the new beta is computed
    as 1 - ᾱ_i / ᾱ_{prev}, where ᾱ_{prev} is the cumulative product at the
    most recently retained step. This ensures the new schedule's forward
    process remains mathematically consistent with the original.

    A timestep_map is maintained to translate the spaced indices (0, 1, 2, ...)
    back to their original positions in the base schedule, so the pre-trained
    model always receives the correct timestep embedding.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs:        keyword arguments forwarded to GaussianDiffusion,
                          including the original 'betas' array which is
                          replaced internally by the derived spaced betas.
    """

    def __init__(self, use_timesteps, **kwargs):
        # Store the selected timestep indices as a set for O(1) lookup.
        self.use_timesteps = set(use_timesteps)
        # Maps spaced index → original index (e.g. timestep_map[3] = 150
        # means the 4th spaced step corresponds to original step 150).
        self.timestep_map = []
        self.original_num_steps = len(kwargs["betas"])

        # Build the base diffusion solely to access its alphas_cumprod array.
        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa

        # Derive the new beta schedule for the spaced timesteps.
        # Each new beta satisfies: ᾱ_new = ᾱ_orig, so the forward process
        # q(x_t | x_0) is unchanged at the retained timesteps.
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # Replace the original betas with the derived spaced betas and
        # initialize the parent GaussianDiffusion with the new schedule.
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(self, model, *args, **kwargs):
        # pylint: disable=signature-differs
        # Wrap the model so it receives original-scale timestep embeddings
        # rather than the spaced indices before computing the posterior.
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(self, model, *args, **kwargs):
        # pylint: disable=signature-differs
        # Same wrapping as p_mean_variance; ensures the model sees original
        # timestep embeddings during training loss computation.
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        """
        Return a _WrappedModel unless the model is already wrapped,
        preventing double-wrapping across multiple calls.
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Timestep rescaling is delegated to _WrappedModel.__call__ so that
        # the mapping to original indices happens before any scaling.
        return t


class _WrappedModel:
    """
    Thin wrapper around the neural network that translates spaced timestep
    indices back to their original positions in the base diffusion schedule
    before passing them to the model.

    This is necessary because the model was trained with timestep embeddings
    corresponding to the original (unspaced) schedule. During spaced sampling,
    the loop iterates over 0, 1, ..., T_spaced-1, but the model must receive
    the original indices (e.g. 0, 100, 200, ...) to produce meaningful outputs.

    :param model:               the underlying neural network.
    :param timestep_map:        list mapping spaced index i → original index.
    :param rescale_timesteps:   if True, rescale original indices to [0, 1000].
    :param original_num_steps:  total number of steps in the base schedule,
                                used for the optional rescaling.
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model             = model
        self.timestep_map      = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, cond_top, inivp, struc, well, well_loc, ts, **kwargs):
        # Convert spaced indices to their corresponding original timestep indices
        # using the pre-built timestep_map lookup table.
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]

        # Optionally rescale to the [0, 1000] range expected by the model's
        # sinusoidal timestep embedding (matches the original DDPM convention).
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        return self.model(x, cond_top, inivp, struc, well, well_loc, new_ts, **kwargs)