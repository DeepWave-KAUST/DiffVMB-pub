import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter


def load_data(
    *, data_dir, batch_size, depth_size, vmax, vmin, device, class_cond=False, deterministic=False
):
    """
    Create an infinite generator over (velocity_model, conditioning_info, ...) batches.

    Each batch yields a tuple of tensors used for depth-progressive seismic
    velocity model building (Part II: realistic field-data constraints):
        - vp_bottom:    target velocity patch at the deeper depth window
        - cond_top:     3-channel conditioning input (vp_top, depth_top, depth_bottom)
        - inivp_bottom: background velocity patch derived from a smoothed migration
                        velocity model, providing a low-wavenumber starting point
        - struc_bottom: migration-derived structural image at the bottom window,
                        replacing the idealized reflectivity used in Part I
        - well:         sparse well-log velocity constraint at the bottom window
        - well_loc:     normalized horizontal position of the well
        - out_dict:     optional auxiliary dict (e.g. class labels)

    :param data_dir:      root directory containing .mat / .npz / .npy files
    :param batch_size:    number of samples per batch
    :param depth_size:    number of depth samples in each extracted window
    :param vmax:          maximum P-wave velocity (m/s) used for normalization;
                          should be set to the upper bound of the training
                          velocity distribution (e.g. 5000 m/s)
    :param vmin:          minimum P-wave velocity (m/s) used for normalization;
                          should be set to the lower bound of the training
                          velocity distribution (e.g. 1400 m/s)
    :param device:        torch device (unused directly; kept for API compatibility)
    :param class_cond:    if True, include class-label tensors in out_dict
    :param deterministic: if True, disable shuffle for reproducible ordering
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    # Recursively collect all supported geophysical data files
    all_files = _list_image_files_recursively(data_dir)

    dataset = BasicDataset(
        data_dir,
        depth_size,
        vmax,
        vmin,
        class_cond=class_cond,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not deterministic,
        num_workers=4,      # parallel data loading workers
        drop_last=True,     # discard the final partial batch for consistent tensor shapes
    )

    # Wrap the DataLoader in an infinite loop so callers can simply call next()
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    """
    Recursively enumerate all geophysical data files (.mat, .npz, .npy)
    under data_dir, sorted alphabetically for reproducibility.
    """
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat", "npz", "npy"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


# ─── Normalization / Denormalization Utilities ────────────────────────────────

def normalizer_vel(x, dmin=1500, dmax=4500):
    """
    Linearly map P-wave velocity from [dmin, dmax] m/s to [-1, 1].
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def denormalizer_vel(x, dmin=1500, dmax=4500):
    """
    Inverse of normalizer_vel: map normalized values in [-1, 1] back to
    physical velocity in m/s.
    """
    return 0.5 * (x + 1) * (dmax - dmin) + dmin


def normalizer_depth(x, dmin=0, dmax=1001):
    """
    Map absolute depth sample indices [dmin, dmax] to [-1, 1].
    Encodes the spatial position of each row within the full velocity model.
    dmax should be set to the total number of depth samples (nz).
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


def normalizer_well_loc(x, dmin=0, dmax=255):
    """
    Map a well's horizontal trace index [dmin, dmax] to [-1, 1].
    dmax should be set to nx - 1 (i.e. the last valid column index).
    """
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0


# ─── Wavelet & Synthetic Seismogram Utilities ─────────────────────────────────

def ricker_wavelet(frequency, nt, dt):
    """
    Generate a zero-phase Ricker (Mexican-hat) wavelet.

    :param frequency: dominant frequency in Hz
    :param nt:        number of time samples
    :param dt:        sample interval in seconds
    :return:          1-D numpy array of length nt, centered at t=0
    """
    t = (nt - 1) * dt
    t_shifted = np.linspace(-t / 2, t / 2, nt)
    pi_square = np.pi ** 2
    wavelet = (
        (1.0 - 2.0 * pi_square * (frequency ** 2) * (t_shifted ** 2))
        * np.exp(-pi_square * (frequency ** 2) * (t_shifted ** 2))
    )
    return wavelet


def convolve_wavelet(nz_exp, nx_exp, ref, wavelet):
    """
    Produce a post-stack synthetic seismogram by convolving a reflectivity
    section with a 1-D wavelet, trace by trace.

    :param nz_exp:  number of depth (time) samples in the output
    :param nx_exp:  number of lateral traces
    :param ref:     2-D reflectivity array of shape (nz_exp, nx_exp)
    :param wavelet: 1-D wavelet array
    :return:        synthetic seismogram of shape (nz_exp, nx_exp)
    """
    seis = np.zeros([nz_exp, nx_exp])
    for i in range(nx_exp):
        seis[:, i] = convolve(ref[:, i], wavelet, mode='same')
    return seis


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BasicDataset(Dataset):
    """
    PyTorch Dataset for depth-progressive velocity model building (Part II).

    Compared with Part I, two additional realistic constraints are introduced:

    1. Migration-derived structural image ('mig'): replaces the idealized
       reflectivity computed from the true velocity. In practice this image
       is obtained by applying reverse-time migration (RTM) or another
       migration algorithm with a smooth migration velocity model, so it
       captures structural geometry without requiring knowledge of the true
       velocity.

    2. Background velocity model ('inivp'): a heavily smoothed version of the
       true velocity, simulating the migration velocity model used as a
       low-wavenumber starting point. During training the smoothing kernel
       width is randomized (sigma in [18, 22] samples) to improve robustness
       to imperfect migration velocities encountered in practice.

    Each sample randomly selects:
      - a 'top' depth window  (shallow, known/observed region)
      - a 'bottom' depth window (deeper, target region to predict)
    and returns the velocity patches, migration-derived structural image,
    background velocity, well constraint, and depth-position encodings.

    Expected file format: .npz with keys:
      'vp'  -- true P-wave velocity,          shape (nz, nx)
      'mig' -- migration-derived structural image, shape (nz, nx)
    """

    def __init__(self, paths, depth_size, vmax, vmin, class_cond=False):
        super().__init__()
        self.local_dataset = _list_image_files_recursively(paths)
        self.class_cond = class_cond
        self.depth_size = depth_size   # height of each extracted window (in samples)
        self.vmax = vmax               # upper velocity bound (m/s) for normalization
        self.vmin = vmin               # lower velocity bound (m/s) for normalization

    def __len__(self):
        return len(self.local_dataset)

    def __getitem__(self, idx):
        path = self.local_dataset[idx]

        # ── Load raw data ──────────────────────────────────────────────────
        # data  = np.load(path)
        data  = sio.loadmat(path)
        vp    = data['acc_vp']    # true P-wave velocity model, shape (nz, nx)
        struc = data['mig']   # migration-derived structural image, shape (nz, nx);
                               # replaces the idealized reflectivity used in Part I

        # Simulate the migration velocity model by applying a Gaussian smooth
        # to the true velocity. The randomized sigma (18–22 samples) mimics
        # the uncertainty in real migration velocity estimation.
        inivp = gaussian_filter(vp.copy(), sigma=random.randint(18, 22))
        nz, nx = vp.shape

        # ── Random lateral flip (data augmentation) ────────────────────────
        # Mirrors the model left-right with 50% probability; applied
        # consistently to all three arrays to preserve their correspondence.
        if random.uniform(0, 1) >= 0.5:
            vp    = np.fliplr(vp)
            inivp = np.fliplr(inivp)
            struc = np.fliplr(struc)

        # ── Normalize velocities to [-1, 1] ───────────────────────────────
        vp    = normalizer_vel(vp,    dmin=self.vmin, dmax=self.vmax)
        inivp = normalizer_vel(inivp, dmin=self.vmin, dmax=self.vmax)

        # ── Sample the top and bottom depth windows ────────────────────────
        # depth_top: deepest row of the 'top' (shallow) window.
        depth_top = random.randint(
            self.depth_size - 1,
            nz - self.depth_size // 2 - 1
        )
        depth_gap = min(
            nz - depth_top - 1,
            random.randint(self.depth_size // 2, self.depth_size)
        )
        depth_bottom = depth_top + depth_gap  # deepest row of the 'bottom' window

        # ── Extract velocity patches ───────────────────────────────────────
        # Top window: shallow velocity context from the previous diffusion step
        vp_top    = vp[depth_top - self.depth_size + 1 : depth_top + 1]           # (depth_size, nx)
        # Bottom window: target to be predicted by the diffusion model
        vp_bottom = vp[depth_bottom - self.depth_size + 1 : depth_bottom + 1]     # (depth_size, nx)
        # Background velocity at the bottom window (smoothed migration velocity)
        inivp_bottom = inivp[depth_bottom - self.depth_size : depth_bottom]       # (depth_size, nx)
        # Migration-derived structural image at the bottom window
        struc_bottom = struc[depth_bottom - self.depth_size + 1 : depth_bottom + 1]  # (depth_size, nx)

        # ── Well-log conditioning (sparse column mask) ─────────────────────
        # Simulate a single borehole at a random lateral position.
        # Only the borehole column retains the true velocity; all other
        # columns are set to zero.
        well_loc = random.randint(0, nx - 1)
        mask = np.zeros_like(vp_bottom)
        mask[:, well_loc] = 1
        well = vp_bottom * mask    # (depth_size, nx), non-zero only at well_loc

        # Option 2 (alternative, currently commented out):
        # Broadcast the single borehole trace to all lateral positions.
        # well = vp_bottom[:, well_loc]
        # well = np.repeat(well[:, np.newaxis], nx, axis=1)

        # ── Build 2-D depth-position encodings ────────────────────────────
        # Label each pixel with its absolute depth index so the network can
        # reason about its spatial position within the full velocity model.
        depth_top_idx    = np.arange(depth_top    - self.depth_size + 1, depth_top    + 1)
        depth_bottom_idx = np.arange(depth_bottom - self.depth_size + 1, depth_bottom + 1)

        # Tile to 2-D maps of shape (depth_size, nx)
        depth_top_idx    = np.tile(depth_top_idx[:, np.newaxis],    (1, nx))
        depth_bottom_idx = np.tile(depth_bottom_idx[:, np.newaxis], (1, nx))

        # Normalize depth indices to [-1, 1]
        depth_top_idx    = normalizer_depth(depth_top_idx,    dmax=nz)
        depth_bottom_idx = normalizer_depth(depth_bottom_idx, dmax=nz)

        # Normalize the well's lateral position to a scalar in [-1, 1]
        well_loc = normalizer_well_loc(well_loc, dmax=nx)
        well_loc = np.expand_dims(np.array(well_loc, dtype=np.float32), axis=0)  # (1,)

        # ── Pack tensors ───────────────────────────────────────────────────
        # Target: single-channel true velocity patch at the bottom window
        vp_bottom = np.expand_dims(np.array(vp_bottom, dtype=np.float32), axis=0)  # (1, D, W)

        # Conditioning input for the diffusion model — 3 channels:
        #   ch 0 – vp_top:           shallow velocity context (previous window output)
        #   ch 1 – depth_top_idx:    absolute depth encoding of the top window rows
        #   ch 2 – depth_bottom_idx: absolute depth encoding of the bottom window rows
        cond_top = np.array(
            np.stack([vp_top, depth_top_idx, depth_bottom_idx], axis=0),
            dtype=np.float32
        )  # (3, D, W)

        # Single-channel background velocity from the smoothed migration velocity model
        inivp_bottom = np.expand_dims(np.array(inivp_bottom, dtype=np.float32), axis=0)  # (1, D, W)

        # Single-channel migration-derived structural image (replaces idealized
        # reflectivity from Part I)
        struc_bottom = np.expand_dims(np.array(struc_bottom, dtype=np.float32), axis=0)  # (1, D, W)

        # Single-channel sparse well constraint (non-zero only at the borehole column)
        well = np.expand_dims(np.array(well, dtype=np.float32), axis=0)  # (1, D, W)

        out_dict = {}   # placeholder for optional class-conditioning labels

        return vp_bottom, cond_top, inivp_bottom, struc_bottom, well, well_loc, out_dict