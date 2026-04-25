import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import random
import torch
from scipy.signal import convolve
from scipy.io import loadmat

class CUDAPrefetcher():
    """CUDA prefetcher.
    Ref:
    https://github.com/NVIDIA/apex/issues/304#
    It may consums more GPU memory.
    Args:
        loader: Dataloader.
        opt (dict): Options.
    """

    def __init__(self, loader, opt=None):
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device('cuda')
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return None
        # put tensors to gpu
        with torch.cuda.stream(self.stream):
            if type(self.batch) == dict:
                for k, v in self.batch.items():
                    if torch.is_tensor(v):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            elif type(self.batch) == list:
                for k in range(len(self.batch)):
                    if torch.is_tensor(self.batch[k]):
                        self.batch[k] = self.batch[k].to(
                            device=self.device, non_blocking=True)
            else:
                assert NotImplementedError

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self):
        self.loader = iter(self.ori_loader)
        self.preload()

def load_data(
    *, data_dir, batch_size, depth_size, width_size, device, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    # all_files = _list_image_files_recursively(data_dir)

    dataset = BasicDataset(
        data_dir,
        depth_size, width_size,
        class_cond=class_cond,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def normalizer_vel(x, dmin=1000, dmax=5000):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def denormalizer_vel(x, dmin=1000, dmax=5000):
    return 0.5 * (x + 1) * (dmax - dmin) + dmin

def normalizer_depth(x, dmin=0, dmax=1001):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def normalizer_well_loc(x, dmin=0, dmax=255):
    return 2.0 * (x - dmin) / (dmax - dmin) - 1.0

def ricker_wavelet(frequency, nt, dt):
    t = (nt-1)*dt
    t_shifted = np.linspace(-t/2, t/2, nt)
    pi_square = np.pi ** 2
    wavelet = (1.0 - 2.0 * pi_square * (frequency ** 2) * (t_shifted ** 2)) * np.exp(-pi_square * (frequency ** 2) * (t_shifted ** 2))
    return wavelet

def convolve_wavelet(nz_exp, nx_exp, ref, wavelet):
    seis = np.zeros([nz_exp, nx_exp])
    for i in range(nx_exp):
        seis[:, i] = convolve(ref[:, i], wavelet, mode='same')
    return seis

class BasicDataset(Dataset):
    def __init__(self, paths, depth_size, width_size, dt=1e-3, class_cond=False):
        super().__init__()
        self.paths = paths
        self.class_cond = class_cond
        self.depth_size = depth_size
        self.width_size = width_size
        self.dt = dt

        model_list = ['BP1994', 'BP2004', 'BP2007', 'Hess', 'Otway', 'Sigsbee']
        for model in model_list:
            dict = loadmat(f'{paths}{model}.mat')
            vel = dict['vel']
            vel = normalizer_vel(vel)
            setattr(self, f'{model}_vel', vel)
            ref = dict['ref']
            setattr(self, f'{model}_ref', ref)

        self.cls_num = len(model_list) + 3

    def __len__(self):
        return 1000000 #len(self.local_dataset)

    def __getitem__(self, idx):
        rand_no = random.uniform(0, 1)

        if rand_no <= 1.0/self.cls_num:
            vp = self.BP1994_vel.copy()
            ref = self.BP1994_ref.copy()
        elif 1.0/self.cls_num < rand_no <= 1.0/self.cls_num*2:
            vp = self.BP2004_vel.copy()
            ref = self.BP2004_ref.copy()
        elif 1.0/self.cls_num*2 < rand_no <= 1.0/self.cls_num*3:
            vp = self.BP2007_vel.copy()
            ref = self.BP2007_ref.copy()
        elif 1.0/self.cls_num*3 < rand_no <= 1.0/self.cls_num*4:
            vp = self.Hess_vel.copy()
            ref = self.Hess_ref.copy()
        elif 1.0/self.cls_num*4 < rand_no <= 1.0/self.cls_num*5:
            vp = self.Otway_vel.copy()
            ref = self.Otway_ref.copy()
        elif 1.0/self.cls_num*5 < rand_no <= 1.0/self.cls_num*6:
            vp = self.Sigsbee_vel.copy()
            ref = self.Sigsbee_ref.copy()
        elif 1.0/self.cls_num*6 < rand_no <= 1.0/self.cls_num*7:
            id = random.randint(1, 42)
            dict = loadmat(f'{self.paths}SEAMArid/vel{id}.mat')
            vp = dict['vel']
            vp = normalizer_vel(vp)
            ref = dict['ref']
        elif 1.0/self.cls_num*7 < rand_no <= 1.0/self.cls_num*8:
            id = random.randint(1, 82)
            dict = loadmat(f'{self.paths}Overthrust/vel{id}.mat')
            vp = dict['vel']
            vp = normalizer_vel(vp)
            ref = dict['ref']
        elif 1.0/self.cls_num*8 < rand_no:
            id = random.randint(1, 70)
            dict = loadmat(f'{self.paths}SEGEAGE/vel{id}.mat')
            vp = dict['vel']
            vp = normalizer_vel(vp)
            ref = dict['ref']

        nz, nx = vp.shape

        depth_top = random.randint(self.depth_size - 1, nz - self.depth_size//2 - 1)
        depth_gap = min(nz - depth_top - 1, random.randint(self.depth_size//2, self.depth_size))
        depth_bottom = depth_top + depth_gap

        width_left = random.randint(0, nx - self.width_size - 1)

        vp_top = vp[depth_top-self.depth_size+1:depth_top+1, width_left:width_left+self.width_size]
        vp_bottom = vp[depth_bottom-self.depth_size+1:depth_bottom+1, width_left:width_left+self.width_size]
        ref_bottom = ref[depth_bottom-self.depth_size+1:depth_bottom+1, width_left:width_left+self.width_size]

        well_loc = random.randint(0, self.width_size - 1)
        well = vp_bottom[:, well_loc]
        well = np.repeat(well[:, np.newaxis], self.width_size, axis=1)

        depth_top = np.arange(depth_top - self.depth_size + 1, depth_top + 1)
        depth_bottom = np.arange(depth_bottom - self.depth_size + 1, depth_bottom + 1)
        depth_top = np.tile(depth_top[:, np.newaxis], (1, self.width_size))
        depth_bottom = np.tile(depth_bottom[:, np.newaxis], (1, self.width_size))
        depth_top = normalizer_depth(depth_top)
        depth_bottom = normalizer_depth(depth_bottom)

        well_loc = normalizer_well_loc(well_loc, dmax=self.width_size)
        well_loc = np.expand_dims(np.array(well_loc, dtype=np.float32), axis=0)

        vp_bottom = np.expand_dims(np.array(vp_bottom, dtype=np.float32), axis=0)
        cond_top = np.array(np.stack([vp_top, depth_top, depth_bottom], axis=0), dtype=np.float32)

        struc_bottom = np.expand_dims(np.array(ref_bottom, dtype=np.float32), axis=0)
        well = np.expand_dims(np.array(well, dtype=np.float32), axis=0)

        out_dict = {}
        return vp_bottom, cond_top, struc_bottom, well, well_loc, out_dict
