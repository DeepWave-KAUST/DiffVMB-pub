![LOGO](https://github.com/DeepWave-Kaust/DiffVMB-pub/blob/main/logo/logo.jpg)

<div align="center">

<h3><strong>Shallow-to-deep velocity model building via diffusion models</strong></h2>

<h4>Shijun Cheng, Randy Harsuko, Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>

## Project structure

This repository is organized as follows:

* :open_file_folder: **diffvmb_part1**: code for Part I of the manuscript;
  * :open_file_folder: **code**: python library containing model, diffusion, and dataset utilities;
  * :page_facing_up: **train.py**: training script for Part I;
  * :page_facing_up: **sample.py**: sampling/inference script for Part I;
* :open_file_folder: **diffvmb_part2**: code for Part II of the manuscript;
  * :open_file_folder: **code**: python library containing model, diffusion, and dataset utilities;
  * :page_facing_up: **train.py**: training script for Part II;
  * :page_facing_up: **sample.py**: sampling/inference script for Part II;
* :open_file_folder: **dataset**: empty folder, to be filled with the downloaded dataset (see Supplementary files);
* :open_file_folder: **trained_model**: empty folder, to be filled with the downloaded model weights (see Supplementary files);
* :open_file_folder: **logo**: folder containing logo;

## Supplementary files

The training/test datasets and pre-trained model weights for both parts of the manuscript are publicly available on Zenodo:

> **DOI: [10.5281/zenodo.19790506](https://doi.org/10.5281/zenodo.19790506)**

### Dataset (`dataset.zip`)

Download and extract `dataset.zip` into the `dataset/` folder. After extraction, the structure is:

dataset/
├── part1/
│   ├── train/          # Training data for Part I (NPZ format)
│   └── test/           # Test data for Part I (MAT format)
└── part2/
├── train/          # Training data for Part II (NPZ format)
└── test/           # Test data for Part II (MAT format)

- **Training data** (`.npz`): each file contains two arrays — `vp` (P-wave velocity model) and `ref` (Part I) or `mig` (Part II) — representing 2-D cross-sections extracted from industrial 3-D velocity models.
- **Test data** (`.mat`): benchmark velocity models including in-distribution models (SEAM Arid, SEG/EAGE, Overthrust for Part I; Syn for Part II) and an out-of-distribution model (Marmousi) to assess generalization.

**Part I** uses an idealized reflectivity model computed from the true velocity as the structural constraint.  
**Part II** introduces two more realistic constraints: a migration-derived structural image obtained by RTM with a smooth background velocity, and the background velocity model itself as an additional low-wavenumber constraint.

### Trained models (`trained_model.zip`)

Download and extract `trained_model.zip` into the `trained_model/` folder. After extraction, the structure is:

trained_model/
├── model_part1.pt      # Pre-trained model for Part I
└── model_part2.pt      # Pre-trained model for Part II

Both models are trained using a depth-progressive conditional diffusion framework built upon the IDDPM architecture, extended with custom multi-condition inputs including shallow velocity context, depth positional encoding, well-log constraints, and structural constraints.

## Getting started :space_invader: :robot:

To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment. Simply run:

```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate diffvmb
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Running code :page_facing_up:

Once the supplementary files have been downloaded and the environment has been installed, you can run the training and sampling scripts for each part independently.

### Part I

To train the Part I model from scratch:
```
cd diffvmb_part1
```
```
python train.py
```

To reproduce the Part I results using the provided trained model and test data:

```
cd diffvmb_part1
```
```
python sample.py
```

### Part II

To train the Part II model from scratch:
```
cd diffvmb_part2
```
```
python train.py
```

To reproduce the Part II results using the provided trained model and test data:

```
cd diffvmb_part2
```
```
python sample.py
```
**Note:** when running sampling with DDIM, set `--timestep_respacing ddim10` (or another `ddim{N}` value) on the command line to control the number of denoising steps.

**Disclaimer:** All experiments have been carried out on an Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GeForce A100 GPU. Different environment configurations may be required for different combinations of workstation and GPU. If your GPU does not support large batch sizes, please reduce the `batch_size` argument in the corresponding training or sampling script.

## Acknowledgements

This implementation is motivated by the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code is adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open-source contribution.

## Cite us 
Cheng et al. (2026) Shallow-to-deep velocity model building via diffusion models.

