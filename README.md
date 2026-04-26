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
  * :page_facing_up: **train_part1.py**: training script for Part I;
  * :page_facing_up: **sample_part1.py**: sampling/inference script for Part I;
* :open_file_folder: **diffvmb_part2**: code for Part II of the manuscript;
  * :open_file_folder: **code**: python library containing model, diffusion, and dataset utilities;
  * :page_facing_up: **train_part2.py**: training script for Part II;
  * :page_facing_up: **sample_part2.py**: sampling/inference script for Part II;
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

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
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
When you have downloaded the supplementary files and have installed the environment, you can run the training and sampling code. 
For traning, you can directly run:
```
python train.py
```

When you test the performance of our trained diffusion model, you can use the test data we provide. 
For in-distribution models (SEAM Aird, SEG/EAGE, Overthrust) sampling, you can directly run:
```
python sample_indistribution.py
```

For out-of-distribution model (Marmousi II) sampling, you can directly run:
```
python sample_marmousi.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `diffvmb/train.py` file.

## Acknowledgements
This implementation is motivated from the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open source code.

## Cite us 
Cheng et al. (2026) Shallow-to-deep velocity model building via diffusion models.

