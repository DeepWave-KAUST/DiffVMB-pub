![LOGO](https://github.com/DeepWave-Kaust/DiffVMB-pub/blob/main/logo/logo.jpg)

<div align="center">

<h3><strong>Shallow-to-deep velocity model building via diffusion models</strong></h2>

<h4>Shijun Cheng, Randy Harsuko, Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>

# Project structure
This repository is organized as follows:

* :open_file_folder: **diffvmb**: python library containing all codes;
* :open_file_folder: **logo**: folder containing logo;
* :open_file_folder: **dataset**: folder to store dataset;

## Supplementary files
To ensure reproducibility, we provide the the data set for training and inference stages and our trainined diffusion model for GSFM.

* **Dataset**:
Download the training and testing data set [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0100/dataset.zip?csf=1&web=1&e=7mO9tu). Then, use `unzip` to extract the contents.

* **Trained model**:
Download our trained diffusion model [here](https://kaust.sharepoint.com/:u:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0100/trained_model.zip?csf=1&web=1&e=48l4Yz). Then, use `unzip` to extract the contents.

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

