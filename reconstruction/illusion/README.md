# Reconstruction with decoded DNN features

## Setup

### Setting up environment

You can make Python environment to run the code with [Anaconda](https://anaconda.org/).

```shellsession
$ conda env create -n <env name> -f env.yaml
$ conda activate <env name>
```

If the reconstruction environment has already been created for the attention or imagery reconstruction scripts, you can reuse that environment and skip this step.

### Downloading generator

Run the following in this directory.

``` shellsession
$ python download.py GAN
```

## Usage

### Illusion reconstruction from trial-averaged brain activity

Run the following command for averaged-trial reconstruction.

``` shellsession
$ python recon_feature_to_GAN.py
```

This will output reconstructed images in `results/reconstruction/recon_images/GAN`.

### Illusion reconstruction from single-trial brain activity

Run the following command to reconstruct all trials.

``` shellsession
$ python recon_feature_to_GAN_single_trial.py
```

This will output reconstructed images in `results/reconstruction/recon_images_single_trial/GAN`.
