# Installation Guide
## Prerequisites
1. A recent version of python. These docs were built using python 3.9.16
2. If you're running locally, you'll also need DeepLabCut's [dependencies](https://deeplabcut.github.io/DeepLabCut/docs/installation.html)

## Creating a conda environment
Run the following command
```bash
conda create -n your-env-name python=your-py-version
```

## Installing Python dependencies
1. Activate your conda environment
    ```bash
    conda activate your-env-name
    ```
1. If you are going to be following the tutorial in the [usage guide](usage.md), install this package + ipython:
    ```bash
    pip install deepxromm[cli]
    ```
1. If you are a developer looking to install and use/extend this package in other Python scripts:
    ```bash
    pip install deepxromm
    ```
