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
1. Once you've configured Conda environment to support DeepLabCut, simply install the package via pip:
    ```bash
    pip install deepxromm
    ```
1. Optionally, if you plan on following the usage guide, install ipython to do interactive python at the command line
    ```bash
    pip install ipython
    ```
