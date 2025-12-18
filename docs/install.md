# Installation Guide
## Prerequisites
1. **Python 3.10** (tested and known to work with DeepXROMM)

> **Note:** DeepXROMM depends on DeepLabCut, which requires Python 3.10+. 

## Creating a conda environment
Run the following command
```bash
conda create -n your-env-name python=3.10
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

> **Note:** The `[cli]` optional dependency adds IPython for interactive use, but all core functionality works with the base package.
