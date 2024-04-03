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
1. Change to a folder to store the code in, clone the GitHub repository, and change into the project directory
   1. Windows example (make sure you install [Git bash](https://git-scm.com/download/win) first!)
       ```powershell
        cd Documents
        git clone https://github.com/sam-delap/sdlc_xmalab.git
        cd sdlc_xmalab
       ```
    1. macOS/Linux example
        ```bash
        cd
        git clone https://github.com/sam-delap/sdlc_xmalab.git
        cd sdlc_xmalab
        ```
1. Install all of the pip requirements
    ```bash
    pip install -r requirements.txt
    ```
