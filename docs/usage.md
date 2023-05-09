# Usage Guide
There are two ways to use this package. You can either:

1. Follow the usage guide below to run everything locally.
2. Use the colab_tutortial.ipynb [Jupyter Notebook](https://drive.google.com/drive/folders/1X91DYNbcu4_tV1FMvF-28XB7p7SS-MBt) and an online computing platform like [Google Colab](https://colab.research.google.com/)
    1. If you are using this option, be sure to make a copy of the notebook before using it so that you can save your changes!

## Getting Started and Creating a New Project
1. If you haven't already, follow the steps in the [installation guide](https://sam-delap.github.io/sdlc_xmalab/install/) to install this package's dependencies!
2. Activate your conda environment
    ```bash
    conda activate your-env-name
    ```
3. Change into the folder where you stored the source code for this package (use \ instead of / on Windows)
    ```bash
    cd /path/to/folder-name
    ```
4. Open an interactive Python session
    ```bash
    ipython
    ```
5. From the terminal, run the following commands (replacing `/path/to/project-folder` with the path to the folder for your project):
    ```python
    import sdlc_xmalab
    working_dir = '/path/to/project-folder'
    experimenter = 'SD'
    sdlc_xmalab.create_new_project(working_dir, experimenter)
    ```
    1. Keep your Python session open. We'll be running more commands here shortly
6. You should now see something that looks like this inside of your project folder:
    ```bash
    sample-proj
    │   project_config.yaml
    │
    ├───sample-proj-SD-YYYY-MM-DD
    ├───trainingdata
    ├───trials
    └───XMA_files
    ```

## Importing your data and beginning network training
1. The simplest approach is to create a new folder inside of the trainingdata folder named after your trial and place your raw videos, as well as distorted 2D points from tracking, in the folder.
2. There are also a number of options for customization in the project_config.yaml file. Check out the config file reference to learn more about what each variable does
3. To start training your network, run the following in your Python terminal
    ```python
    sdlc_xmalab.train_network(working_dir)
    ```

