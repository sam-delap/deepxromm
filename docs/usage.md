# Usage Guide
There are two ways to use this package. You can either:

1. Follow the usage guide below to run everything locally.
1. Use the colab_tutortial.ipynb [Jupyter Notebook](https://drive.google.com/drive/folders/1X91DYNbcu4_tV1FMvF-28XB7p7SS-MBt) and an online computing platform like [Google Colab](https://colab.research.google.com/)
    1. If you are using this option, be sure to make a copy of the notebook before using it so that you can save your changes!

## Getting started and creating a new project
1. If you haven't already, follow the steps in the [installation guide](install.md) to install this package's dependencies!
1. Activate your conda environment
    ```bash
    conda activate your-env-name
    ```
1. Change into the folder where you stored the source code for this package (use \ instead of / on Windows)
    1. Windows
        ```powershell
        cd Documents\sdlc_xmalab
        ```
    2. macOS/Linux
        ```bash
        cd ~/sdlc_xmalab
        ```
1. Open an interactive Python session
    ```bash
    ipython
    ```
1. From the terminal, run the following commands (replacing `/path/to/project-folder` with the path to the folder for your project and `SD` with your initials):
    ```python
    from deepxromm import DeepXROMM 
    working_dir = '/path/to/project-folder'
    experimenter = 'SD'
    deepxromm = DeepXROMM.create_new_project(working_dir, experimenter)
    ```
    1. Keep your Python session open. We'll be running more commands here shortly
1. You should now see something that looks like this inside of your project folder:
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
1. There are also a number of options for customization in the project_config.yaml file. Check out the [config file reference](config.md) to learn more about what each variable does
1. To start training your network, run the following in your Python terminal
    ```python
    deepxromm.train_network()
    ```

## Using a trained network to track your trial(s)
1. Make sure any trials that you want to analyze are in appropriately named folders in the `trials` directory, and each folder contains a CSV and distorted cam1/cam2 videos that are named **folder_name**.csv, **folder_name**_cam1.avi, and **folder_name**_cam2.avi, respectively
1. Import the package and initialize a deepxromm instance as a above, and run the following command in your Python terminal:
    ```python
    deepxromm.analyze_videos()
    ```
1. This will save a file named **trial_name**-Predicted2DPoints.csv to the it# file (where number is the number next to iteration: in your project_folder/project-name-SD-YYYY-MM-DD/config.yaml file) inside of your trials/trial_name folder
1. You can analyze the network's performance by importing this CSV as a 2D Points file into XMAlab with the following settings

![XMAlab import settings](XMA_import_settings.png){: .center}
## Using autocorrect()
This package comes pre-built with autocorrect() functions that leverage the same image filtering functions as XMAlab, and use the marker's outline to do centroid detection on each marker. You can modify the autocorrect function's performance using the **image processing** parameters from the [config file reference](config.md). You can also visualize the centroid detection process using the **test_autocorrect()** parameters.
### Testing autocorrect() parameters on a single marker/frame combination
You'll need a Python environment that is capable of displaying images, like a [Jupyter Notebook](https://jupyter.org/), for these steps  

1. Go to your project_config.yaml file and find the "Autocorrect() Testing Vars" section of the config  
1. Change the value of **test_autocorrect** to true by replacing the word "false" with the word "true", like this:  
    ```YAML
    test_autocorrect: true
    ```
1. Specify a trial (trial_name), camera (cam), frame number (frame_num), and marker name (marker) to test the autocorrect function on  
1. Import the package and initialize a deepxromm instance as a above and run the following code snippet
    ```python
    deepxromm.autocorrect_trial(working_dir)
    ```
1. Tune autocorrect() settings until you are satisfied with the testing output
### Using autocorrect for a whole trial
1. If you tested autocorrect, set the test_autocorrect variable in your config file to false
    ```YAML
    test_autocorrect: false
    ```
1. Import the package and initialize a deepxromm instance as a above and run the following code snippet
    ```python
    deepxromm.autocorrect_trial()
    ```
1. This will save a file named **trial_name**-AutoCorrected2DPoints.csv to the it# file (where number is the number next to iteration: in your project_folder/project-name-SD-YYYY-MM-DD/config.yaml file) inside of your trials/trial_name folder
    ```YAML
    iteration: 0
    ```
1. You can analyze autocorrect's performance by importing this CSV as a 2D Points file into XMAlab with the following settings

![XMAlab import settings](XMA_import_settings.png){: .center}

 
