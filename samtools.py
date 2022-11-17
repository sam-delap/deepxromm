# Import packages
import os
import deeplabcut
from deeplabcut.utils import xrommtools
import cv2
import numpy as np
from ruamel.yaml import YAML

def create_new_project(working_dir=os.getcwd(), experimenter='NA'):
    '''Create a new xrommtools project'''
    dirs = ["trainingdata", "trials", "XMA_files"]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            continue

    config = open("project_config.yaml", 'w')

    # Create a fake video to pass into the deeplabcut workflow
    frame = np.zeros((480, 480, 3), np.uint8)
    out = cv2.VideoWriter('dummy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (480,480))
    for _ in range(4000):
        out.write(frame)
    out.release()

    # Create a new project
    yaml = YAML()
    task = working_dir.split("\\")[len(working_dir.split("\\")) - 1]
    path_config_file = deeplabcut.create_new_project(task, experimenter, [working_dir + "\\dummy.avi"], working_dir + "\\", copy_videos=True)
    template = f"""
    task: {task}
    experimenter: {experimenter}
    working_dir: {working_dir}
    path_config_file: {path_config_file}
    dataset_name: 
    nframes:
    """

    tmp = yaml.load(template)

    yaml.dump(tmp, config)
    config.close()

    # Remove temporary files
    try:
        os.rmdir(path_config_file[:path_config_file.find("config")] + "labeled-data\\dummy")
    except FileNotFoundError:
        pass

    try:
        os.remove(path_config_file[:path_config_file.find("config")] + "\\videos\\dummy.avi")
    except FileNotFoundError:
        pass

    try:
        os.remove("dummy.avi")
    except FileNotFoundError:
        pass

def train_network(working_dir=os.getcwd()):
    config_file = open(working_dir + "\\project_config.yaml", 'r')
    yaml = YAML()
    project = yaml.load(config_file)

    # Establish project vars
    path_config_file = project['path_config_file']
    data_path = working_dir + "\\trainingdata"
    dataset_name = project['dataset_name']
    experimenter = str(project['experimenter'])
    nframes = project['nframes']

    if dataset_name is None:
        raise Exception("Please specify a name for this dataset in the config file")
    if nframes is None:
        raise Exception("Please specify the number of frames in the training dataset")

    try:
        xrommtools.xma_to_dlc(path_config_file, data_path, dataset_name, experimenter, nframes)
    except UnboundLocalError:
        pass
    deeplabcut.create_training_dataset(path_config_file)
    deeplabcut.train_network(path_config_file)