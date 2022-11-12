# Import packages
import os
import deeplabcut
from deeplabcut.utils import xrommtools
import cv2
import numpy as np
from ruamel.yaml import YAML

def create_new_project(working_dir=os.getcwd(), experimenter='Johnny Appleseed'):
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
    yaml = YAML()
    project = yaml.load(working_dir + "\\project_config.yaml")

def tif_to_avi(working_dir=os.getcwd()):
    '''Converts training data tif files to AVI videos'''
    XMA_dir = working_dir + "/trainingdata"
    try:
        os.chdir(XMA_dir)
    except FileNotFoundError:
        if os.path.isdir(working_dir + "/trainingdata"):
            raise Exception("Please create a trainingdata directory inside your parent directory")
    # For each trial
    for dir in os.listdir():
        # Change to the trials directory
        try:
            os.chdir(dir)
        except NotADirectoryError:
            raise Exception("Please separate your training_images directory into per-trial folders")
        try:
            os.chdir("training_images")   
        except:
            raise Exception("Please create a training_images directory inside your trial directory") 
        # Assuming all directories within a trial are for undistorted trial images, find how many tifdirs there are
        tifdirs = [f for f in os.listdir() if os.path.isdir(f)]
        if len(tifdirs) == 0:
            raise Exception("Please export a set of undistorted trial images into this directory")
        
        for tifdir in tifdirs:
            # Find filename pattern
            os.chdir(tifdir)
            filename = os.listdir('.')[0]
            index = filename.find('.')
            pattern = filename[:index+1]
            # Use ffmpeg
            index2 = filename.find('_cam')
            index3 = filename.find('_UND')
            cmd = "ffmpeg -i " + pattern + "%04d.tif -r 30 -b:v 9653K " + "..\\\\..\\\\..\\\\" + pattern[:index2] + "\\\\" + pattern[:index3] + ".avi"
            print(cmd)   
            os.system(cmd)
            os.chdir("../")