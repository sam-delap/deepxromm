"""Utility for creating or loading project configs."""
import warnings

import cv2
import deeplabcut
import numpy as np
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML

from .xma_data_processor import XMADataProcessor

class Project:
    def __init__(self):
        raise NotImplementedError("Use create_new_config or load_config instead.")

    @staticmethod
    def create_new_config(working_dir=None, experimenter="NA", mode="2D"):
        """Creates a new config from scratch."""
        try:
            working_dir = Path(working_dir) if working_dir is not None else Path.cwd()
            (working_dir / "trainingdata").mkdir(parents=True)
            (working_dir / "trials").mkdir(parents=True)
        except FileExistsError as e:
            print('It looks like this project folder already exists. Try importing it with DeepXROMM.load_project(working_dir)')
            raise e

        # Create a fake video to pass into the deeplabcut workflow
        dummy_video_path = working_dir / "dummy.avi"
        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        out = cv2.VideoWriter(str(dummy_video_path), cv2.VideoWriter_fourcc(*"DIVX"), 15, (480, 480))
        out.write(frame)
        out.release()

        # Create a new project
        task = working_dir.name
        path_config_file = deeplabcut.create_new_project(
            task,
            experimenter,
            [str(dummy_video_path)],
            str(working_dir / ''), # Add the trailing slash
            copy_videos=True
        )

        config_path = Path(__file__).parent / 'default_config.yaml'
        yaml = YAML()
        with config_path.open() as file:
            config_data = yaml.load(file)

        config_data.update({"task": task,
                            "experimenter": experimenter,
                            "working_dir": str(working_dir),
                            "path_config_file": path_config_file,
                            "tracking_mode": mode})

        if mode == "per_cam":
            task_2 = f"{task}_cam2"
            path_config_file_2 = deeplabcut.create_new_project(
                task_2,
                experimenter,
                [str(dummy_video_path)],
                str(working_dir / ''),  # Add the trailing slash
                copy_videos=True,
            )
            config_data["path_config_file_2"] = path_config_file_2

        # Save configuration
        config_file = working_dir / "project_config.yaml"
        with config_file.open('w') as config:
            yaml.dump(config_data, config)

        # Cleanup
        try:
            (Path(path_config_file).parent / "labeled-data/dummy").rmdir()
        except FileNotFoundError:
            pass

        try:
            (Path(path_config_file).parent / "videos/dummy.avi").unlink()
        except FileNotFoundError:
            pass

        dummy_video_path.unlink()
        return config_data

    @staticmethod
    def load_config(working_dir=None):
        '''Load an existing project'''
        working_dir = Path(working_dir) if working_dir is not None else Path.cwd()

        # Open the config
        config_path = working_dir / "project_config.yaml"
        yaml = YAML()
        with config_path.open('r') as config_file:
            project = yaml.load(config_file)

        experimenter = str(project['experimenter'])
        project['experimenter'] = experimenter
        if project['dataset_name'] == 'MyData':
            warnings.warn('Default project name in use', SyntaxWarning)

        # Navigate to the training data directory
        training_data_path = working_dir / "trainingdata"
        trials = [folder for folder in training_data_path.iterdir() if folder.is_dir() and not folder.name.startswith('.')]
        if len(trials) == 0:
            raise FileNotFoundError("Empty trials directory found. Expected trial folders within the 'trainingdata' directory")
        trial = trials[0] # Assuming there's at least one trial directory
        trial_path = training_data_path / trial.name

        # Load trial CSV
        try:
            trial_csv_path = trial_path / f'{trial.name}.csv'
            trial_csv = pd.read_csv(trial_csv_path)
        except FileNotFoundError as e:
            print(f'Please make sure that your trainingdata 2DPoints csv file is named {trial.name}.csv')
            raise e

        # Give search_area a minimum of 10
        project['search_area'] = int(project['search_area'] + 0.5) if project['search_area'] >= 10 else 10

        # Drop untracked frames (all NaNs)
        trial_csv = trial_csv.dropna(how='all')

        # Make sure there aren't any partially tracked frames
        if trial_csv.isna().sum().sum() > 0:
            raise AttributeError(f'Detected {len(trial_csv) - len(trial_csv.dropna())} partially tracked frames. \
        Please ensure that all frames are completely tracked')

        # Check/set the default value for tracked frames
        if project['nframes'] <= 0:
            project['nframes'] = len(trial_csv)

        elif project['nframes'] != len(trial_csv):
            warnings.warn('Project nframes tracked does not match 2D Points file. \
            If this is intentional, ignore this message')

        data_processor = XMADataProcessor(config=project)
        # Check the current nframes against the threshold value * the number of frames in the cam1 video
        cam1_video_path = data_processor.find_cam_file(str(trial_path), "cam1")
        video = cv2.VideoCapture(cam1_video_path)

        if project['nframes'] < int(video.get(cv2.CAP_PROP_FRAME_COUNT)) * project['tracking_threshold']:
            tracking_threshold = project['tracking_threshold']
            warnings.warn(f'Project nframes is less than the recommended {tracking_threshold * 100}% of the total frames')

        # Check DLC bodyparts (marker names)
        default_bodyparts = ['bodypart1', 'bodypart2', 'bodypart3', 'objectA']
        path_to_trial = working_dir / 'trainingdata' / trial
        bodyparts = data_processor.get_bodyparts_from_xma(
            str(path_to_trial),
            mode=project['tracking_mode'])

        dlc_config_loader = YAML()
        dlc_config_path = Path(project['path_config_file'])
        with dlc_config_path.open('r') as dlc_config:
            dlc_yaml = dlc_config_loader.load(dlc_config)

        if dlc_yaml['bodyparts'] == default_bodyparts:
            dlc_yaml['bodyparts'] = bodyparts
        elif dlc_yaml['bodyparts'] != bodyparts:
            raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

        with dlc_config_path.open('w') as dlc_config:
            yaml.dump(dlc_yaml, dlc_config)

        # Check DLC bodyparts (marker names) for config 2 if needed
        if project['tracking_mode'] == 'per_cam':
            dlc_config_loader = YAML()
            try:
                dlc_config_path_2 = Path(project['path_config_file_2'])
            except KeyError as e:
                print("Path to second DLC config not found. Did you create the project as a per-cam project?")
                print("If not, re-run 'create_new_project' using mode='per_cam'")
                raise e
            with dlc_config_path_2.open('r') as dlc_config:
                dlc_yaml = dlc_config_loader.load(dlc_config)
            # Better conditional logic could definitely be had to reduce function calls here
            if dlc_yaml['bodyparts'] == default_bodyparts:
                dlc_yaml['bodyparts'] = bodyparts
            elif dlc_yaml['bodyparts'] != bodyparts:
                raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

            with dlc_config_path_2.open('w') as dlc_config:
                yaml.dump(dlc_yaml, dlc_config)

        # Check test_autocorrect params for defaults
        if project['test_autocorrect']:
            if project['trial_name'] == 'your_trial_here':
                raise SyntaxError('Please specify a trial to test autocorrect() with')
            if project['marker'] == 'your_marker_here':
                raise SyntaxError('Please specify a marker to test autocorrect() with')

        # Update changed attributes to match in the file
        with config_path.open('w') as file:
            yaml.dump(project, file)

        return project
