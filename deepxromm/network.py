"""Primary interface for training the XROMM network using DLC"""

import os

import deeplabcut

from .xrommtools import xma_to_dlc
from .xma_data_processor import XMADataProcessor


class Network:
    """Trains an XROMM labeling network using DLC."""

    def __init__(self, config):
        self._data_path = os.path.join(config["working_dir"], "trainingdata")
        self._data_processor = XMADataProcessor(config)
        self._config = config

    def xma_to_dlc(self):
        """Converts XMA-formatted data into something that is ready for DeepLabCut to analyze"""
        mode = self._config["tracking_mode"]
        if mode == "2D":
            try:
                xma_to_dlc(
                    self._config["path_config_file"],
                    self._data_path,
                    self._config["dataset_name"],
                    self._config["experimenter"],
                    self._config["nframes"],
                )
            except UnboundLocalError:
                pass
        elif mode == "per_cam":
            xma_to_dlc(
                path_config_file=self._config["path_config_file"],
                path_config_file_cam2=self._config["path_config_file_2"],
                data_path=self._data_path,
                dataset_name=self._config["dataset_name"],
                scorer=self._config["experimenter"],
                nframes=self._config["nframes"],
                nnetworks=2,
            )
        elif mode == "rgb":
            self._data_processor.make_rgb_video(self._data_path)
        else:
            raise AttributeError(f"Unsupported mode: {mode}")

    def check_labels(self):
        """Check labels for labeled data within the trial"""
        deeplabcut.check_labels(self._config["path_config_file"], draw_skeleton=False)
        if self._config["tracking_mode"] == "per_cam":
            deeplabcut.check_labels(self._config["path_config_file_2"], draw_skeleton=False)

    def create_training_dataset(self):
        """Creates a training dataset for DeepLabCut data"""
        deeplabcut.create_training_dataset(self._config["path_config_file"])
        if self._config["tracking_mode"] == "per_cam":
            deeplabcut.create_training_dataset(self._config["path_config_file_2"])

    def train(self):
        """Starts training data"""
        deeplabcut.train_network(
            self._config["path_config_file"], maxiters=self._config["maxiters"]
        )

        if self._config["tracking_mode"] == "per_cam":
            deeplabcut.train_network(
                self._config["path_config_file_2"], maxiters=self._config["maxiters"]
            )
