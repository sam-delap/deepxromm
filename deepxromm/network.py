"""Primary interface for training the XROMM network using DLC"""

from pathlib import Path

import deeplabcut

from .xrommtools import xma_to_dlc
from .xma_data_processor import XMADataProcessor


class Network:
    """Trains an XROMM labeling network using DLC."""

    def __init__(self, config):
        self.working_dir = Path(config["working_dir"])
        self._trainingdata_path = self.working_dir / "trainingdata"  # Keep for RGB mode
        self._data_processor = XMADataProcessor(config)
        self._config = config

    def xma_to_dlc(self):
        """Convert XMAlab data to DLC format"""
        mode = self._config["tracking_mode"]
        if mode == "2D":
            try:
                xma_to_dlc(
                    Path(self._config["path_config_file"]),
                    "trainingdata",  # Use relative path from working_dir
                    self._config["dataset_name"],
                    self._config["experimenter"],
                    self._config["nframes"],
                    data_processor=self._data_processor,
                )
            except UnboundLocalError:
                pass
        elif mode == "per_cam":
            xma_to_dlc(
                path_config_file=Path(self._config["path_config_file"]),
                trials_suffix="trainingdata",  # Use relative path from working_dir
                dataset_name=self._config["dataset_name"],
                scorer=self._config["experimenter"],
                nframes=self._config["nframes"],
                data_processor=self._data_processor,
                nnetworks=2,
                path_config_file_cam2=Path(self._config["path_config_file_2"]),
            )
        elif mode == "rgb":
            print("We've selected an RGB video")
            self._data_processor.make_rgb_videos("trainingdata")
            self._data_processor.xma_to_dlc_rgb("trainingdata")
        else:
            raise AttributeError(f"Unsupportede mode: {mode}")

    def create_training_dataset(self):
        """Create training dataset for data"""
        deeplabcut.create_training_dataset(self._config["path_config_file"])
        if self._config["tracking_mode"] == "per_cam":
            deeplabcut.create_training_dataset(self._config["path_config_file_2"])

    def train(self):
        """Starts training a network"""
        deeplabcut.train_network(
            self._config["path_config_file"], maxiters=self._config["maxiters"]
        )

        if self._config["tracking_mode"] == "per_cam":
            deeplabcut.train_network(
                self._config["path_config_file_2"], maxiters=self._config["maxiters"]
            )
