"""
This module tracks information about the DeepLabCut project(s) nested within a deepxromm project
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import deeplabcut

from deepxromm.config_utilities import load_config_file, save_config_file
from deepxromm.logging import logger
from deepxromm.trial import Trial
from deepxromm.xrommtools import get_marker_names, get_marker_and_cam_names

DEFAULT_BODYPARTS = ["bodypart1", "bodypart2", "bodypart3", "objectA"]


# Abstract class
@dataclass
class DlcConfig(ABC):
    """Interacts with and stores information about the DeepLabCut project(s) within a deepxromm project"""

    path_config_file: Path
    mode: str = ""
    dataset_name: str = "MyData"
    maxiters: int = 150000
    bodyparts_func: Callable = get_marker_names

    # Read-write properties
    @property
    def iteration(self):
        """Iteration of training for DLC projects"""
        dlc_config = load_config_file(self.path_config_file)
        return dlc_config["iteration"]

    @iteration.setter
    def iteration(self, value):
        """Setter for iteration within DLC projects"""
        dlc_config = load_config_file(self.path_config_file)
        current_iteration = dlc_config["iteration"]
        if current_iteration == value:
            logger.warning(f"Iteration already set to {value} in DLC config. Skipping.")
            return

        if dlc_config["iteration"] < value:
            logger.error(
                f"Iteration is set to {current_iteration}, which is greater than {value}"
            )
            raise ValueError("Iteration should only increase, never decrease")

        logger.info(f"Updating DLC config iteration to {value}...")
        dlc_config["iteration"] = value
        save_config_file(dlc_config, self.path_config_file)

    # Read-only properties
    # Public methods
    def get_bodyparts(self, trial_csv_path: Path):
        """Return bodyparts in the format they'll need to be in for DeepLabCut"""
        return self.bodyparts_func(trial_csv_path)

    # Abstract methods (need to be implemented in subclasses)
    @abstractmethod
    def xma_to_dlc(self, trial: Trial):
        """Convert XMA-formatted data into DeepLabCut input"""
        pass


# Class factory
class DlcConfigFactory:
    """Factory class for instantiating new DeepLabCut configs"""

    _CONFIG_MODES = ["2D", "per_cam", "rgb"]

    def __init__(self):
        raise NotImplementedError(
            "Use 'create_new_config' or 'load_existing_config' instead"
        )

    @classmethod
    def create_new_config(cls, task: str, mode: str = "2D", **kwargs) -> DlcConfig:
        """Create new DLC projects. Accepts all keywords of deeplabcut.create_new_project"""
        # Create DLC projects
        path_config_file = Path(deeplabcut.create_new_project(task, **kwargs))
        if mode == "per_cam":
            task_2 = f"{task}_cam2"
            path_config_file_2 = Path(deeplabcut.create_new_project(task_2, **kwargs))
            dlc_config = cls._instantiate_dlc_config(
                mode, path_config_file, path_config_file_2=path_config_file_2
            )
        else:
            dlc_config = cls._instantiate_dlc_config(mode, path_config_file)

        # Clean up DLC defaults
        try:
            (Path(path_config_file).parent / "labeled-data/dummy").rmdir()
        except FileNotFoundError:
            pass

        try:
            (Path(path_config_file).parent / "videos/dummy.avi").unlink()
        except FileNotFoundError:
            pass

        return dlc_config

    @classmethod
    def load_existing_config(
        cls, mode: str, path_config_file: Path, path_config_file_2: Path | None = None
    ) -> DlcConfig:
        # Load existing config file
        dlc_config = load_config_file(path_config_file)

        # Extract all class properties
        dlc_config = cls._instantiate_dlc_config(
            mode, path_config_file, path_config_file_2
        )

        return dlc_config

    @classmethod
    def _instantiate_dlc_config(
        cls,
        mode: str,
        path_config_file: Path,
        path_config_file_2: Path | None = None,
        **kwargs,
    ) -> DlcConfig:
        """Instantiate the correct type of DlcConfig"""
        match mode:
            case "2D":
                dlc_config = DlcConfig2D(path_config_file=path_config_file, **kwargs)
            case "per_cam":
                if path_config_file_2 is None:
                    raise ValueError(
                        "Please specify a value for 2nd DLC project config. Value is currently unset"
                    )
                dlc_config = DlcConfigPerCam(
                    path_config_file=path_config_file,
                    path_config_file_2=path_config_file_2,
                    **kwargs,
                )
            case "rgb":
                dlc_config = DlcConfigRGB(path_config_file=path_config_file, **kwargs)

        return dlc_config


@dataclass
class DlcConfig2D(DlcConfig):
    """DLC config information for 2D projects"""

    mode: str = "2D"

    def xma_to_dlc(self, trial: Trial):
        """Convert XMA-formatted data into DeepLabCut input"""
        pass


class DlcConfigPerCam(DlcConfig):
    """DLC config information for per_cam projects"""

    @property
    def path_config_file_2(self):
        return self.path_config_file_2

    @path_config_file_2.setter
    def path_config_file_2(self, value):
        self.path_config_file_2 = Path(value)

    @property
    def mode(self):
        """Mode of the DLC config"""
        return "per_cam"


@dataclass
class DlcConfigRGB(DlcConfig):
    """DLC config information for RGB projects"""

    mode: str = "rgb"
    swapped_markers: bool = False
    crossed_markers: bool = False
    bodyparts_func: Callable = get_marker_and_cam_names

    def get_bodyparts(self, trial_csv_path: Path):
        """Return bodyparts in the format they'll need to be in for DeepLabCut"""
        return self.bodyparts_func(trial_csv_path)
