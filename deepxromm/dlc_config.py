"""
This module tracks information about the DeepLabCut project(s) nested within a deepxromm project
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
import shutil

import deeplabcut

from deepxromm.config_utilities import load_config_file, save_config_file
from deepxromm.logging import logger
from deepxromm.video_encoding import validate_codec
from deepxromm.trial import Trial
from deepxromm.xrommtools import get_marker_names, get_marker_and_cam_names

DEFAULT_BODYPARTS = ["bodypart1", "bodypart2", "bodypart3", "objectA"]


# Abstract class
@dataclass(kw_only=True)
class DlcConfig(ABC):
    """Interacts with and stores information about the DeepLabCut project(s) within a deepxromm project"""

    _path_config_file: Path
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

    @property
    def path_config_file(self) -> Path:
        """Ensure path_config_file always returns a Path"""
        return Path(self._path_config_file)

    @path_config_file.setter
    def path_config_file(self, value: str | Path) -> None:
        """Ensure path_config_file is always set as a Path"""
        if isinstance(value, str):
            value = Path(value)

        self._path_config_file = value

    # Read-only properties

    # Public methods
    def get_bodyparts(self, trial_csv_path: Path) -> list[str]:
        """Return bodyparts in the format they'll need to be in for DeepLabCut"""
        return self.bodyparts_func(trial_csv_path)

    def train_network(self, **kwargs) -> None:
        """Train a DeepLabCut network"""
        deeplabcut.train_network(self.path_config_file, **kwargs)

    def clear_labeled_data(self, dataset_name: str) -> None:
        """Clear the labeled data folder to make way for new labeled data"""
        shutil.rmtree(self.path_config_file.parent / "labeled-data" / dataset_name)

    # Private methods
    def _configure_it_folder(self, trial: Trial) -> bool:
        """Configure it{iteration} folder for storing analysis info. Also checks for existing PredPoints CSV"""
        it_folder_name = f"it{self.iteration}"
        it_folder = trial.trial_path / it_folder_name
        csv_exists = True
        try:
            trial.find_trial_csv(suffix=it_folder_name, identifier="Predicted2DPoints")
        except FileNotFoundError:
            csv_exists = False

        # Ensure it_folder exists
        it_folder.mkdir(parents=True, exist_ok=True)

        # Tell the user whether or not the CSV is already there
        return csv_exists

    # Abstract methods
    @abstractmethod
    def analyze_videos(self, trial: Trial):
        """Analyze videos for a trial with an existing DeepLabCut network"""


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
                dlc_config = DlcConfig2D(_path_config_file=path_config_file, **kwargs)
            case "per_cam":
                if path_config_file_2 is None:
                    raise ValueError(
                        "Please specify a value for 2nd DLC project config. Value is currently unset"
                    )
                dlc_config = DlcConfigPerCam(
                    _path_config_file=path_config_file,
                    _path_config_file_2=path_config_file_2,
                    **kwargs,
                )
            case "rgb":
                dlc_config = DlcConfigRGB(_path_config_file=path_config_file, **kwargs)

        return dlc_config


@dataclass(kw_only=True)
class DlcConfig2D(DlcConfig):
    """DLC config information for 2D projects"""

    mode: str = "2D"

    def xma_to_dlc(self, trial: Trial):
        """Convert XMA-formatted data into DeepLabCut input"""
        pass

    def analyze_videos(self, trial: Trial, **kwargs):
        """Analyze videos with an existing DeepLabCut network"""
        csv_exists = self._configure_it_folder(trial)
        if csv_exists:
            logger.warning(
                f"There are already predicted points in iteration {self.iteration} subfolders... skipping point prediction"
            )
            return

        cameras = [1, 2]
        for camera in cameras:
            # Error handling handled by find_cam_file helper
            video = trial.find_cam_file(identifier=f"cam{camera}")
            deeplabcut.analyze_videos(
                str(self.path_config_file),
                [
                    str(video)
                ],  # DLC uses endswith filtering for suffixes for some reason
                destfolder=trial.trial_path / f"it{self.iteration}",
                save_as_csv=True,
                **kwargs,
            )


@dataclass(kw_only=True)
class DlcConfigPerCam(DlcConfig):
    """DLC config information for per_cam projects"""

    _path_config_file_2: Path | str
    mode: str = "per_cam"

    @property
    def path_config_file_2(self) -> Path:
        """Ensure path_config_file always returns a Path"""
        return Path(self._path_config_file)

    @path_config_file_2.setter
    def path_config_file_2(self, value: str | Path) -> None:
        """Ensure path_config_file is always set as a Path"""
        if isinstance(value, str):
            value = Path(value)

        self._path_config_file_2 = value

    def train_network(self, **kwargs):
        """Train a DeepLabCut network"""
        deeplabcut.train_network(self.path_config_file, **kwargs)
        deeplabcut.train_network(self.path_config_file_2, **kwargs)

    def analyze_videos(self, trial: Trial, **kwargs):
        """Analyze videos with an existing DeepLabCut network"""
        csv_exists = self._configure_it_folder(trial)
        if csv_exists:
            logger.warning(
                f"There are already predicted points in iteration {self.iteration} subfolders... skipping point prediction"
            )
            return

        # Error handling handled by find_cam_file helper
        cam1_video = trial.find_cam_file(identifier="cam1")
        deeplabcut.analyze_videos(
            str(self.path_config_file),
            [
                str(cam1_video)
            ],  # DLC uses endswith filtering for suffixes for some reason
            destfolder=trial.trial_path / f"it{self.iteration}",
            save_as_csv=True,
            **kwargs,
        )

        cam2_video = trial.find_cam_file("cam2")
        deeplabcut.analyze_videos(
            str(self.path_config_file_2),
            [
                str(cam2_video)
            ],  # DLC uses endswith filtering for suffixes for some reason
            destfolder=trial.trial_path / f"it{self.iteration}",
            save_as_csv=True,
            **kwargs,
        )

    def clear_labeled_data(self, dataset_name: str) -> None:
        """Clear the labeled data folder of deepxromm data to make way for new labeled data"""
        shutil.rmtree(
            self.path_config_file.parent / "labeled-data" / f"{dataset_name}_cam1"
        )
        shutil.rmtree(
            self.path_config_file_2.parent / "labeled-data" / f"{dataset_name}_cam2"
        )


DEFAULT_CODEC = "avc1"


@dataclass
class DlcConfigRGB(DlcConfig):
    """DLC config information for RGB projects"""

    mode: str = "rgb"
    swapped_markers: bool = False
    crossed_markers: bool = False
    _video_codec: str = DEFAULT_CODEC
    bodyparts_func: Callable = get_marker_and_cam_names

    # Read-write properties
    @property
    def video_codec(self):
        """Video codec used to encode/decode videos using OpenCV"""
        return self._video_codec

    @video_codec.setter
    def video_codec(self, value: str):
        """Validate that video codec is available on system, then set if it is"""
        if not validate_codec(value):
            raise RuntimeError(f"Codec {value} is not available on this system")
        self._video_codec = value

    # Public methods
    def get_bodyparts(self, trial_csv_path: Path):
        """Return bodyparts in the format they'll need to be in for DeepLabCut"""
        return self.bodyparts_func(trial_csv_path)

    def analyze_videos(self, trial: Trial, **kwargs):
        """Analyze videos for a trial with an existing DeepLabCut network"""
        trial.make_rgb_video(codec=self.video_codec, **kwargs)
        current_files = trial.trial_path.glob("*")
        logger.debug(f"Current files in directory {current_files}")
        video_path = trial.trial_path / f"{trial.trial_name}_rgb.avi"
        deeplabcut.analyze_videos(
            str(self.path_config_file),
            str(
                video_path
            ),  # DLC relies on .endswith to determine suffix, so this needs to be a string
            destfolder=trial.trial_path / f"it{self.iteration}",
            save_as_csv=True,
            **kwargs,
        )
