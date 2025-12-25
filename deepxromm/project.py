"""
This module is responsible for creating and updating deepxromm projects
"""

from abc import ABC
from dataclasses import dataclass
import tempfile
from pathlib import Path
import warnings

import cv2
import deeplabcut
import numpy as np
import pandas as pd
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from deepxromm.augmenter import OutlierExtractionParams
from deepxromm.config_utilities import load_config_file, save_config_file
from deepxromm.xma_data_processor import XMADataProcessor
from deepxromm.logging import logger
from deepxromm.autocorrector import AutocorrectParams

# Sets the default codec for use in creating/loading project configs
DEFAULT_CODEC = "avc1"


@dataclass
class Project(ABC):
    """Parent class for project configs - can't be instantiated"""

    task: str
    _experimenter: str
    _working_dir: Path
    _path_config_file: Path
    _path_config_file_2: Path | None = (
        None  # Remove once nothing acts on this param outside of per_cam stuff
    )
    _dataset_name: str = "MyData"
    nframes: int = 0
    maxiters: int = 150000
    tracking_threshold: float = 0.1
    _video_codec: str = DEFAULT_CODEC
    _mode: str = ""
    autocorrect_settings: AutocorrectParams = AutocorrectParams()
    augmenter_settings: OutlierExtractionParams = OutlierExtractionParams()
    cam1s_are_the_same_view: bool = True

    @property
    def working_dir(self):
        """Ensuring path_config_file is always returned as a path"""
        return Path(self._working_dir)

    @working_dir.setter
    def working_dir(self, value):
        """Ensuring path_config_file is always returned as a path"""
        self._working_dir = value

    @property
    def path_config_file(self):
        """Ensuring path_config_file is always returned as a path"""
        return Path(self._path_config_file)

    @path_config_file.setter
    def path_config_file(self, value):
        """Ensuring path_config_file is always returned as a path"""
        self._path_config_file = value

    @property
    def dataset_name(self):
        """Name of the dataset that will be created within DeepLabCut"""
        if self._dataset_name == "MyData":
            logger.warning("Default project name in use")

        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value: str):
        self._dataset_name = value

    @property
    def video_codec(self):
        """Video codec used to encode/decode videos using OpenCV"""
        return self._video_codec

    @property
    def project_config_path(self):
        """Path to project_config.yaml file"""
        return Path(self.working_dir / "project_config.yaml")

    @video_codec.setter
    def video_codec(self, value: str):
        """Validate that video codec is available on system, then set if it is"""
        if not _validate_codec(value):
            raise RuntimeError(f"Codec {value} is not available on this system")
        self._video_codec = value
        self.update_config_file()

    # Read-only properties
    @property
    def mode(self):
        return self._mode

    @property
    def experimenter(self):
        return self._experimenter

    @staticmethod
    def load_config_file(config_file_path: Path):
        """Load a YAML file as a commented map"""
        yaml = YAML()
        with open(config_file_path, "r") as fp:
            config = yaml.load(fp)

        return config

    @staticmethod
    def save_config_file(config_data: CommentedMap, config_file_path: Path):
        """Load a YAML file as a commented map"""
        yaml = YAML()
        with open(config_file_path, "w") as fp:
            config = yaml.dump(config_data, fp)

        return config

    def check_config_for_updates(self):
        """Check the config for updates and update any values that have changed."""
        if not self.project_config_path.exists():
            logger.debug(
                "Didn't find project config this time around. I'm sure this is fine..."
            )
            return

        config_data = load_config_file(self.project_config_path)

        # Experimental params (mode and experimenter are read-only)
        self.task = config_data["task"]
        self.working_dir = Path(config_data["working_dir"])
        self.path_config_file = Path(config_data["path_config_file"])
        self.nframes = config_data["nframes"]
        self.maxiters = config_data["maxiters"]
        self.tracking_threshold = config_data["tracking_threshold"]

        # Autocorrect settings
        self.autocorrect_settings.search_area = config_data["search_area"]
        self.autocorrect_settings.threshold = config_data["threshold"]
        self.autocorrect_settings.krad = config_data["krad"]
        self.autocorrect_settings.gsigma = config_data["gsigma"]
        self.autocorrect_settings.img_wt = config_data["img_wt"]
        self.autocorrect_settings.blur_wt = config_data["blur_wt"]
        self.autocorrect_settings.gamma = config_data["gamma"]

        # Autocorrect testing params
        self.autocorrect_settings.trial_name = config_data["trial_name"]
        self.autocorrect_settings.cam = config_data["cam"]
        self.autocorrect_settings.frame_num = config_data["frame_num"]
        self.autocorrect_settings.marker = config_data["marker"]
        self.autocorrect_settings.test_autocorrect = config_data["test_autocorrect"]

        # Video similarity
        self.cam1s_are_the_same_view = config_data["cam1s_are_the_same_view"]
        self.video_codec = config_data["video_codec"]

        # Retraining
        self.augmenter_settings.outlier_algorithm = config_data["augmenter"][
            "outlier_algorithm"
        ]
        self.augmenter_settings.extraction_algorithm = config_data["augmenter"][
            "extraction_algorithm"
        ]

    def update_config_file(self):
        """Update the config to the values of the current object"""
        if self.project_config_path.exists():
            config_data = load_config_file(self.project_config_path)
        else:
            config_data = load_config_file(
                Path(__file__).parent / "default_config.yaml"
            )

        # Experimental params
        config_data = _migrate_tracking_mode(config_data)
        config_data["task"] = self.task
        config_data["experimenter"] = self.experimenter
        config_data["working_dir"] = str(self.working_dir)
        config_data["path_config_file"] = str(self.path_config_file)
        config_data["nframes"] = self.nframes
        config_data["maxiters"] = self.maxiters
        config_data["tracking_threshold"] = self.tracking_threshold
        config_data["mode"] = self.mode

        # Autocorrect settings
        config_data["search_area"] = self.autocorrect_settings.search_area
        config_data["threshold"] = self.autocorrect_settings.threshold
        config_data["krad"] = self.autocorrect_settings.krad
        config_data["gsigma"] = self.autocorrect_settings.gsigma
        config_data["img_wt"] = self.autocorrect_settings.img_wt
        config_data["blur_wt"] = self.autocorrect_settings.blur_wt
        config_data["gamma"] = self.autocorrect_settings.gamma

        # Autocorrect testing params
        config_data["trial_name"] = self.autocorrect_settings.trial_name
        config_data["cam"] = self.autocorrect_settings.cam
        config_data["frame_num"] = self.autocorrect_settings.frame_num
        config_data["marker"] = self.autocorrect_settings.marker
        config_data["test_autocorrect"] = self.autocorrect_settings.test_autocorrect

        # Video similarity
        config_data["cam1s_are_the_same_view"] = self.cam1s_are_the_same_view
        config_data["video_codec"] = self.video_codec

        # Retraining
        config_data["augmenter"]["outlier_algorithm"] = (
            self.augmenter_settings.outlier_algorithm.value
        )
        config_data["augmenter"]["extraction_algorithm"] = (
            self.augmenter_settings.extraction_algorithm.value
        )

        save_config_file(config_data, self.project_config_path)


@dataclass
class Project2D(Project):
    _mode: str = "2D"

    def __post_init__(self):
        """After initializing, check the config and update if necessary"""
        self.check_config_for_updates()


@dataclass
class ProjectPerCam(Project):
    _path_config_file_2: Path
    _mode: str = "per_cam"

    def __post_init__(self):
        """After initializing, check the config and update if necessary"""
        self.check_config_for_updates()

    def check_config_for_updates(self):
        """Check the config for updates and update any values that have changed."""
        if not self.project_config_path.exists():
            logger.debug(
                "Didn't find project config this time around. I'm sure this is fine..."
            )
            return

        super().check_config_for_updates()
        config_data = load_config_file(self.project_config_path)

        self.path_config_file_2 = Path(config_data["path_config_file_2"])

    def update_config_file(self):
        """Update the config to the values of the current object"""
        super().update_config_file()
        config_data = load_config_file(self.project_config_path)

        # Experimental params
        config_data["path_config_file_2"] = str(self.path_config_file_2)

        save_config_file(config_data, self.project_config_path)

    @property
    def path_config_file_2(self):
        """Ensuring path_config_file is always returned as a path"""
        return Path(self._path_config_file_2)

    @path_config_file_2.setter
    def path_config_file_2(self, value):
        """Ensuring path_config_file is always returned as a path"""
        self._path_config_file_2 = value


@dataclass
class ProjectRGB(Project):
    _mode: str = "rgb"
    swapped_markers: bool = False
    crossed_markers: bool = False

    def __post_init__(self):
        """After initializing, check the config and update if necessary"""
        self.check_config_for_updates()

    def check_config_for_updates(self):
        """Check the config for updates and update any values that have changed."""
        if not self.project_config_path.exists():
            logger.debug(
                "Didn't find project config this time around. I'm sure this is fine..."
            )
            return

        super().check_config_for_updates()
        config_data = load_config_file(self.project_config_path)

        # Experimental params (mode and experimenter are read-only)
        self.swapped_markers = config_data["swapped_markers"]
        self.crossed_markers = config_data["crossed_markers"]

    def update_config_file(self):
        """Update the config to the values of the current object"""
        super().update_config_file()
        config_data = load_config_file(self.project_config_path)

        # Experimental params
        config_data["swapped_markers"] = self.swapped_markers
        config_data["crossed_markers"] = self.crossed_markers

        save_config_file(config_data, self.project_config_path)


class ProjectFactory:
    _PROJECT_MODES = ["2D", "per_cam", "rgb"]

    def __init__(self):
        raise NotImplementedError("Use create_new_config or load_config instead.")

    @classmethod
    def create_new_config(
        cls,
        working_dir: str | Path = Path.cwd(),
        experimenter="NA",
        mode="2D",
        codec=DEFAULT_CODEC,
    ) -> Project:
        """Creates a new config from scratch."""
        if isinstance(working_dir, str):
            working_dir = Path(working_dir)
        (working_dir / "trainingdata").mkdir(parents=True, exist_ok=True)
        (working_dir / "trials").mkdir(parents=True, exist_ok=True)

        # Create a fake video to pass into the deeplabcut workflow
        dummy_video_path = working_dir / "dummy.avi"
        frame = np.zeros((480, 480, 3), dtype=np.uint8)
        out = cv2.VideoWriter(
            str(dummy_video_path), cv2.VideoWriter_fourcc(*codec), 15, (480, 480)
        )
        out.write(frame)
        out.release()

        # Create a new DLC project
        task = working_dir.name
        path_config_file = deeplabcut.create_new_project(
            task,
            experimenter,
            [str(dummy_video_path)],
            str(working_dir / ""),  # Add the trailing slash
            copy_videos=True,
        )

        path_config_file_2 = None
        if mode == "per_cam":
            task_2 = f"{task}_cam2"
            path_config_file_2 = deeplabcut.create_new_project(
                task_2,
                experimenter,
                [str(dummy_video_path)],
                str(working_dir / ""),  # Add the trailing slash
                copy_videos=True,
            )

        # Instantiate project
        project = cls._instantiate_project(
            mode,
            task,
            experimenter,
            working_dir,
            path_config_file,
            path_config_file_2,
            codec,
        )

        # Cleanup
        try:
            (Path(path_config_file).parent / "labeled-data/dummy").rmdir()
        except FileNotFoundError:
            pass

        try:
            (Path(path_config_file).parent / "videos/dummy.avi").unlink()
        except FileNotFoundError:
            pass

        try:
            dummy_video_path.unlink()
        except FileNotFoundError:
            pass

        project.update_config_file()

        return project

    @classmethod
    def load_config(cls, working_dir: str | Path = Path.cwd()):
        """Load an existing project"""
        if isinstance(working_dir, str):
            working_dir = Path(working_dir)

        # Open the config
        config = load_config_file(working_dir / "project_config.yaml")
        config = _migrate_tracking_mode(config)

        # Extract values necessary to instantiate the project
        task = config["task"]
        experimenter = config["experimenter"]
        working_dir = Path(config["working_dir"])
        path_config_file = Path(config["path_config_file"])
        mode = config["mode"]
        codec = config["video_codec"]

        path_config_file_2 = None
        if "path_config_file_2" in config:
            path_config_file_2 = Path(config["path_config_file_2"])

        project = cls._instantiate_project(
            mode,
            task,
            experimenter,
            working_dir,
            path_config_file,
            path_config_file_2,
            codec,
        )

        # Initiate data processor utility
        data_processor = XMADataProcessor(project=project)
        training_trials = data_processor.list_trials("trainingdata")
        if len(training_trials) == 0:
            raise FileNotFoundError(
                "Empty trials directory found. Expected trial folders within the 'trainingdata' directory"
            )
        trial_path = training_trials[0]

        # Load trial CSV
        trial_csv_path = data_processor.find_trial_csv(trial_path)
        trial_csv = pd.read_csv(trial_csv_path)

        # Drop untracked frames (all NaNs)
        trial_csv = trial_csv.dropna(how="all")

        # Make sure there aren't any partially tracked frames
        if trial_csv.isna().sum().sum() > 0:
            raise AttributeError(
                f"Detected {len(trial_csv) - len(trial_csv.dropna())} partially tracked frames. Please ensure that all frames are completely tracked"
            )

        # Check/set the default value for tracked frames
        if project.nframes <= 0:
            project.nframes = len(trial_csv)

        elif project.nframes != len(trial_csv):
            logger.warning(
                "Project nframes tracked does not match 2D Points file. If this is intentional, ignore this message"
            )

        # Check the current nframes against the threshold value * the number of frames in the cam1 video
        cam1_video_path = data_processor.find_cam_file(trial_path, "cam1")
        video = cv2.VideoCapture(cam1_video_path)

        if (
            project.nframes
            < int(video.get(cv2.CAP_PROP_FRAME_COUNT)) * project.tracking_threshold
        ):
            tracking_threshold = project.tracking_threshold
            logger.warning(
                f"Project nframes is less than the recommended {tracking_threshold * 100}% of the total frames"
            )

        # Check DLC bodyparts (marker names)
        default_bodyparts = ["bodypart1", "bodypart2", "bodypart3", "objectA"]
        bodyparts = project.dlc_config.get_bodyparts(trial_csv_path)

        dlc_yaml = load_config_file(project.path_config_file)
        dlc_bodyparts = dlc_yaml["bodyparts"]
        logger.debug(f"DLC bodyparts: {dlc_bodyparts}")

        if dlc_yaml["bodyparts"] == default_bodyparts:
            dlc_yaml["bodyparts"] = bodyparts
        elif dlc_yaml["bodyparts"] != bodyparts:
            raise SyntaxError(
                "XMAlab CSV marker names are different than DLC bodyparts."
            )

        save_config_file(dlc_yaml, project.path_config_file)

        # Check DLC bodyparts (marker names) for config 2 if needed
        if project.mode == "per_cam":
            dlc_yaml = load_config_file(project.path_config_file_2)
            # Better conditional logic could definitely be had to reduce function calls here
            if dlc_yaml["bodyparts"] == default_bodyparts:
                dlc_yaml["bodyparts"] = bodyparts
            elif dlc_yaml["bodyparts"] != bodyparts:
                raise SyntaxError(
                    "XMAlab CSV marker names are different than DLC bodyparts."
                )

            save_config_file(dlc_yaml, project.path_config_file_2)

        project.update_config_file()

        return project

    @classmethod
    def _instantiate_project(
        cls,
        mode: str,
        task: str,
        experimenter: str,
        working_dir: Path,
        path_config_file: Path,
        path_config_file_2: Path | None = None,
        codec: str = DEFAULT_CODEC,
    ):
        """Instantiate new project"""
        match mode:
            case "2D":
                project = Project2D(
                    task,
                    experimenter,
                    working_dir,
                    _path_config_file=path_config_file,
                    _video_codec=codec,
                )
            case "per_cam":
                if path_config_file_2 is None:
                    raise ValueError(
                        "Please specify a value for 2nd DLC project config. Value is currently unset"
                    )
                project = ProjectPerCam(
                    task,
                    experimenter,
                    working_dir,
                    _path_config_file_2=path_config_file_2,
                    _path_config_file=path_config_file,
                    _video_codec=codec,
                )
            case "rgb":
                project = ProjectRGB(
                    task,
                    experimenter,
                    working_dir,
                    _path_config_file=path_config_file,
                    _video_codec=codec,
                )
            case _:
                raise ValueError(
                    f"Unsupported mode {mode}. Valid modes: {cls._PROJECT_MODES}"
                )

        return project


def _migrate_tracking_mode(config: dict):
    """Migrate deprecated 'tracking_mode' to 'mode' with backwards compatibility.

    This function handles the transition from 'tracking_mode' to 'mode' introduced
    in version 0.2.5. Support for 'tracking_mode' will be removed in version 1.0.

    Args:
        config: Dictionary containing project configuration

    Returns:
        Modified config dictionary with migration applied

    Raises:
        ValueError: If both keys exist with conflicting values

    Note:
        Support for 'tracking_mode' will be removed in version 1.0.
    """
    has_mode = "mode" in config
    has_tracking_mode = "tracking_mode" in config

    if has_mode and has_tracking_mode:
        if config["mode"] != config["tracking_mode"]:
            raise ValueError(
                f"Conflicting values detected in config: 'mode' is set to "
                f"'{config['mode']}' but 'tracking_mode' is set to "
                f"'{config['tracking_mode']}'. Please remove the deprecated "
                f"'tracking_mode' key from your config and use only 'mode'."
            )
        else:
            warnings.warn(
                "Config contains both 'mode' and 'tracking_mode' with the same "
                "value. Removing deprecated 'tracking_mode'. This key will be "
                "removed in version 1.0.",
                DeprecationWarning,
                stacklevel=3,
            )
            del config["tracking_mode"]
    elif has_tracking_mode:
        warnings.warn(
            "'tracking_mode' is deprecated and will be removed in version 1.0. "
            "Use 'mode' instead. Automatically migrating your config...",
            DeprecationWarning,
            stacklevel=3,
        )
        config["mode"] = config["tracking_mode"]
        del config["tracking_mode"]

    return config


def _validate_codec(codec: str, width: int = 640, height: int = 480) -> bool:
    """
    Validate if a video codec is available on the current system.

    Args:
        codec: FourCC codec code (e.g., "avc1", "DIVX", "XVID")
        width: Test video width (default 640)
        height: Test video height (default 480)

    Returns:
        True if codec is available and functional, False otherwise

    Note:
        Special cases "uncompressed" and 0 always return True as they
        use different encoding mechanisms.
    """
    # Special cases that don't use cv2.VideoWriter with fourcc
    if codec == "uncompressed" or codec == 0:
        return True

    # Test codec by creating a temporary VideoWriter
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            test_path = Path(tmpdir) / "codec_test.avi"
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(test_path),
                fourcc,
                1.0,  # Low FPS for test
                (width, height),
            )

            # Check if writer opened successfully
            is_valid = writer.isOpened()

            # Try writing a test frame to ensure codec actually works
            if is_valid:
                test_frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(test_frame)

            writer.release()
            logger.debug(
                f"Codec validation for '{codec}': {'PASSED' if is_valid else 'FAILED'}"
            )
            return is_valid

        except Exception as e:
            logger.error(f"Codec validation for '{codec}' failed with exception: {e}")
            return False


def _get_codec_error_message(failed_codec: str, operation: str) -> str:
    """
    Generate descriptive error message for codec validation failure.

    Args:
        failed_codec: The codec that failed validation
        operation: Operation type ("split" or "merge")

    Returns:
        Formatted error message with suggestions
    """
    return f"""
Video codec '{failed_codec}' is not available on this system for {operation}_rgb operation.

Common alternative codecs to try:
- "avc1"  : H.264 codec (best quality, not always available)
- "DIVX"  : DivX codec (widely available)
- "XVID"  : Xvid codec (widely available)
- "mp4v"  : MPEG-4 codec (generally available)
- "MJPG"  : Motion JPEG (always available, larger file sizes)
- "uncompressed" : Raw video via ffmpeg (largest files, highest quality)

To change the codec, update your project config file:
video_codec: "DIVX"  # Change this line to one of the alternatives above

Note: Codec availability depends on your OpenCV build and system codecs.
""".strip()
