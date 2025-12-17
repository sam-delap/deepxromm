"""Primary interface for training the XROMM network using DLC"""

from pathlib import Path

import deeplabcut
import pandas as pd

from .xma_data_processor import XMADataProcessor


class Network:
    """Trains an XROMM labeling network using DLC."""

    def __init__(self, config):
        self.working_dir = Path(config["working_dir"])
        self._trainingdata_path = self.working_dir / "trainingdata"  # Keep for RGB mode
        self._data_processor = XMADataProcessor(config)
        self._config = config

    def xma_to_dlc(self) -> None:
        """Convert XMAlab data to DLC format"""
        mode = self._config["mode"]
        trials = self._data_processor.list_trials("trainingdata")
        dfs, idx, pointnames = self._data_processor.read_trial_csv_with_validation(
            trials
        )

        # Validate we have enough frames
        total_frames = sum(len(x) for x in idx)
        nframes = int(self._config["nframes"])
        if total_frames < nframes:
            raise ValueError(
                f"Requested {nframes} frames but only found {total_frames} "
                f"valid frames across {len(trials)} trials"
            )

        picked_frames = self._data_processor._extract_frame_selection_loop(idx, nframes)

        cameras = [1, 2]
        if mode == "2D":
            self._process_cameras_2d(trials, picked_frames, dfs, pointnames, cameras)

        elif mode == "per_cam":
            config_files = [
                Path(self._config["path_config_file"]).parent,
                Path(self._config["path_config_file_2"]).parent,
            ]
            for camera, config_file in zip(cameras, config_files):
                self._process_camera_per_cam(
                    camera, config_file, trials, picked_frames, dfs, pointnames
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
        if self._config["mode"] == "per_cam":
            deeplabcut.create_training_dataset(self._config["path_config_file_2"])

    def train(self):
        """Starts training a network"""
        deeplabcut.train_network(
            self._config["path_config_file"], maxiters=self._config["maxiters"]
        )

        if self._config["mode"] == "per_cam":
            deeplabcut.train_network(
                self._config["path_config_file_2"], maxiters=self._config["maxiters"]
            )

    def _process_cameras_2d(
        self,
        trials: list[Path],
        picked_frames: list[list[int]],
        dfs: list[pd.DataFrame],
        pointnames: list[str],
        cameras: list[int] = [1, 2],
    ):
        """
        Process both cameras for 2D mode (nnetworks=1).

        Both cameras are combined into a single DLC project and dataset.

        Args:
            trials: List of trial directory paths
            picked_frames: List of frame indices per trial
            dfs: List of DataFrames (one per trial)
            pointnames: List of body part names
            cameras: List of cameras
        """

        config_dir = Path(self._config["path_config_file"]).parent
        dataset_name = self._config["dataset_name"]
        newpath = config_dir / "labeled-data" / dataset_name
        if newpath.exists():
            contents = list(newpath.glob("*"))
            if len(contents) > 0:
                print(
                    f"Directory {newpath} already contains data. "
                    "Please use a different dataset name or clear the directory."
                )
                return
        else:
            newpath.mkdir(parents=True, exist_ok=True)

        relnames = []
        data = pd.DataFrame()

        for camera in cameras:
            print(f"Extracting camera {camera} trial images and 2D points...")
            for trialnum, trial in enumerate(trials):
                # Find the camera video/image source
                cam_identifier = f"cam{camera}"
                source_path = self._data_processor.find_cam_file(trial, cam_identifier)

                # Sort frames to extract for given trial
                frames = sorted(picked_frames[trialnum])
                trial_relnames = self._data_processor.extract_frames_from_video(
                    source_path=source_path,
                    frame_indices=frames,
                    output_dir=newpath,
                    output_name_base=trial.name,
                    mode="2D",
                    camera=camera,
                    compression=0,
                )
                relnames.extend(trial_relnames)

                # Extract 2D points for this camera
                temp_data = self._data_processor.extract_2d_points_for_camera(
                    dfs[trialnum], camera, frames
                )

                # Reset column names for combining data from multiple cameras
                temp_data.columns = range(temp_data.shape[1])
                data = pd.concat([data, temp_data])

        # Create and save DLC dataset
        self._data_processor.save_dlc_dataset(
            data, self._config["experimenter"], relnames, pointnames, newpath
        )
        print("DLC dataset extracted from provided XMAlab trials")

    def _process_camera_per_cam(
        self,
        camera: int,
        config_dir: Path,
        trialnames: list[Path],
        picked_frames: list[list[int]],
        dfs: list[pd.DataFrame],
        pointnames: list[str],
    ) -> None:
        """
        Process single camera for per_cam mode (nnetworks=2).

        Each camera gets its own separate DLC project and dataset.

        Args:
            camera: Camera number (1 or 2)
            config_dir: DLC config file parent directory for this camera
            trialnames: List of trial directory paths
            picked_frames: List of frame indices per trial
            dfs: List of DataFrames (one per trial)
            pointnames: List of body part names
            data_processor: XMADataProcessor instance
        """
        print(f"Extracting camera {camera} trial images and 2D points...")

        # Setup output directory with camera-specific dataset name
        dataset_name = self._config["dataset_name"]
        scorer = self._config["experimenter"]
        camera_dataset_name = f"{dataset_name}_cam{camera}"
        newpath = config_dir / "labeled-data" / camera_dataset_name
        if newpath.exists():
            contents = list(newpath.glob("*"))
            if len(contents) > 0:
                print(
                    f"Directory {newpath} already contains data. "
                    "Please use a different dataset name or remove the existing dataset to update it"
                )
                return
        else:
            newpath.mkdir(parents=True, exist_ok=True)

        # Process each trial
        relnames = []
        data = pd.DataFrame()

        for trialnum, trial_path in enumerate(trialnames):
            trial_name = trial_path.name

            # Extract frames using unified interface
            frames = sorted(picked_frames[trialnum])
            # Find the camera video/image source
            cam_identifier = f"cam{camera}"
            cam_file = self._data_processor.find_cam_file(trial_path, cam_identifier)
            source_path = trial_path / cam_file

            trial_relnames = self._data_processor.extract_frames_from_video(
                source_path=source_path,
                frame_indices=frames,
                output_dir=newpath,
                output_name_base=trial_name,
                mode="per_cam",
                camera=camera,
                compression=0,
            )
            relnames.extend(trial_relnames)

            # Extract 2D points for this camera
            temp_data = self._data_processor.extract_2d_points_for_camera(
                dfs[trialnum], camera, frames
            )
            data = pd.concat([data, temp_data])

        # Create and save DLC dataset
        self._data_processor.save_dlc_dataset(
            data, scorer, relnames, pointnames, newpath
        )
        print("DLC dataset extracted from provided XMAlab trials")
