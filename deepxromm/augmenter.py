from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import re
import shutil
import yaml

import deeplabcut
import pandas as pd

from deepxromm.xma_data_processor import XMADataProcessor
from deepxromm.logging import logger


class OutlierAlgorithm(Enum):
    """Enum class to define outlier algorithm"""

    FITTING = "fitting"
    JUMP = "jump"
    UNCERTAIN = "uncertain"
    LIST = "list"  # User must specify 'frames2use'
    # Does not support manual outlier extraction due to need for DeepLabCut GUI


class ExtractionAlgorithm(Enum):
    """Enum class to define extraction algorithm"""

    UNIFORM = "uniform"
    KMEANS = "kmeans"


@dataclass
class OutlierExtractionParams:
    _outlier_algorithm: str = "jump"
    _extraction_algorithm: str = "kmeans"

    @property
    def outlier_algorithm(self):
        return OutlierAlgorithm(self._outlier_algorithm)

    @property
    def extraction_algorithm(self):
        return ExtractionAlgorithm(self._extraction_algorithm)


class Augmenter:
    """Augments training data with DLC"""

    def __init__(self, project):
        self.nframes = project.nframes
        self.working_dir = project.working_dir
        self.mode = project.mode
        self.augmenter_settings = project.augmenter_settings
        self.path_config_file = project.path_config_file
        self.path_config_file_2 = (
            project.path_config_file_2 if project.mode == "per_cam" else None
        )
        self._data_processor = XMADataProcessor(project)

        with self.path_config_file.open("r") as fp:
            dlc_config = yaml.safe_load(fp)
        self._iteration = int(dlc_config["iteration"])

    def extract_outlier_frames(self, **kwargs) -> None:
        """Extract outlier frames from DLC output and store them in 'outliers.yaml' file for the user"""
        if (
            self.augmenter_settings.outlier_algorithm == OutlierAlgorithm.LIST
            and "frames2use" not in kwargs
        ):
            raise ValueError(
                "Using the 'list' outlieralgorithm with extract_frames requires specifying the 'frames2use' parameter so that DLC can extract a user-defined list of frames"
            )

        for trial_path in self._data_processor.list_trials():
            if self.mode == "rgb":
                self._extract_outlier_frames_rgb(trial_path, **kwargs)
            else:
                self._extract_outlier_frames_2cam(trial_path, **kwargs)

    def merge_datasets(self, update_nframes=True, update_init_weights=True):
        """Create a refined dataset that includes existing training data and outliers"""
        training_trials = self._data_processor.list_trials("trainingdata")
        training_trial_names = [trial.name for trial in training_trials]
        for trial_path in self._data_processor.list_trials():
            analysis_path = trial_path / f"it{self._iteration}"
            with open(analysis_path / "outliers.yaml", "r") as fp:
                outliers = yaml.safe_load(fp)

            # Convert outliers back to 0-indexed because we're working with a DataFrame
            outliers = [outlier - 1 for outlier in outliers]

            outlier_csv_path = self._data_processor.find_trial_csv(
                analysis_path, "outliers"
            )

            outlier_csv = pd.read_csv(outlier_csv_path)
            outlier_csv = outlier_csv.loc[outliers, :].reset_index(drop=True)

            # Assumes that the trial is named the same whether it is in novel trials or in trainingdata to DLC extraction
            training_trial_path = self.working_dir / "trainingdata" / trial_path.name
            if trial_path.name in training_trial_names:
                self._merge_existing_trial_with_outlier_data(
                    training_trial_path, outlier_csv
                )
            else:
                self._create_new_trial_with_outlier_data(
                    trial_path, training_trial_path, outlier_csv
                )
            self.nframes = self.nframes + len(outlier_csv)

        # Update DLC iteration
        with open(self.path_config_file, "r") as fp:
            dlc_config = yaml.safe_load(fp)
        next_iteration = self._iteration + 1
        dlc_config["iteration"] = next_iteration
        with open(self.path_config_file, "w") as fp:
            yaml.safe_dump(dlc_config, fp, sort_keys=False)

        logger.info(
            f"DeepLabCut training iteration updated from {self._iteration} to {next_iteration}"
        )

        # Update nframes in config
        if not update_nframes:
            logger.info(
                "User has specified not to update nframes. Please update nframes in project_config.yaml to ensure all outlier frames are included in training data"
            )
            return

        logger.info("Updating nframes to include newly tracked outlier data...")
        with open(self.working_dir / "project_config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        config["nframes"] = self.nframes
        with open(self.working_dir / "project_config.yaml", "w") as fp:
            config = yaml.safe_dump(config, fp, sort_keys=False)

        logger.info(
            f"nframes updated to include new outlier data. New nframes for training data: {self.nframes}"
        )

    def _extract_outlier_frames_rgb(self, trial_path: Path, **kwargs):
        """Extract outlier frames for RGB projects"""
        analysis_path = trial_path / f"it{self._iteration}"
        cam_file = self._data_processor.find_cam_file(trial_path, "rgb")
        outliers = self._get_outliers_for_camera(
            cam_file, self.path_config_file, **kwargs
        )
        with open(analysis_path / "outliers.yaml", "w") as fp:
            yaml.safe_dump(outliers, fp)

    def _extract_outlier_frames_2cam(self, trial_path: Path, **kwargs):
        """Extract outlier frames for 2-cam projects"""
        analysis_path = trial_path / f"it{self._iteration}"
        outliers = {}
        for camera in ["cam1", "cam2"]:
            cam_file = self._data_processor.find_cam_file(trial_path, camera)
            if camera == "cam2" and self.path_config_file_2 is not None:
                outliers[camera] = self._get_outliers_for_camera(
                    cam_file, self.path_config_file_2, **kwargs
                )
            else:
                outliers[camera] = self._get_outliers_for_camera(
                    cam_file, self.path_config_file, **kwargs
                )
            with open(analysis_path / f"{camera}_outliers.yaml", "w") as fp:
                yaml.safe_dump(outliers[camera], fp)
        matched_outliers = [
            matched_outlier
            for matched_outlier in outliers["cam1"]
            if matched_outlier in outliers["cam2"]
        ]
        if len(matched_outliers) == 0:
            logger.warning(
                "No outliers matched. Concatenating cam1 and cam2 outliers in matched outliers"
            )
            matched_outliers = outliers["cam1"] + outliers["cam2"]
        with open(analysis_path / "outliers.yaml", "w") as fp:
            yaml.safe_dump(matched_outliers, fp)

    def _get_outliers_for_camera(
        self, cam_file: Path, path_config_file: Path, **kwargs
    ):
        """Get outliers for a single camera of a project"""
        # DLC's looking for the path we saved its output to
        analysis_path = cam_file.parent / f"it{self._iteration}"
        # DLC will save these to labeled-data/<video_name>
        deeplabcut.extract_outlier_frames(
            path_config_file,
            [str(cam_file)],
            outlieralgorithm=self.augmenter_settings.outlier_algorithm.value,
            extractionalgorithm=self.augmenter_settings.extraction_algorithm.value,
            destfolder=analysis_path,
            automatic=True,
            **kwargs,
        )
        vid_dataset_path = self.path_config_file.parent / "labeled-data" / cam_file.stem
        images = list(vid_dataset_path.glob("img*.png"))
        outliers = []
        for img in images:
            img_name = img.name
            match = re.search(r"img(\d+)\.png", img_name)
            if match is None:
                raise ValueError(
                    "Couldn't parse frame number from image files. Something wrong with DeepLabCut?"
                )
            outliers.append(int(match.group(1)) + 1)
        return outliers  # Match DLC index with the way it will show up in XMAlab
        # Remove DLC-generated labeled-data folders

    def _merge_existing_trial_with_outlier_data(
        self, training_trial_path: Path, outlier_csv: pd.DataFrame
    ):
        """Merge outlier data into existing trial CSV"""
        current_trial_csv_path = self._data_processor.find_trial_csv(
            training_trial_path
        )
        current_trial_csv = pd.read_csv(current_trial_csv_path)
        df_combined = pd.concat([current_trial_csv, outlier_csv])
        df_combined.sort_index(inplace=True)
        df_combined_unique = df_combined.drop_duplicates()
        df_combined_unique.to_csv(current_trial_csv_path, index=False)

    def _create_new_trial_with_outlier_data(
        self, trial_path: Path, training_trial_path: Path, outlier_csv: pd.DataFrame
    ):
        """Create new trial for outlier data from trial that was not part of original training data"""
        training_trial_path.mkdir()

        # Copy data in in XMAlab format
        if self.mode == "rgb":
            rgb_cam_file = self._data_processor.find_cam_file(trial_path, "rgb")
            shutil.copy(str(rgb_cam_file), str(training_trial_path / rgb_cam_file.name))
        else:
            cam1_cam_file = self._data_processor.find_cam_file(trial_path, "cam1")
            cam2_cam_file = self._data_processor.find_cam_file(trial_path, "cam2")
            shutil.copy(
                str(cam1_cam_file), str(training_trial_path / cam1_cam_file.name)
            )
            shutil.copy(
                str(cam2_cam_file), str(training_trial_path / cam2_cam_file.name)
            )

        outlier_csv.to_csv(
            training_trial_path / f"{training_trial_path.name}.csv",
            na_rep="NaN",
            index=False,
        )
