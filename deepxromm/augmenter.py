from enum import Enum
from pathlib import Path
import re
import shutil
import yaml

import deeplabcut
import pandas as pd
import numpy as np

from deepxromm.xma_data_processor import XMADataProcessor


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


class Augmenter:
    """Augments training data with DLC"""

    def __init__(self, config: dict):
        self.nframes = config["nframes"]
        self.working_dir = Path(config["working_dir"])
        self.mode = config["mode"]
        if self.mode not in ["2D", "per_cam", "rgb"]:
            raise ValueError(f"Unsupported mode: {self.mode}")
        self.outlier_algorithm = OutlierAlgorithm(
            config["augmenter"]["outlier_algorithm"]
        )
        self.extraction_algorithm = ExtractionAlgorithm(
            config["augmenter"]["extraction_algorithm"]
        )
        self.path_config_file = Path(config["path_config_file"])
        self.path_config_file_2 = (
            Path(config["path_config_file_2"]) if self.mode == "per_cam" else None
        )
        self._data_processor = XMADataProcessor(config)

        with self.path_config_file.open("r") as fp:
            dlc_config = yaml.safe_load(fp)
        self._iteration = int(dlc_config["iteration"])

    def extract_outlier_frames(self, **kwargs) -> None:
        """Extract outlier frames from DLC output and store them in 'outliers.yaml' file for the user"""
        if (
            self.outlier_algorithm == OutlierAlgorithm.LIST
            and "frames2use" not in kwargs
        ):
            raise ValueError(
                "Using the 'list' outlieralgorithm with extract_frames requires specifying the 'frames2use' parameter so that DLC can extract a user-defined list of frames"
            )

        for trial_path in self._data_processor.list_trials():
            # DLC's looking for the path we saved its output to
            analysis_path = trial_path / f"it{self._iteration}"
            if self.mode == "rgb":
                cam_file = self._data_processor.find_cam_file(trial_path, "rgb")
                # DLC will save these to labeled-data/<video_name> in its
                deeplabcut.extract_outlier_frames(
                    self.path_config_file,
                    [str(cam_file)],
                    outlieralgorithm=self.outlier_algorithm.value,
                    extractionalgorithm=self.extraction_algorithm.value,
                    destfolder=analysis_path,
                    automatic=True,
                    **kwargs,
                )
                vid_dataset_path = (
                    self.path_config_file.parent / "labeled-data" / cam_file.stem
                )
                images = list(vid_dataset_path.glob("img*.png"))
                outliers = []
                for img in images:
                    img_name = img.name
                    match = re.search(r"img(\d+)\.png", img_name)
                    if match is None:
                        raise ValueError(
                            "Couldn't parse frame number from image files. Something wrong with DeepLabCut?"
                        )
                    outliers.append(
                        int(match.group(1)) + 1
                    )  # Match DLC index with the way it will show up in XMAlab
                with open(analysis_path / "outliers.yaml", "w") as fp:
                    yaml.safe_dump(outliers, fp)
            else:

                outliers = {}
                for camera in ["cam1", "cam2"]:
                    cam_file = self._data_processor.find_cam_file(trial_path, camera)
                    if camera == "cam2" and self.path_config_file_2 is not None:
                        deeplabcut.extract_outlier_frames(
                            self.path_config_file_2,
                            [str(cam_file)],
                            outlieralgorithm=self.outlier_algorithm.value,
                            extractionalgorithm=self.extraction_algorithm.value,
                            destfolder=analysis_path,
                            automatic=True,
                            **kwargs,
                        )
                    else:
                        deeplabcut.extract_outlier_frames(
                            self.path_config_file,
                            [str(cam_file)],
                            outlieralgorithm=self.outlier_algorithm.value,
                            extractionalgorithm=self.extraction_algorithm.value,
                            destfolder=analysis_path,
                            automatic=True,
                            **kwargs,
                        )
                    vid_dataset_path = (
                        self.path_config_file.parent / "labeled-data" / cam_file.stem
                    )
                    images = list(vid_dataset_path.glob("img*.png"))
                    # Image numbers are off-by-one from XMAlab frame numbers, because
                    # they are 0-indexed and XMAlab frame numbers are one-indexed
                    outliers[camera] = []
                    for img in images:
                        img_name = img.name
                        match = re.search(r"img(\d+)\.png", img_name)
                        if match is None:
                            raise ValueError(
                                "Couldn't parse frame number from image files. Something wrong with DeepLabCut?"
                            )
                        outliers[camera].append(
                            int(match.group(1)) + 1
                        )  # Match DLC index with XMAlab frame number
                    with open(analysis_path / f"{camera}_outliers.yaml", "w") as fp:
                        yaml.safe_dump(outliers[camera], fp)
                matched_outliers = [
                    matched_outlier
                    for matched_outlier in outliers["cam1"]
                    if matched_outlier in outliers["cam2"]
                ]
                if len(matched_outliers) == 0:
                    print(
                        "No outliers matched. Concatenating cam1 and cam2 outliers in matched outliers"
                    )
                    matched_outliers = outliers["cam1"] + outliers["cam2"]
                with open(analysis_path / "outliers.yaml", "w") as fp:
                    yaml.safe_dump(matched_outliers, fp)
        # Remove DLC-generated labeled-data folders

    def merge_datasets(self, update_nframes=True, update_init_weights=True):
        """Create a refined dataset that includes existing training data and outliers"""
        training_trials = self._data_processor.list_trials("trainingdata")
        training_trial_names = [trial.name for trial in training_trials]
        for trial_path in self._data_processor.list_trials():
            analysis_path = trial_path / f"it{self._iteration}"
            with open(analysis_path / "outliers.yaml", "r") as fp:
                outliers = yaml.safe_load(fp)

            # Convert outliers back to 0-indexed because we're working with a DataFrame
            if len(outliers) < 1000:
                # Use list comprehension for smaller datasets
                outliers = [outlier - 1 for outlier in outliers]
            else:
                # Use np.array() for larger datasets (performance boost)
                outliers = np.array(outliers) - 1

            outlier_csv_path = self._data_processor.find_trial_csv(
                analysis_path, "outliers"
            )

            outlier_csv = pd.read_csv(outlier_csv_path)

            # Trim outlier CSV down to just outlier rows
            outlier_csv = outlier_csv.loc[outliers, :].reset_index(drop=True)

            # Copy data around as-needed to set up the trainingdata for follow-on XMA to DLC extraction
            # Assumes that the trial is named the same whether it is in novel trials or in trainingdata
            if trial_path.name in training_trial_names:
                # Trial already exists, merge outlier data into current CSV
                current_trial_csv_path = self._data_processor.find_trial_csv(
                    self.working_dir / "trainingdata" / trial_path.name
                )
                current_trial_csv = pd.read_csv(current_trial_csv_path)
                df_combined = pd.concat([current_trial_csv, outlier_csv])
                df_combined.sort_index(inplace=True)
                df_combined_unique = df_combined.drop_duplicates()
                df_combined_unique.to_csv(current_trial_csv_path, index=False)
            else:
                # Trial doesn't exist, create the new trial folder and copy data in
                # Create new training trial folder
                new_training_trial = self.working_dir / "trainingdata" / trial_path.name
                new_training_trial.mkdir()

                # Copy data in in XMAlab format
                if self.mode == "rgb":
                    rgb_cam_file = self._data_processor.find_cam_file(trial_path, "rgb")
                    shutil.copy(
                        str(rgb_cam_file), str(new_training_trial / rgb_cam_file.name)
                    )
                else:
                    cam1_cam_file = self._data_processor.find_cam_file(
                        trial_path, "cam1"
                    )
                    cam2_cam_file = self._data_processor.find_cam_file(
                        trial_path, "cam2"
                    )
                    shutil.copy(
                        str(cam1_cam_file), str(new_training_trial / cam1_cam_file.name)
                    )
                    shutil.copy(
                        str(cam2_cam_file), str(new_training_trial / cam2_cam_file.name)
                    )

                outlier_csv.to_csv(
                    new_training_trial / f"{new_training_trial.name}.csv",
                    na_rep="NaN",
                    index=False,
                )
            self.nframes = self.nframes + len(outlier_csv)

        # Update DLC iteration
        with open(self.path_config_file, "r") as fp:
            dlc_config = yaml.safe_load(fp)
        next_iteration = self._iteration + 1
        dlc_config["iteration"] = next_iteration
        with open(self.path_config_file, "w") as fp:
            yaml.safe_dump(dlc_config, fp, sort_keys=False)

        print(
            f"DeepLabCut training iteration updated from {self._iteration} to {next_iteration}"
        )

        # Update nframes in config
        if not update_nframes:
            print(
                "User has specified not to update nframes. Please update nframes in project_config.yaml to ensure all outlier frames are included in training data"
            )
            return

        print("Updating nframes to include newly tracked outlier data...")
        with open(self.working_dir / "project_config.yaml", "r") as fp:
            config = yaml.safe_load(fp)
        config["nframes"] = self.nframes
        with open(self.working_dir / "project_config.yaml", "w") as fp:
            config = yaml.safe_dump(config, fp, sort_keys=False)

        print(
            f"nframes updated to include new outlier data. New nframes for training data: {self.nframes}"
        )
