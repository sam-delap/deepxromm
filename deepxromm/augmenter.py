from enum import Enum
from pathlib import Path
import re
import yaml

import deeplabcut

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
                    match = re.search(r"img\(d+)\.png", img)
                    if match is None:
                        raise ValueError(
                            "Couldn't parse frame number from image files. Something wrong with DeepLabCut?"
                        )
                    outliers.append(
                        int(match.group(1)) + 1
                    )  # Match DLC index with XMAlab frame number
                with open(analysis_path / f"outliers.yaml", "w") as fp:
                    yaml.safe_dump(outliers, fp)

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
            with open(analysis_path / f"outliers.yaml", "w") as fp:
                yaml.safe_dump(matched_outliers, fp)
