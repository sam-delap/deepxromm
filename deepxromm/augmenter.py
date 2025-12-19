from pathlib import Path
from enum import Enum
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
        self.outlier_algorithm = OutlierAlgorithm(
            config["augmenter"]["outlier_algorithm"]
        )
        self.extraction_algorithm = ExtractionAlgorithm(
            config["augmenter"]["extraction_algorithm"]
        )
        self.dlc_config_path = Path(config["path_config_file"])
        self.dlc_config_path_2 = (
            Path(config["path_config_file_2"]) if self.mode == "per_cam" else None
        )
        self._data_processor = XMADataProcessor(config)

        with dlc_config_path.open("r") as fp:
            dlc_config = yaml.safe_load(fp)
        self._iteration = int(dlc_config["iteration"])

    def extract_outlier_frames(self, **kwargs) -> None:
        """Extract outlier frames from DLC output and store them in 'outliers.yaml' file for the user"""
        if self.mode not in ["2D", "per_cam", "rgb"]:
            raise ValueError(f"Unsupported mode: {mode}")

        picked_frames = []
        for trial_path in self._data_processor.list_trials():
            # DLC's looking for the path we saved its output to
            analysis_path = trial_path / f"it{iteration}"
            if mode == "rgb":
                # We can assume that the merged RGB file exists at this point, because data has been analyzed
                cam_file = self._data_processor.find_cam_file(trial_path, "rgb")
                outliers[camera] = deeplabcut.extract_outlier_frames(
                    self.dlc_config_path,
                    [str(cam_file)],
                    outlieralgorithm=self.outlier_algorithm.value,
                    extractionalgorithm=self.extraction_algorithm.value,
                    destfolder=iteration_folder,
                    **kwargs,
                )
                with open(analysis_path / f"outliers.yaml", "w") as fp:
                    yaml.safe_dump(outliers, fp, sort_keys=False)
                return

            outliers = {}
            for camera in ["cam1", "cam2"]:
                cam_file = self._data_processor.find_cam_file(trial_path, camera)
                if camera == "cam2" and self.path_config_file_2 is not None:
                    outliers[camera] = deeplabcut.extract_outlier_frames(
                        self.dlc_config_path_2,
                        [str(cam_file)],
                        outlieralgorithm=self.outlier_algorithm.value,
                        extractionalgorithm=self.extraction_algorithm.value,
                        destfolder=iteration_folder,
                        **kwargs,
                    )
                else:
                    outliers[camera] = deeplabcut.extract_outlier_frames(
                        self.dlc_config_path_2,
                        [str(cam_file)],
                        outlieralgorithm=self.outlier_algorithm.value,
                        extractionalgorithm=self.extraction_algorithm.value,
                        destfolder=iteration_folder,
                        **kwargs,
                    )
                with open(analysis_path / f"{camera}_outliers.yaml", "w") as fp:
                    yaml.safe_dump(outliers, fp, sort_keys=False)

            merged_outliers = [
                matched_outlier
                for matched_outlier in outliers["cam1"]
                if matched_outlier in outliers["cam2"]
            ]
            with open(analysis_path / f"outliers.yaml", "w") as fp:
                yaml.safe_dump(outliers, fp, sort_keys=False)
