import os
from pathlib import Path
import shutil
import unittest
import random

import pandas as pd

from deepxromm import DeepXROMM

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class Test2DTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - 2D, combined trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(self.working_dir, codec=DEEPXROMM_TEST_CODEC)

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Make vars for pathing to find files easily
        self.trial_csv = trial_dir / "test.csv"
        self.cam1_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        # Copy sample CSV data (use existing sample file)
        shutil.copy("trial_slice.csv", str(self.trial_csv))
        shutil.copy("trial_cam1_slice.avi", str(self.cam1_path))
        shutil.copy("trial_cam2_slice.avi", str(self.cam2_path))

        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")
        cam1_img_path = str(
            (labeled_data_path / "test_cam1_0001.png").relative_to(dlc_config.parent)
        )
        cam2_img_path = str(
            (labeled_data_path / "test_cam2_0001.png").relative_to(dlc_config.parent)
        )
        cam1_first_row = dlc_data.loc[cam1_img_path, :]
        cam2_first_row = dlc_data.loc[cam2_img_path, :]
        for val in cam1_first_row.index:
            xmalab_key = f"{val[1]}_cam1_{val[2].upper()}"
            xmalab_data_point = xmalab_first_row[xmalab_key]
            dlc_data_point = cam1_first_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

        for val in cam2_first_row.index:
            xmalab_key = f"{val[1]}_cam2_{val[2].upper()}"
            xmalab_data_point = xmalab_first_row[xmalab_key]
            dlc_data_point = cam2_first_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_last_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        last_file = Path(dlc_data.index[-1])
        last_frame_number = last_file.stem.split("_")[-1]
        last_frame_int = int(last_frame_number)

        # Load XMAlab last row
        xmalab_last_row = xmalab_data.loc[last_frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(
            (labeled_data_path / f"test_cam1_{last_frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        cam1_last_row = dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(
            (labeled_data_path / f"test_cam2_{last_frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        cam2_last_row = dlc_data.loc[cam2_img_path, :]

        for val in cam1_last_row.index:
            xmalab_key = f"{val[1]}_cam1_{val[2].upper()}"
            xmalab_data_point = xmalab_last_row[xmalab_key]
            dlc_data_point = cam1_last_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

        for val in cam2_last_row.index:
            xmalab_key = f"{val[1]}_cam2_{val[2].upper()}"
            xmalab_data_point = xmalab_last_row[xmalab_key]
            dlc_data_point = cam2_last_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_random_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        file = Path(random.choice(dlc_data.index))
        frame_number = file.stem.split("_")[-1]
        frame_int = int(frame_number)

        # Load XMAlab last row
        xmalab_row = xmalab_data.loc[frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(
            (labeled_data_path / f"test_cam1_{frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        cam1_row = dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(
            (labeled_data_path / f"test_cam2_{frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        cam2_row = dlc_data.loc[cam2_img_path, :]

        for val in cam1_row.index:
            xmalab_key = f"{val[1]}_cam1_{val[2].upper()}"
            xmalab_data_point = xmalab_row[xmalab_key]
            dlc_data_point = cam1_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

        for val in cam2_row.index:
            xmalab_key = f"{val[1]}_cam2_{val[2].upper()}"
            xmalab_data_point = xmalab_row[xmalab_key]
            dlc_data_point = cam2_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def create_training_dataset_succeeds(self):
        """Test that this project will create a dataset correctly"""
        self.deepxromm.create_training_dataset()

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)
