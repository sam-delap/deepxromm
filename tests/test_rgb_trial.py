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


class TestRGBTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - RGB trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(
            self.working_dir, mode="rgb", codec=DEEPXROMM_TEST_CODEC
        )

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Make vars for pathing to find files easily
        self.trial_csv = trial_dir / "test.csv"
        self.rgb_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        # Move sample frame input to trainingdata
        shutil.copy("trial_slice.csv", str(self.trial_csv))
        shutil.copy("trial_cam1_slice.avi", str(self.rgb_path))
        shutil.copy("trial_cam2_slice.avi", str(self.cam2_path))
        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""

        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        # Load DLC data
        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Load DLC first row
        rgb_img_path = str(
            (labeled_data_path / "test_rgb_0001.png").relative_to(dlc_config.parent)
        )
        rgb_first_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_first_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_first_row[xmalab_key]
            dlc_data_point = rgb_first_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_last_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

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

        # Load DLC rgb last row
        rgb_img_path = str(
            (labeled_data_path / f"test_rgb_{last_frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        rgb_last_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_last_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_last_row[xmalab_key]
            dlc_data_point = rgb_last_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_random_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

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
        rgb_img_path = str(
            (labeled_data_path / f"test_rgb_{frame_number}.png").relative_to(
                dlc_config.parent
            )
        )
        rgb_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_row[xmalab_key]
            dlc_data_point = rgb_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_create_training_dataset_succeeds(self):
        """Test that this project will create a dataset correctly"""
        self.deepxromm.create_training_dataset()

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestDlcToXmaRGB(unittest.TestCase):
    """Test dlc_to_xma function in rgb mode with round-trip verification"""

    def setUp(self):
        """Create 2D project and generate mock DLC analysis output"""
        self.working_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(
            self.working_dir, mode="rgb", codec=DEEPXROMM_TEST_CODEC
        )

        # Copy trial slice data
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)
        self.trial_csv = trial_dir / "test.csv"
        self.rgb_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        shutil.copy("trial_slice.csv", str(self.trial_csv))
        shutil.copy("trial_cam1_slice.avi", str(self.rgb_path))
        shutil.copy("trial_cam2_slice.avi", str(self.cam2_path))

        # Run xma_to_dlc to create training dataset
        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

        # Copy in mock DLC data
        rgb_df = pd.read_hdf("trial_rgbdlc.h5")
        output_dir = self.working_dir / "trials/test/it0"
        output_dir.mkdir(parents=True, exist_ok=True)
        self.mock_rgb_h5 = (
            output_dir / "test_rgbDLC_resnet50_test_projectDec1shuffle1_100000.h5"
        )
        self.mock_rgb_csv = (
            output_dir / "test_rgbDLC_resnet50_test_projectDec1shuffle1_100000.csv"
        )
        rgb_df.to_hdf(self.mock_rgb_h5, key="df_with_missing", mode="w")
        rgb_df.to_csv(self.mock_rgb_csv, na_rep="NaN")

        # Run DLC to XMA
        self.deepxromm.dlc_to_xma()

    def test_dlc_to_xma_creates_xmalab_format_files(self):
        """
        Given I have DLC analysis output with likelihood columns
        When I call dlc_to_xma
        Then it creates both CSV and HDF5 files
        """
        # Run dlc_to_xma
        output_dir = self.working_dir / "trials/test/it0"

        # Verify outputs exist
        xma_csv = output_dir / "test-Predicted2DPoints.csv"
        xma_h5 = output_dir / "test-Predicted2DPoints.h5"

        self.assertTrue(xma_csv.exists(), "XMAlab CSV not created")
        self.assertTrue(xma_h5.exists(), "XMAlab HDF5 not created")

    def test_dlc_to_xma_removes_likelihood_columns(self):
        """
        Given DLC analysis output has likelihood columns
        When I call dlc_to_xma
        Then the output XMAlab CSV has no likelihood columns
        And only contains X,Y coordinates
        """
        # Load and convert (split by camera)
        output_dir = self.working_dir / "trials/test/it0"

        # Check output
        xma_csv = output_dir / "test-Predicted2DPoints.csv"
        df = pd.read_csv(xma_csv)

        # Verify no likelihood columns
        for col in df.columns:
            self.assertNotIn("likelihood", col.lower())
            self.assertTrue(col.endswith(("_X", "_Y")))

    def test_first_frame_matches(self):
        """
        Given DLC-formatted output from a training run of this trial
        When I run dlc_to_xma
        Then the first frame of the reconstructed XMA data matches the DLC-formatted data
        """
        output_dir = self.working_dir / "trials/test/it0"
        rgb_dlc_data = pd.read_hdf(self.mock_rgb_h5)

        xmalab_data = pd.read_csv(output_dir / "test-Predicted2DPoints.csv")
        xmalab_first_row = xmalab_data.loc[0, :]

        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        rgb_first_row = rgb_dlc_data.loc[0, :]
        for val in rgb_first_row.index:
            if "likelihood" in val or "marker001" not in val or "x" not in val:
                continue
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_first_row[xmalab_key]
            dlc_data_point = rgb_first_row[val]
            with self.subTest(folder=val):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_last_frame_matches(self):
        """
        Given DLC-formatted output from a training run of this trial
        When I run dlc_to_xma
        Then the last frame of the reconstructed XMA data matches the DLC-formatted data
        """
        output_dir = self.working_dir / "trials/test/it0"
        rgb_dlc_data = pd.read_hdf(
            output_dir / "test_rgbDLC_resnet50_test_projectDec1shuffle1_100000.h5"
        )

        xmalab_data = pd.read_csv(output_dir / "test-Predicted2DPoints.csv")
        # Find XMAlab last row
        xmalab_last_row_int = xmalab_data.index[-1]
        xmalab_last_row = xmalab_data.loc[xmalab_last_row_int, :]

        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        rgb_last_row = rgb_dlc_data.loc[xmalab_last_row_int, :]
        for val in rgb_last_row.index:
            if "likelihood" in val:
                continue
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = round(xmalab_last_row[xmalab_key], 4)
            dlc_data_point = round(rgb_last_row[val], 4)
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_random_frame_matches(self):
        """
        Given DLC-formatted output from a training run of this trial
        When I run dlc_to_xma
        Then a random frame of the reconstructed XMA data matches the DLC-formatted data
        """
        output_dir = self.working_dir / "trials/test/it0"
        rgb_dlc_data = pd.read_hdf(self.mock_rgb_h5)

        xmalab_data = pd.read_csv(output_dir / "test-Predicted2DPoints.csv")
        # Get random row from xmalab data
        xmalab_rand_row_int = random.choice(xmalab_data.index)
        xmalab_rand_row = xmalab_data.loc[xmalab_rand_row_int, :]

        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        rgb_rand_row = rgb_dlc_data.loc[xmalab_rand_row_int, :]
        for val in rgb_rand_row.index:
            if "likelihood" in val:
                continue
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = round(xmalab_rand_row[xmalab_key], 4)
            dlc_data_point = round(rgb_rand_row[val], 4)
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        if project_path.exists():
            shutil.rmtree(project_path)
