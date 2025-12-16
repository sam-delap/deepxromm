import os
from pathlib import Path
import shutil
import unittest
import random

import pandas as pd
import numpy as np

from deepxromm import DeepXROMM
from deepxromm.xrommtools import dlc_to_xma

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


# Helper functions for creating mock DLC analysis output
def add_likelihood_columns_to_training_data(training_df):
    """
    Transform DLC training dataset into analysis output format by adding likelihood columns.

    Likelihood distribution simulates DLC behavior:
    - 70% high confidence (0.95-0.99)
    - 20% moderate (0.85-0.95)
    - 10% low (0.60-0.85)

    Args:
        training_df: DLC training DataFrame from CollectedData_*.h5 (no likelihood columns)

    Returns:
        DataFrame with likelihood columns inserted after each (x,y) pair
    """
    n_frames = len(training_df)

    # Create new multi-index structure with likelihood columns
    new_columns = []
    for scorer in training_df.columns.get_level_values(0).unique():
        for bodypart in training_df.columns.get_level_values(1).unique():
            new_columns.extend(
                [
                    (scorer, bodypart, "x"),
                    (scorer, bodypart, "y"),
                    (scorer, bodypart, "likelihood"),
                ]
            )

    new_index = pd.MultiIndex.from_tuples(
        new_columns, names=["scorer", "bodyparts", "coords"]
    )
    result_df = pd.DataFrame(index=training_df.index, columns=new_index)

    # Copy x,y values and generate realistic likelihoods
    for scorer in training_df.columns.get_level_values(0).unique():
        for bodypart in training_df.columns.get_level_values(1).unique():
            result_df[(scorer, bodypart, "x")] = training_df[(scorer, bodypart, "x")]
            result_df[(scorer, bodypart, "y")] = training_df[(scorer, bodypart, "y")]

            # Generate realistic likelihoods with variance
            likelihoods = []
            for _ in range(n_frames):
                rand = np.random.random()
                if rand < 0.7:  # 70% high confidence
                    likelihoods.append(np.random.uniform(0.95, 0.99))
                elif rand < 0.9:  # 20% moderate
                    likelihoods.append(np.random.uniform(0.85, 0.95))
                else:  # 10% low
                    likelihoods.append(np.random.uniform(0.60, 0.85))

            result_df[(scorer, bodypart, "likelihood")] = likelihoods

    return result_df


def assert_xma_roundtrip_matches(reconstructed_csv, original_csv, tolerance=0.01):
    """
    Verify dlc_to_xma output matches original XMAlab CSV.

    This validates the round-trip: XMA → DLC training → mock analysis → dlc_to_xma → XMA

    Args:
        reconstructed_csv: Path to dlc_to_xma output CSV
        original_csv: Path to original XMAlab CSV (ground truth)
        tolerance: Relative tolerance for floating point comparison
    """
    original = pd.read_csv(original_csv)
    reconstructed = pd.read_csv(reconstructed_csv)

    # Compare structure
    assert (
        original.shape == reconstructed.shape
    ), f"Shape mismatch: original {original.shape} vs reconstructed {reconstructed.shape}"
    assert list(original.columns) == list(
        reconstructed.columns
    ), "Column names don't match between original and reconstructed"

    # Compare data values with tolerance for float operations
    pd.testing.assert_frame_equal(
        original, reconstructed, rtol=tolerance, check_exact=False, check_names=True
    )


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

    def test_create_training_dataset_succeeds(self):
        """Test that this project will create a dataset correctly"""
        self.deepxromm.create_training_dataset()

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestDlcToXma2D(unittest.TestCase):
    """Test dlc_to_xma function in 2D mode with round-trip verification"""

    def setUp(self):
        """Create 2D project and generate mock DLC analysis output"""
        self.working_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(
            self.working_dir, mode="2D", codec=DEEPXROMM_TEST_CODEC
        )

        # Copy trial slice data
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)
        self.trial_csv = trial_dir / "test.csv"
        self.cam1_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        shutil.copy("trial_slice.csv", str(self.trial_csv))
        shutil.copy("trial_cam1_slice.avi", str(self.cam1_path))
        shutil.copy("trial_cam2_slice.avi", str(self.cam2_path))

        # Run xma_to_dlc to create training dataset
        self.deepxromm = DeepXROMM.load_project(self.working_dir)
        self.deepxromm.xma_to_dlc()

        # Generate mock DLC analysis output
        self.create_mock_dlc_analysis_2d()

    def create_mock_dlc_analysis_2d(self):
        """Create realistic DLC analysis output for 2D mode"""
        # Read DLC training dataset created by xma_to_dlc
        dlc_config = Path(self.deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        training_df = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Add likelihood columns with realistic variance
        analysis_df = add_likelihood_columns_to_training_data(training_df)

        # Save as mock analysis output
        output_dir = self.working_dir / "trials/test/it0"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Use DLC naming convention
        self.mock_h5 = (
            output_dir / "testDLC_resnet50_test_projectDec1shuffle1_100000.h5"
        )
        self.mock_csv = (
            output_dir / "testDLC_resnet50_test_projectDec1shuffle1_100000.csv"
        )

        analysis_df.to_hdf(self.mock_h5, key="df_with_missing", mode="w")
        analysis_df.to_csv(self.mock_csv, na_rep="NaN")

    def test_dlc_to_xma_creates_xmalab_format_files(self):
        """
        Given I have DLC analysis output with likelihood columns
        When I call dlc_to_xma
        Then it creates XMAlab format CSV and HDF5 files
        """
        # Load mock DLC data and split by camera
        full_data = pd.read_hdf(self.mock_h5)
        # Filter rows by camera in index
        cam1_data = full_data[full_data.index.str.contains("cam1")]
        cam2_data = full_data[full_data.index.str.contains("cam2")]

        # Run dlc_to_xma
        output_dir = self.working_dir / "trials/test/it0"
        dlc_to_xma(
            cam1data=cam1_data,
            cam2data=cam2_data,
            trialname="test",
            savepath=output_dir,
        )

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
        full_data = pd.read_hdf(self.mock_h5)
        cam1_data = full_data[full_data.index.str.contains("cam1")]
        cam2_data = full_data[full_data.index.str.contains("cam2")]
        output_dir = self.working_dir / "trials/test/it0"
        dlc_to_xma(cam1_data, cam2_data, "test", output_dir)

        # Check output
        xma_csv = output_dir / "test-Predicted2DPoints.csv"
        df = pd.read_csv(xma_csv)

        # Verify no likelihood columns
        for col in df.columns:
            self.assertNotIn("likelihood", col.lower())
            self.assertTrue(col.endswith(("_X", "_Y")))

    def test_dlc_to_xma_roundtrip_preserves_data(self):
        """
        Given original trial_slice.csv (ground truth)
        When I run: XMA → DLC training → mock DLC analysis → dlc_to_xma
        Then the reconstructed XMA data matches the original

        This is the KEY test - validates entire pipeline integrity
        """
        # Run dlc_to_xma (split by camera)
        full_data = pd.read_hdf(self.mock_h5)
        cam1_data = full_data[full_data.index.str.contains("cam1")]
        cam2_data = full_data[full_data.index.str.contains("cam2")]
        output_dir = self.working_dir / "trials/test/it0"
        dlc_to_xma(cam1_data, cam2_data, "test", output_dir)

        # Compare reconstructed with original
        reconstructed_csv = output_dir / "test-Predicted2DPoints.csv"
        assert_xma_roundtrip_matches(reconstructed_csv, self.trial_csv, tolerance=0.01)

    def test_dlc_to_xma_handles_csv_file_input(self):
        """
        Given DLC analysis output as CSV file paths
        When I call dlc_to_xma with file path strings (not DataFrames)
        Then it loads and processes the data correctly
        """
        output_dir = self.working_dir / "trials/test/it0"
        dlc_to_xma(
            cam1data=str(self.mock_csv),
            cam2data=str(self.mock_csv),
            trialname="test",
            savepath=output_dir,
        )

        # Verify outputs exist and are valid
        xma_csv = output_dir / "test-Predicted2DPoints.csv"
        self.assertTrue(xma_csv.exists())

        # Verify round-trip still matches
        assert_xma_roundtrip_matches(xma_csv, self.trial_csv, tolerance=0.01)

    def test_dlc_to_xma_handles_hdf5_file_input(self):
        """
        Given DLC analysis output as HDF5 file paths
        When I call dlc_to_xma with file path strings
        Then it loads and processes the data correctly
        """
        output_dir = self.working_dir / "trials/test/it0"
        dlc_to_xma(
            cam1data=str(self.mock_h5),
            cam2data=str(self.mock_h5),
            trialname="test",
            savepath=output_dir,
        )

        # Verify round-trip matches
        xma_csv = output_dir / "test-Predicted2DPoints.csv"
        assert_xma_roundtrip_matches(xma_csv, self.trial_csv, tolerance=0.01)

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        if project_path.exists():
            shutil.rmtree(project_path)
