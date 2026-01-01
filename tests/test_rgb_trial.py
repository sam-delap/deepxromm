import os
from pathlib import Path
import shutil
import unittest
import random

import pandas as pd
import cv2

from deepxromm import DeepXROMM
from deepxromm.config_utilities import load_config_file, save_config_file
from .utils import set_up_project, copy_mock_dlc_data_rgb

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class TestRGBMarkerCombos(unittest.TestCase):
    """Verify that the different RGB marker combinations work"""

    def setUp(self):
        """Configure a project that has just been created"""
        self.working_dir = Path.cwd() / "tmp"
        self.deepxromm_proj = DeepXROMM.create_new_project(self.working_dir, mode="rgb")
        self.deepxromm_proj.project.dlc_config.video_codec = DEEPXROMM_TEST_CODEC
        self.deepxromm_proj.project.update_config_file()

        frame = cv2.imread(str(SAMPLE_FRAME))

        # Make a trial directory
        (self.working_dir / "trainingdata/dummy").mkdir(parents=True, exist_ok=True)

        # Cam 1
        video_path_1 = self.working_dir / "trainingdata/dummy/dummy_cam1.avi"
        out = cv2.VideoWriter(
            str(video_path_1), cv2.VideoWriter_fourcc(*"DIVX"), 15, (1024, 512)
        )
        out.write(frame)
        out.release()

        # Cam 2
        video_path_2 = self.working_dir / "trainingdata/dummy/dummy_cam2.avi"
        out = cv2.VideoWriter(
            str(video_path_2), cv2.VideoWriter_fourcc(*"DIVX"), 15, (1024, 512)
        )
        out.write(frame)
        out.release()

        # CSV
        df = pd.DataFrame(
            {
                "foo_cam1_X": 0,
                "foo_cam1_Y": 0,
                "foo_cam2_X": 0,
                "foo_cam2_Y": 0,
                "bar_cam1_X": 0,
                "bar_cam1_Y": 0,
                "bar_cam2_X": 0,
                "bar_cam2_Y": 0,
                "baz_cam1_X": 0,
                "baz_cam1_Y": 0,
                "baz_cam2_X": 0,
                "baz_cam2_Y": 0,
            },
            index=[1],
        )
        csv_path = self.working_dir / "trainingdata/dummy/dummy.csv"
        df.to_csv(str(csv_path), index=False)
        cv2.destroyAllWindows()

    def test_bodyparts_add_swapped(self):
        """Can we add swapped markers?"""
        self.deepxromm_proj.project.dlc_config.swapped_markers = True
        self.deepxromm_proj.project.update_config_file()
        DeepXROMM.load_project(self.working_dir)

        config_obj = load_config_file(
            self.deepxromm_proj.project.dlc_config.path_config_file
        )

        self.assertEqual(
            config_obj["bodyparts"],
            [
                "foo_cam1",
                "foo_cam2",
                "bar_cam1",
                "bar_cam2",
                "baz_cam1",
                "baz_cam2",
                "sw_foo_cam1",
                "sw_foo_cam2",
                "sw_bar_cam1",
                "sw_bar_cam2",
                "sw_baz_cam1",
                "sw_baz_cam2",
            ],
        )

    def test_bodyparts_add_from_csv_in_3d(self):
        """If the user wants to do 3D tracking, we output the desired list of bodyparts"""
        DeepXROMM.load_project(self.working_dir)
        config_obj = load_config_file(
            self.deepxromm_proj.project.dlc_config.path_config_file
        )
        self.assertEqual(
            config_obj["bodyparts"],
            ["foo_cam1", "foo_cam2", "bar_cam1", "bar_cam2", "baz_cam1", "baz_cam2"],
        )

    def test_bodyparts_add_synthetic_and_crossed(self):
        """Can we add both swapped and crossed markers?"""
        tmp = load_config_file(self.working_dir / "project_config.yaml")
        tmp["swapped_markers"] = True
        tmp["crossed_markers"] = True
        save_config_file(tmp, self.working_dir / "project_config.yaml")
        DeepXROMM.load_project(self.working_dir)

        config_obj = load_config_file(
            self.deepxromm_proj.project.dlc_config.path_config_file
        )

        self.assertEqual(
            config_obj["bodyparts"],
            [
                "foo_cam1",
                "foo_cam2",
                "bar_cam1",
                "bar_cam2",
                "baz_cam1",
                "baz_cam2",
                "sw_foo_cam1",
                "sw_foo_cam2",
                "sw_bar_cam1",
                "sw_bar_cam2",
                "sw_baz_cam1",
                "sw_baz_cam2",
                "cx_foo_cam1x2",
                "cx_bar_cam1x2",
                "cx_baz_cam1x2",
            ],
        )

    def test_bodyparts_add_crossed(self):
        """Can we add crossed markers?"""
        tmp = load_config_file(self.working_dir / "project_config.yaml")
        tmp["crossed_markers"] = True
        save_config_file(tmp, self.working_dir / "project_config.yaml")
        DeepXROMM.load_project(self.working_dir)

        config_obj = load_config_file(
            self.deepxromm_proj.project.dlc_config.path_config_file
        )

        self.assertEqual(
            config_obj["bodyparts"],
            [
                "foo_cam1",
                "foo_cam2",
                "bar_cam1",
                "bar_cam2",
                "baz_cam1",
                "baz_cam2",
                "cx_foo_cam1x2",
                "cx_bar_cam1x2",
                "cx_baz_cam1x2",
            ],
        )

    def tearDown(self):
        """Clean up once tests are done running"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)


class TestRGBTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - RGB trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        self.trial_csv, self.deepxromm_proj = set_up_project(self.working_dir, "rgb")
        self.deepxromm_proj.xma_to_dlc()

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""

        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        # Load DLC data
        dlc_config = Path(self.deepxromm_proj.project.dlc_config.path_config_file)
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
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)
        self.deepxromm_proj.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(self.deepxromm_proj.project.dlc_config.path_config_file)
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
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)
        self.deepxromm_proj.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(self.deepxromm_proj.project.dlc_config.path_config_file)
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
        self.deepxromm_proj.create_training_dataset()

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestDlcToXmaRGB(unittest.TestCase):
    """Test dlc_to_xma function in rgb mode with round-trip verification"""

    def setUp(self):
        """Create RGB project and generate mock DLC analysis output"""
        self.working_dir = Path.cwd() / "tmp"
        self.trial_csv, self.deepxromm_proj = set_up_project(self.working_dir, "rgb")
        self.deepxromm_proj.xma_to_dlc()
        self.mock_rgb_h5 = copy_mock_dlc_data_rgb(self.working_dir)
        self.deepxromm_proj.dlc_to_xma()

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


class TestAutocorrectRGB(unittest.TestCase):
    """Test that autocorrect runs properly for RGB project"""

    def setUp(self):
        """Create RGB project and generate mock DLC analysis output"""
        self.working_dir = Path.cwd() / "tmp"
        self.trial_csv, self.deepxromm_proj = set_up_project(self.working_dir, "rgb")
        self.mock_rgb_h5 = copy_mock_dlc_data_rgb(self.working_dir)

    def run_autocorrect(self):
        """Run autocorrect using the provided deepxromm project"""
        self.deepxromm_proj.autocorrect_trials()

    def tearDown(self):
        """Remove the created temp project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
