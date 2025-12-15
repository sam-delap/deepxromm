"""Unit tests for deepxromm"""

import os
from pathlib import Path
import shutil
import unittest
from datetime import datetime as dt
import random

import cv2
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from ruamel.yaml import YAML

from deepxromm import DeepXROMM
from deepxromm.xma_data_processor import XMADataProcessor


SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class TestProjectCreation(unittest.TestCase):
    """Tests behaviors related to XMA-DLC project creation"""

    @classmethod
    def setUpClass(cls):
        """Create a sample project"""
        super(TestProjectCreation, cls).setUpClass()
        cls.project_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(cls.project_dir, codec=DEEPXROMM_TEST_CODEC)

    def test_project_creates_correct_folders(self):
        """Do we have all of the correct folders?"""
        for folder in ["trainingdata", "trials"]:
            with self.subTest(folder=folder):
                self.assertTrue((self.project_dir / folder).exists())

    def test_project_creates_config_file(self):
        """Do we have a project config?"""
        self.assertTrue((self.project_dir / "project_config.yaml").exists())

    def test_project_config_has_these_variables(self):
        """Can we access each of the variables that's supposed to be in the config?"""
        yaml = YAML()
        variables = [
            "task",
            "experimenter",
            "working_dir",
            "path_config_file",
            "dataset_name",
            "nframes",
            "maxiters",
            "tracking_threshold",
            "tracking_mode",
            "swapped_markers",
            "crossed_markers",
            "search_area",
            "threshold",
            "krad",
            "gsigma",
            "img_wt",
            "blur_wt",
            "gamma",
            "cam",
            "frame_num",
            "trial_name",
            "marker",
            "test_autocorrect",
            "cam1s_are_the_same_view",
        ]

        yaml = YAML()
        config_path = self.project_dir / "project_config.yaml"
        with config_path.open() as config:
            project = yaml.load(config)
            for variable in variables:
                with self.subTest(i=variable):
                    self.assertIsNotNone(project[variable])

    @classmethod
    def tearDownClass(cls):
        """Remove the created temp project"""
        super(TestProjectCreation, cls).tearDownClass()
        shutil.rmtree(cls.project_dir)


class TestDefaultsPerformance(unittest.TestCase):
    """Test that the config will still be configured properly if the user only provides XMAlab input"""

    def setUp(self):
        """Create a sample project where the user only inputs XMAlab data"""
        self.working_dir = Path.cwd() / "tmp"
        self.config = self.working_dir / "project_config.yaml"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, codec=DEEPXROMM_TEST_CODEC
        )
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

    def test_can_find_frames_from_csv(self):
        """Can I accurately find the number of frames in the video if the user doesn't tell me?"""
        print(list(self.working_dir.iterdir()))
        deepxromm = DeepXROMM.load_project(self.working_dir)
        config = deepxromm.config
        self.assertEqual(
            config["nframes"], 1, msg=f"Actual nframes: {config['nframes']}"
        )

    def test_analyze_errors_if_no_folders_in_trials_dir(self):
        """If there are no trials to analyze, do we return an error?"""
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.analyze_videos()

    def test_autocorrect_errors_if_no_folders_in_trials_dir(self):
        """If there are no trials to autocorrect, do we return an error?"""
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.autocorrect_trials()

    def test_warn_users_if_nframes_doesnt_match_csv(self):
        """If the number of frames in the CSV doesn't match the number of frames specified, do I issue a warning?"""
        yaml = YAML()
        with self.config.open() as config:
            tmp = yaml.load(config)

        # Modify the number of frames (similar to how a user would)
        tmp["nframes"] = 2
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            DeepXROMM.load_project(self.working_dir)

    def test_yaml_file_updates_nframes_after_load_if_frames_is_0(self):
        """If the user doesn't specify how many frames they want analyzed,
        does their YAML file get updated with how many are in the CSV?"""
        yaml = YAML()
        DeepXROMM.load_project(self.working_dir)
        with self.config.open() as config:
            tmp = yaml.load(config)
        self.assertEqual(tmp["nframes"], 1, msg=f"Actual nframes: {tmp['nframes']}")

    def test_warn_if_user_has_tracked_less_than_threshold_frames(self):
        """If the user has tracked less than threshold % of their trial,
        do I give them a warning?"""
        DeepXROMM.load_project(self.working_dir)

        # Increase the number of frames in the video to 100 so I can test this
        frame = cv2.imread(str(SAMPLE_FRAME))
        video_path = self.working_dir / "trainingdata/dummy/dummy_cam1.avi"
        out = cv2.VideoWriter(
            str(video_path), cv2.VideoWriter_fourcc(*"DIVX"), 15, (1024, 512)
        )
        for _ in range(100):
            out.write(frame)
        out.release()

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            DeepXROMM.load_project(self.working_dir)

    def test_bodyparts_add_from_csv_if_not_defined(self):
        """If the user hasn't specified the bodyparts from their trial, we can pull them from the CSV"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        DeepXROMM.load_project(self.working_dir)
        config_path = Path(self.working_dir) / f"tmp-NA-{date}" / "config.yaml"

        with config_path.open() as dlc_config:
            config_obj = yaml.load(dlc_config)
        self.assertEqual(config_obj["bodyparts"], ["foo", "bar", "baz"])

    def test_bodyparts_add_from_csv_in_3d(self):
        """If the user wants to do 3D tracking, we output the desired list of bodyparts"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["tracking_mode"] = "rgb"
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f"tmp-NA-{date}" / "config.yaml"
        yaml = YAML()
        with path_to_config.open("r") as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(
            config_obj["bodyparts"],
            ["foo_cam1", "foo_cam2", "bar_cam1", "bar_cam2", "baz_cam1", "baz_cam2"],
        )

    def test_bodyparts_add_synthetic(self):
        """Can we add swapped markers?"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["tracking_mode"] = "rgb"
        tmp["swapped_markers"] = True
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f"tmp-NA-{date}" / "config.yaml"
        yaml = YAML()
        with path_to_config.open("r") as dlc_config:
            config_obj = yaml.load(dlc_config)

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

    def test_bodyparts_add_crossed(self):
        """Can we add crossed markers?"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["tracking_mode"] = "rgb"
        tmp["crossed_markers"] = True
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f"tmp-NA-{date}" / "config.yaml"
        yaml = YAML()
        with path_to_config.open("r") as dlc_config:
            config_obj = yaml.load(dlc_config)

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

    def test_bodyparts_add_synthetic_and_crossed(self):
        """Can we add both swapped and crossed markers?"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["tracking_mode"] = "rgb"
        tmp["swapped_markers"] = True
        tmp["crossed_markers"] = True
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f"tmp-NA-{date}" / "config.yaml"
        yaml = YAML()
        with path_to_config.open("r") as dlc_config:
            config_obj = yaml.load(dlc_config)

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

    def test_bodyparts_error_if_different_from_csv(self):
        """If the user specifies different bodyparts than their CSV, raise an error"""
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        path_to_config = self.working_dir / f"tmp-NA-{date}" / "config.yaml"
        yaml = YAML()
        with path_to_config.open("r") as dlc_config:
            config_dlc = yaml.load(dlc_config)

        config_dlc["bodyparts"] = ["foo", "bar"]

        with path_to_config.open("w") as dlc_config:
            yaml.dump(config_dlc, dlc_config)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def test_autocorrect_error_if_trial_not_set(self):
        """If the user doesn't specify a trial to test autocorrect with, do we error?"""
        yaml = YAML()
        with self.config.open("r") as config:
            tmp = yaml.load(config)

        tmp["test_autocorrect"] = True
        tmp["marker"] = "foo"
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def test_autocorrect_error_if_marker_not_set(self):
        """If the user doesn't specify a marker to test autocorrect with, do we error?"""
        yaml = YAML()

        with self.config.open("r") as config:
            tmp = yaml.load(config)

        tmp["test_autocorrect"] = True
        tmp["trial_name"] = "test"

        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def tearDown(self):
        """Remove the created temp project"""
        shutil.rmtree(self.working_dir)


class TestSampleFrame(unittest.TestCase):
    """Test function behaviors using a frame from an actual trial"""

    def setUp(self):
        """Create trial"""
        self.working_dir = Path.cwd() / "tmp"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, codec=DEEPXROMM_TEST_CODEC
        )
        frame = cv2.imread(str(SAMPLE_FRAME))

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Move sample frame input to trainingdata
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(trial_dir / "test.csv"))

        # Cam 1 and Cam 2 trainingdata setup
        for cam in ["cam1", "cam2"]:
            video_path = trial_dir / f"dummy_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"DIVX"), 30, (1024, 512)
            )
            out.write(frame)
            out.release()

        # Move sample frame input to trials (it0 and trials)
        trials_dir = self.working_dir / "trials/test/it0"
        trials_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(
            str(SAMPLE_FRAME_INPUT), str(self.working_dir / "trials/test/test.csv")
        )
        shutil.copy(
            str(SAMPLE_FRAME_INPUT), str(trials_dir / "test-Predicted2DPoints.csv")
        )

        # Cam 1 and Cam 2 trials setup
        for cam in ["cam1", "cam2"]:
            video_path = self.working_dir / f"trials/test/test_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"DIVX"), 30, (1024, 512)
            )
            out.write(frame)
            out.release()

        cv2.destroyAllWindows()

    def test_autocorrect_always_uses_pred_points_csv(self):
        """Test that autocorrect always uses predicted points file"""

        # Delete other CSVs to ensure autocorrect is only running against the it0 CSV
        # Main trial CSV
        test_trial_csv = Path(self.working_dir / "trials/test/test.csv")
        test_trial_csv.unlink()

        # Training data CSV
        training_data_csv = Path(self.working_dir / "trainingdata/test/test.csv")
        training_data_csv.unlink()

        # Run autocorrect
        self.deepxromm.autocorrect_trials()

        # Copy CSVs back in to make sure other tests pass (better way to do?)
        new_trial_csv = Path(self.working_dir / "trials/test/test.csv")
        new_training_data_csv = Path(self.working_dir / "trainingdata/test/test.csv")
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(new_trial_csv))
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(new_training_data_csv))

    def test_autocorrect_does_correction(self):
        """Make sure that autocorrect corrects the frame after making changes"""
        # Run autocorrect on the sample frame
        self.deepxromm.autocorrect_trials()

        # Load CSVs
        function_output_path = (
            self.working_dir / "trials/test/it0/test-AutoCorrected2DPoints.csv"
        )
        function_output = pd.read_csv(str(function_output_path), dtype="float64")
        sample_output = pd.read_csv(str(SAMPLE_AUTOCORRECT_OUTPUT), dtype="float64")

        # Drop cam2 markers and check for changes
        columns_to_drop = function_output.columns[
            function_output.columns.str.contains("cam2", case=False)
        ]
        function_output.drop(columns_to_drop, axis=1, inplace=True)
        function_output = function_output.round(6)
        sample_output = sample_output.round(6)

        try:
            # Use assert_frame_equal to check if the data frames are the same
            assert_frame_equal(
                function_output, sample_output, check_exact=False, rtol=1e-6, atol=1e-6
            )
        except AssertionError as e:
            print(f"Autocorrector diff: {function_output.compare(sample_output)}")
            raise e

    def test_image_hashing_identical_trials_returns_0(self):
        """Make sure the image hashing function is working properly"""
        # Create an identical second trial
        frame_path = self.working_dir.parent / "sample_frame.jpg"
        frame = cv2.imread(str(frame_path))
        trial_dir = self.working_dir / "trials/test2_same"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create videos
        for cam in ["cam1", "cam2"]:
            video_path = trial_dir / f"test2_same_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"DIVX"), 30, (1024, 512)
            )
            out.write(frame)
            out.release()

        # Do similarity comparison
        similarity = self.deepxromm.analyze_video_similarity_project()
        # Since both videos are the same, the image similarity output should be 0
        self.assertEqual(sum(similarity.values()), 0)

    def test_image_hashing_different_trials_returns_nonzero(self):
        """Image hashing different videos returns nonzero answer"""
        # Create a different second trial
        frame = np.zeros((480, 480, 3), np.uint8)
        trial_dir = self.working_dir / "trials/test2_diff"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create videos
        for cam in ["cam1", "cam2"]:
            video_path = trial_dir / f"test2_diff_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path), cv2.VideoWriter_fourcc(*"DIVX"), 30, (480, 480)
            )
            out.write(frame)
            out.release()

        # Do similarity comparison
        similarity = self.deepxromm.analyze_video_similarity_project()
        # Since the videos are different, should return nonzero answer
        self.assertNotEqual(sum(similarity.values()), 0)

    def test_marker_similarity_returns_0_if_identical(self):
        """Check that identical data has a similarity value of 0"""
        # Move sample data into test trial
        shutil.copy(
            str(SAMPLE_FRAME_INPUT), str(self.working_dir / "trials/test/test.csv")
        )

        # Move sample data into test2 trial
        (self.working_dir / "trials/test2").mkdir(parents=True, exist_ok=True)
        shutil.copy(
            str(SAMPLE_FRAME_INPUT), str(self.working_dir / "trials/test2/test2.csv")
        )

        # Do cross-correlation
        marker_similarity = self.deepxromm.analyze_marker_similarity_project()
        self.assertEqual(sum(marker_similarity.values()), 0)

    def test_marker_similarity_returns_not_0_if_different(self):
        """Check that different data has a similarity value of not 0"""
        # Move sample data into test trial
        shutil.copy(
            str(SAMPLE_FRAME_INPUT), str(self.working_dir / "trials/test/test.csv")
        )

        # Move autocorrect data into test2 trial
        (self.working_dir / "trials/test2").mkdir(parents=True, exist_ok=True)
        shutil.copy(
            str(SAMPLE_AUTOCORRECT_OUTPUT),
            str(self.working_dir / "trials/test2/test2.csv"),
        )

        # Do cross-correlation
        marker_similarity = self.deepxromm.analyze_marker_similarity_project()
        self.assertNotEqual(sum(marker_similarity.values()), 0)

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class Test2DTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - 2D, combined trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, codec=DEEPXROMM_TEST_CODEC
        )

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Make vars for pathing to find files easily
        self.trial_csv = trial_dir / "test.csv"
        self.cam1_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        # Copy sample CSV data (use existing sample file)
        shutil.copy("trial.csv", str(self.trial_csv))
        shutil.copy("trial_cam1.avi", str(self.cam1_path))
        shutil.copy("trial_cam2.avi", str(self.cam2_path))

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")
        cam1_img_path = str(labeled_data_path / "test_cam1_0001.png")
        cam2_img_path = str(labeled_data_path / "test_cam2_0001.png")
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
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        last_file = Path(dlc_data.index[-1])
        last_frame_number = last_file.stem.split("_")[-1]
        last_frame_int = int(last_frame_number)

        # Load XMAlab last row
        xmalab_last_row = xmalab_data.loc[last_frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(labeled_data_path / f"test_cam1_{last_frame_number}.png")
        cam1_last_row = dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(labeled_data_path / f"test_cam2_{last_frame_number}.png")
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
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        file = Path(random.choice(dlc_data.index))
        frame_number = file.stem.split("_")[-1]
        frame_int = int(frame_number)

        # Load XMAlab last row
        xmalab_row = xmalab_data.loc[frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(labeled_data_path / f"test_cam1_{frame_number}.png")
        cam1_row = dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(labeled_data_path / f"test_cam2_{frame_number}.png")
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

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestPerCamTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - 2D, separate trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, mode="per_cam", codec=DEEPXROMM_TEST_CODEC
        )

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Make vars for pathing to find files easily
        self.trial_csv = trial_dir / "test.csv"
        self.cam1_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        # Copy sample CSV data (use existing sample file)
        shutil.copy("trial.csv", str(self.trial_csv))
        shutil.copy("trial_cam1.avi", str(self.cam1_path))
        shutil.copy("trial_cam2.avi", str(self.cam2_path))

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()
        cam1_dlc_proj = Path(deepxromm.config["path_config_file"]).parent
        cam2_dlc_proj = Path(deepxromm.config["path_config_file_2"]).parent

        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        cam1_labeled_data_path = cam1_dlc_proj / "labeled-data/MyData_cam1"
        cam2_labeled_data_path = cam2_dlc_proj / "labeled-data/MyData_cam2"
        cam1_dlc_data = pd.read_hdf(cam1_labeled_data_path / "CollectedData_NA.h5")
        cam2_dlc_data = pd.read_hdf(cam2_labeled_data_path / "CollectedData_NA.h5")
        cam1_img_path = str(cam1_labeled_data_path / "test_0001.png")
        cam2_img_path = str(cam2_labeled_data_path / "test_0001.png")
        cam1_first_row = cam1_dlc_data.loc[cam1_img_path, :]
        cam2_first_row = cam2_dlc_data.loc[cam2_img_path, :]
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
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        cam1_dlc_proj = Path(deepxromm.config["path_config_file"]).parent
        cam2_dlc_proj = Path(deepxromm.config["path_config_file_2"]).parent

        cam1_labeled_data_path = cam1_dlc_proj / "labeled-data/MyData_cam1"
        cam2_labeled_data_path = cam2_dlc_proj / "labeled-data/MyData_cam2"

        cam1_dlc_data = pd.read_hdf(cam1_labeled_data_path / "CollectedData_NA.h5")
        cam2_dlc_data = pd.read_hdf(cam2_labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        last_file = Path(cam1_dlc_data.index[-1])
        last_frame_number = last_file.stem.split("_")[-1]
        last_frame_int = int(last_frame_number)

        # Load XMAlab last row
        xmalab_last_row = xmalab_data.loc[last_frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(cam1_labeled_data_path / f"test_{last_frame_number}.png")
        cam1_last_row = cam1_dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(cam2_labeled_data_path / f"test_{last_frame_number}.png")
        cam2_last_row = cam2_dlc_data.loc[cam2_img_path, :]

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
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        cam1_dlc_proj = Path(deepxromm.config["path_config_file"]).parent
        cam2_dlc_proj = Path(deepxromm.config["path_config_file_2"]).parent

        cam1_labeled_data_path = cam1_dlc_proj / "labeled-data/MyData_cam1"
        cam2_labeled_data_path = cam2_dlc_proj / "labeled-data/MyData_cam2"

        cam1_dlc_data = pd.read_hdf(cam1_labeled_data_path / "CollectedData_NA.h5")
        cam2_dlc_data = pd.read_hdf(cam2_labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        file = Path(random.choice(cam1_dlc_data.index))
        frame_number = file.stem.split("_")[-1]
        frame_int = int(frame_number)

        # Load XMAlab last row
        xmalab_row = xmalab_data.loc[frame_int - 1]

        # Load DLC cam1 last row
        cam1_img_path = str(cam1_labeled_data_path / f"test_{frame_number}.png")
        cam1_row = cam1_dlc_data.loc[cam1_img_path, :]

        # Load DLC cam2 last row
        cam2_img_path = str(cam2_labeled_data_path / f"test_{frame_number}.png")
        cam2_row = cam2_dlc_data.loc[cam2_img_path, :]

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

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestRGBTrialProcess(unittest.TestCase):
    """Test function performance on an actual trial - RGB trial workflow"""

    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, mode="rgb", codec=DEEPXROMM_TEST_CODEC
        )

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Make vars for pathing to find files easily
        self.trial_csv = trial_dir / "test.csv"
        self.cam1_path = trial_dir / "test_cam1.avi"
        self.cam2_path = trial_dir / "test_cam2.avi"

        # Move sample frame input to trainingdata
        shutil.copy("trial.csv", str(self.trial_csv))
        shutil.copy("trial_cam1.avi", str(self.cam1_path))
        shutil.copy("trial_cam2.avi", str(self.cam2_path))

    def test_first_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        xmalab_data = pd.read_csv(self.trial_csv)
        xmalab_first_row = xmalab_data.loc[0, :]

        # Load DLC data
        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Load DLC first row
        rgb_img_path = str(labeled_data_path / "test_rgb_0001.png")
        rgb_first_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_first_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_first_row[xmalab_key]
            dlc_data_point = rgb_first_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_last_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        last_file = Path(dlc_data.index[-1])
        last_frame_number = last_file.stem.split("_")[-1]
        last_frame_int = int(last_frame_number)

        # Load XMAlab last row
        xmalab_last_row = xmalab_data.loc[last_frame_int - 1]

        # Load DLC rgb last row
        rgb_img_path = str(labeled_data_path / f"test_rgb_{last_frame_number}.png")
        rgb_last_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_last_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_last_row[xmalab_key]
            dlc_data_point = rgb_last_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def test_random_frame_matches_in_dlc_csv(self):
        """When I run xma_to_dlc, does the DLC CSV have the same data as my original file?"""
        deepxromm = DeepXROMM.load_project(self.working_dir)
        deepxromm.xma_to_dlc()

        # Load XMAlab data
        xmalab_data = pd.read_csv(self.trial_csv)

        # Load DLC data
        dlc_config = Path(deepxromm.config["path_config_file"])
        labeled_data_path = dlc_config.parent / "labeled-data/MyData"
        dlc_data = pd.read_hdf(labeled_data_path / "CollectedData_NA.h5")

        # Determine last frame included in training set
        file = Path(random.choice(dlc_data.index))
        frame_number = file.stem.split("_")[-1]
        frame_int = int(frame_number)

        # Load XMAlab last row
        xmalab_row = xmalab_data.loc[frame_int - 1]

        # Load DLC cam1 last row
        rgb_img_path = str(labeled_data_path / f"test_rgb_{frame_number}.png")
        rgb_row = dlc_data.loc[rgb_img_path, :]

        for val in rgb_row.index:
            xmalab_key = f"{val[1]}_{val[2].upper()}"
            xmalab_data_point = xmalab_row[xmalab_key]
            dlc_data_point = rgb_row[val]
            with self.subTest(folder=xmalab_key):
                self.assertTrue(xmalab_data_point == dlc_data_point)

    def tearDown(self):
        """Remove the created temp project"""
        project_path = Path.cwd() / "tmp"
        shutil.rmtree(project_path)


class TestXMADataProcessorHelpers:
    """Test helper methods in XMADataProcessor using pytest"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Setup test environment and cleanup after"""
        self.working_dir = tmp_path
        self.config = {
            "working_dir": str(self.working_dir),
            "swapped_markers": False,
            "crossed_markers": False,
        }
        from deepxromm.xma_data_processor import XMADataProcessor

        self.processor = XMADataProcessor(self.config)

        # Create test trial structure
        trial_dir = self.working_dir / "trainingdata" / "trial1"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create sample CSV with known data
        df = pd.DataFrame(
            {
                "marker1_cam1_X": [10.5, 20.3, np.nan],
                "marker1_cam1_Y": [15.2, 25.1, np.nan],
                "marker1_cam2_X": [12.1, 22.4, np.nan],
                "marker1_cam2_Y": [17.3, 27.2, np.nan],
                "marker2_cam1_X": [30.0, np.nan, 50.0],
                "marker2_cam1_Y": [35.0, np.nan, 55.0],
                "marker2_cam2_X": [32.0, np.nan, 52.0],
                "marker2_cam2_Y": [37.0, np.nan, 57.0],
            }
        )
        df.to_csv(trial_dir / "trial1.csv", index=False)

        yield

        # Cleanup handled automatically by tmp_path fixture

    def test_build_dlc_dataframe_creates_correct_structure(self):
        """Does build_dlc_dataframe create proper MultiIndex structure?"""
        # Arrange
        data = pd.DataFrame({0: [10.5, 12.1], 1: [15.2, 17.3]})
        scorer = "TestScorer"
        relnames = ["img1.png", "img2.png"]
        pointnames = ["marker1"]

        # Act
        result = self.processor.build_dlc_dataframe(data, scorer, relnames, pointnames)

        # Assert
        assert result.index.tolist() == relnames
        assert isinstance(result.columns, pd.MultiIndex)
        assert result.columns.names == ["scorer", "bodyparts", "coords"]
        assert (scorer, "marker1", "x") in result.columns
        assert (scorer, "marker1", "y") in result.columns

    def test_build_dlc_dataframe_handles_nan_properly(self):
        """Does build_dlc_dataframe clean NaN values correctly?"""
        # Arrange
        data = pd.DataFrame({0: [10.5, " NaN"], 1: [" NaN ", 17.3]})
        scorer = "TestScorer"
        relnames = ["img1.png", "img2.png"]
        pointnames = ["marker1"]

        # Act
        result = self.processor.build_dlc_dataframe(data, scorer, relnames, pointnames)

        # Assert
        assert pd.isna(result.iloc[0, 1])  # First row, y coord
        assert pd.isna(result.iloc[1, 0])  # Second row, x coord
        assert result.iloc[0, 0] == 10.5
        assert result.iloc[1, 1] == 17.3

    def test_read_trial_csv_validates_consistent_pointnames(self):
        """Does read_trial_csv catch inconsistent point names across trials?"""
        # Arrange: Create second trial with different markers
        trial2_dir = self.working_dir / "trainingdata" / "trial2"
        trial2_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "different_marker_cam1_X": [10.5],
                "different_marker_cam1_Y": [15.2],
                "different_marker_cam2_X": [12.1],
                "different_marker_cam2_Y": [17.3],
            }
        )
        df.to_csv(trial2_dir / "trial2.csv", index=False)

        trials = [
            self.working_dir / "trainingdata" / "trial1",
            self.working_dir / "trainingdata" / "trial2",
        ]

        # Act & Assert
        with pytest.raises(ValueError, match="(?i)point names"):
            self.processor.read_trial_csv_with_validation(trials)

    def test_read_trial_csv_filters_empty_rows(self):
        """Does read_trial_csv properly identify rows with valid data?"""
        # Arrange
        trials = [self.working_dir / "trainingdata" / "trial1"]

        # Act
        dfs, idx, pointnames = self.processor.read_trial_csv_with_validation(trials)

        # Assert: All 3 rows have >= 50% non-NaN values
        # Row 0: 6 non-NaN out of 8 (75%)
        # Row 1: 4 non-NaN out of 8 (50%)
        # Row 2: 6 non-NaN out of 8 (75%)
        assert len(idx[0]) == 3
        assert 0 in idx[0]
        assert 1 in idx[0]
        assert 2 in idx[0]

    def test_extract_2d_points_for_camera_extracts_cam1(self):
        """Does extract_2d_points_for_camera extract correct camera 1 data?"""
        # Arrange
        trial_path = self.working_dir / "trainingdata" / "trial1"
        df = pd.read_csv(trial_path / "trial1.csv", sep=",", header=None)
        df = df.loc[1:,].reset_index(drop=True)  # Remove header row
        picked_frames = [0, 2]
        camera = 1

        # Act
        result = self.processor.extract_2d_points_for_camera(df, camera, picked_frames)

        # Assert
        assert result.shape[0] == 2  # 2 frames
        assert result.shape[1] == 4  # 2 markers * 2 coords
        assert float(result.iloc[0, 0]) == 10.5  # marker1_cam1_X
        assert float(result.iloc[0, 1]) == 15.2  # marker1_cam1_Y

    def test_extract_2d_points_for_camera_extracts_cam2(self):
        """Does extract_2d_points_for_camera extract correct camera 2 data?"""
        # Arrange
        trial_path = self.working_dir / "trainingdata" / "trial1"
        df = pd.read_csv(trial_path / "trial1.csv", sep=",", header=None)
        df = df.loc[1:,].reset_index(drop=True)
        picked_frames = [0, 2]
        camera = 2

        # Act
        result = self.processor.extract_2d_points_for_camera(df, camera, picked_frames)

        # Assert
        assert result.shape[0] == 2
        assert result.shape[1] == 4
        assert float(result.iloc[0, 0]) == 12.1  # marker1_cam2_X
        assert float(result.iloc[0, 1]) == 17.3  # marker1_cam2_Y

    def test_list_trials_ignores_hidden_folders(self):
        """Does list_trials skip hidden directories?"""
        # Arrange: Create a hidden folder
        hidden_dir = self.working_dir / "trainingdata" / ".hidden"
        hidden_dir.mkdir(parents=True, exist_ok=True)

        # Act
        trials = self.processor.list_trials("trainingdata")

        # Assert
        assert len(trials) == 1
        assert trials[0].name == "trial1"

    def test_list_trials_raises_on_empty_directory(self):
        """Does list_trials raise error for empty data path?"""
        # Arrange
        empty_dir = self.working_dir / "empty_data"
        empty_dir.mkdir(parents=True, exist_ok=True)

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="(?i)no trials found"):
            self.processor.list_trials("empty_data")

    def test_list_trials_with_custom_suffix(self):
        """Does list_trials work with custom suffix like 'trainingdata'?"""
        # Arrange: Create working_dir/trainingdata/trial1, trial2
        trial2_dir = self.working_dir / "trainingdata" / "trial2"
        trial2_dir.mkdir(parents=True, exist_ok=True)

        # Act
        trials = self.processor.list_trials("trainingdata")

        # Assert
        assert len(trials) == 2
        trial_names = [t.name for t in trials]
        assert "trial1" in trial_names
        assert "trial2" in trial_names

    def test_list_trials_only_returns_directories(self):
        """Does list_trials ignore files in the trials directory?"""
        # Arrange: Create working_dir/trials/trial1/ and working_dir/trials/readme.txt
        trials_dir = self.working_dir / "trials"
        trials_dir.mkdir(parents=True, exist_ok=True)
        (trials_dir / "trial1").mkdir(exist_ok=True)
        (trials_dir / "readme.txt").write_text("test file")

        # Act
        trials = self.processor.list_trials("trials")

        # Assert
        assert len(trials) == 1
        assert trials[0].name == "trial1"
        assert trials[0].is_dir()

    def test_list_trials_uses_working_dir_from_config(self):
        """Does list_trials correctly use config['working_dir'] as base?"""
        # Arrange: Create specific working_dir structure
        specific_trials = self.working_dir / "my_trials"
        specific_trials.mkdir(parents=True, exist_ok=True)
        (specific_trials / "test_trial").mkdir(exist_ok=True)

        # Act
        trials = self.processor.list_trials("my_trials")

        # Assert
        assert len(trials) == 1
        assert trials[0].name == "test_trial"
        # Verify path starts with correct working_dir
        assert str(trials[0]).startswith(str(self.working_dir))

    def test_list_trials_with_nested_suffix(self):
        """Does list_trials handle nested paths like 'data/experiments'?"""
        # Arrange: Create working_dir/data/experiments/trial1
        nested_dir = self.working_dir / "data" / "experiments"
        nested_dir.mkdir(parents=True, exist_ok=True)
        (nested_dir / "trial1").mkdir(exist_ok=True)

        # Act
        trials = self.processor.list_trials("data/experiments")

        # Assert
        assert len(trials) == 1
        assert trials[0].name == "trial1"

    def test_list_trials_prevents_directory_traversal(self):
        """Does list_trials reject suffix with '..' or absolute paths?"""
        # Test 1: Relative traversal
        with pytest.raises(ValueError, match="(?i)security.*traversal"):
            self.processor.list_trials("../outside_project")

        # Test 2: Nested traversal
        with pytest.raises(ValueError, match="(?i)security.*traversal"):
            self.processor.list_trials("valid/../traversal")

        # Test 3: Absolute path
        with pytest.raises(ValueError, match="(?i)security.*absolute"):
            self.processor.list_trials("/absolute/path")

        # Test 4: Hidden traversal
        with pytest.raises(ValueError, match="(?i)security.*traversal"):
            self.processor.list_trials("../../etc/passwd")

    def test_list_trials_returns_path_objects(self):
        """Does list_trials return Path objects (not strings)?"""
        # Act
        trials = self.processor.list_trials("trainingdata")

        # Assert
        assert len(trials) >= 1
        for trial in trials:
            assert isinstance(trial, Path)

    def test_extract_2d_points_validates_camera_number(self):
        """Does extract_2d_points_for_camera reject invalid camera numbers?"""
        # Arrange
        trial_path = self.working_dir / "trainingdata" / "trial1"
        df = pd.read_csv(trial_path / "trial1.csv", sep=",", header=None)
        df = df.loc[1:,].reset_index(drop=True)

        # Act & Assert
        with pytest.raises(ValueError, match="(?i)camera.*must be 1 or 2"):
            self.processor.extract_2d_points_for_camera(df, 3, [0])


class TestExtractFramesFromVideo:
    """Tests for unified frame extraction method"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Setup test environment and cleanup after"""
        self.working_dir = tmp_path
        self.config = {
            "working_dir": str(self.working_dir),
            "swapped_markers": False,
            "crossed_markers": False,
        }

        self.processor = XMADataProcessor(self.config)

        # Create test output directory
        self.output_dir = self.working_dir / "output"
        self.output_dir.mkdir(exist_ok=True)

        yield

    def _create_test_video(self, video_path: Path, num_frames: int = 10):
        """Helper to create a test video with identifiable frames"""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))

        for i in range(num_frames):
            # Create frame with frame number written on it
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Make each frame slightly different (brightness based on frame number)
            # Clamp to uint8 range to avoid numpy warnings
            brightness = min(i * 20, 255)
            frame[:, :] = (brightness, brightness, brightness)
            # Add text showing frame number
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
            out.write(frame)

        out.release()
        return video_path

    def _create_test_image_folder(self, folder_path: Path, num_images: int = 10):
        """Helper to create a folder with test images"""
        folder_path.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            # Create image with distinguishable content
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            brightness = min(i * 20, 255)
            img[:, :] = (brightness, brightness, brightness)
            cv2.putText(
                img,
                f"Image {i}",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                3,
            )
            cv2.imwrite(str(folder_path / f"img_{i:04d}.png"), img)

        return folder_path

    def test_extract_frames_from_video_file_with_indices(self):
        """Can extract specific frames (0-indexed) from video file?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=100)

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 10, 50, 99],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert
        assert len(result) == 4
        assert (self.output_dir / "trial1_cam1_0001.png").exists()
        assert (self.output_dir / "trial1_cam1_0011.png").exists()
        assert (self.output_dir / "trial1_cam1_0051.png").exists()
        assert (self.output_dir / "trial1_cam1_0100.png").exists()

    def test_extract_frames_from_video_rgb_mode_naming(self):
        """Does RGB mode produce {name}_rgb_{frame}.png format?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=20)

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 5, 10],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="rgb",
        )

        # Assert
        assert len(result) == 3
        assert (self.output_dir / "trial1_rgb_0001.png").exists()
        assert (self.output_dir / "trial1_rgb_0006.png").exists()
        assert (self.output_dir / "trial1_rgb_0011.png").exists()

    def test_extract_frames_from_video_2d_mode_naming(self):
        """Does 2D mode produce {name}_cam{N}_{frame}.png format?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=20)

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 10],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=2,
        )

        # Assert
        assert len(result) == 2
        assert (self.output_dir / "trial1_cam2_0001.png").exists()
        assert (self.output_dir / "trial1_cam2_0011.png").exists()

    def test_extract_frames_from_video_per_cam_mode_naming(self):
        """Does per_cam mode produce {name}_{frame}.png format?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=20)

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 5],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="per_cam",
        )

        # Assert
        assert len(result) == 2
        assert (self.output_dir / "trial1_0001.png").exists()
        assert (self.output_dir / "trial1_0006.png").exists()

    def test_extract_frames_from_image_folder(self):
        """Can extract frames from image directory instead of video?"""
        # Arrange
        img_folder = self.working_dir / "images"
        self._create_test_image_folder(img_folder, num_images=100)

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=img_folder,
            frame_indices=[0, 10, 50],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert
        assert len(result) == 3
        # Check that images were extracted/copied
        assert (self.output_dir / "trial1_cam1_0001.png").exists()
        assert (self.output_dir / "trial1_cam1_0011.png").exists()
        assert (self.output_dir / "trial1_cam1_0051.png").exists()

    def test_extract_frames_creates_output_dir(self):
        """Does extraction create output directory if missing?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)
        nonexistent_dir = self.working_dir / "new_output" / "nested"

        # Act
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 5],
            output_dir=nonexistent_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert
        assert nonexistent_dir.exists()
        assert len(result) == 2

    def test_extract_frames_compression_levels(self):
        """Does compression parameter affect PNG file sizes?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)
        output_no_compression = self.working_dir / "output_0"
        output_max_compression = self.working_dir / "output_9"

        # Act
        self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 1, 2],
            output_dir=output_no_compression,
            output_name_base="trial1",
            mode="2D",
            camera=1,
            compression=0,
        )
        self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 1, 2],
            output_dir=output_max_compression,
            output_name_base="trial1",
            mode="2D",
            camera=1,
            compression=9,
        )

        # Assert: compression=9 should produce smaller or equal file sizes
        file_0 = output_no_compression / "trial1_cam1_0001.png"
        file_9 = output_max_compression / "trial1_cam1_0001.png"
        size_0 = file_0.stat().st_size
        size_9 = file_9.stat().st_size
        assert size_9 <= size_0

    def test_extract_frames_returns_absolute_paths(self):
        """Does method return list of absolute path strings?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)

        # Act
        paths = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 1],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert
        assert all(isinstance(p, str) for p in paths)
        assert all(Path(p).is_absolute() for p in paths)

    def test_extract_frames_zero_indexed(self):
        """Does method use 0-based indexing (frame 0 = first frame)?"""
        # Arrange: Video with unique first frame
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)

        # Act: Extract frame 0
        result = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert: Frame 0 should be the first frame (darkest)
        extracted_frame = cv2.imread(str(self.output_dir / "trial1_cam1_0001.png"))
        # First frame has brightness = 0 * 20 = 0
        assert np.mean(extracted_frame) < 5  # Very dark

    def test_extract_frames_invalid_mode_raises_error(self):
        """Does invalid mode parameter raise ValueError?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)

        # Act & Assert
        with pytest.raises(ValueError, match="(?i)invalid mode"):
            self.processor.extract_frames_from_video(
                source_path=video_path,
                frame_indices=[0],
                output_dir=self.output_dir,
                output_name_base="trial1",
                mode="invalid",
            )

    def test_extract_frames_2d_mode_requires_camera(self):
        """Does 2D mode raise error if camera parameter missing?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)

        # Act & Assert
        with pytest.raises(ValueError, match="(?i)camera.*required.*2D"):
            self.processor.extract_frames_from_video(
                source_path=video_path,
                frame_indices=[0],
                output_dir=self.output_dir,
                output_name_base="trial1",
                mode="2D",
                camera=None,
            )

    def test_extract_frames_handles_missing_source(self):
        """Does method raise FileNotFoundError for missing video/folder?"""
        # Act & Assert
        with pytest.raises(FileNotFoundError, match="(?i)source.*not found"):
            self.processor.extract_frames_from_video(
                source_path=Path("nonexistent.avi"),
                frame_indices=[0],
                output_dir=self.output_dir,
                output_name_base="trial1",
                mode="2D",
                camera=1,
            )

    def test_extract_frames_progress_shows_human_readable_indices(self, capsys):
        """Do progress messages show 1-indexed frame numbers?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=15)

        # Act
        self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 5, 10],
            output_dir=self.output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Assert: Check stdout for 1-indexed frame numbers
        captured = capsys.readouterr()
        output = captured.out
        # Should show "Extracting frame 1", "frame 6", "frame 11" (1-indexed)
        assert "frame 1" in output.lower() or "Extracting frame 1" in output
        assert "frame 6" in output.lower() or "Extracting frame 6" in output
        assert "frame 11" in output.lower() or "Extracting frame 11" in output


class TestFrameExtractionIntegration:
    """End-to-end integration tests for frame extraction workflows"""

    @pytest.fixture(autouse=True)
    def setup_and_teardown(self, tmp_path):
        """Setup full test environment for integration tests"""
        self.working_dir = tmp_path

        # Create DLC-style config directories
        self.dlc_project = self.working_dir / "dlc_project"
        self.dlc_project.mkdir()
        (self.dlc_project / "labeled-data").mkdir()

        self.config = {
            "working_dir": str(self.working_dir),
            "path_config_file": str(self.dlc_project / "config.yaml"),
            "dataset_name": "test_dataset",
            "experimenter": "test_scorer",
            "swapped_markers": False,
            "crossed_markers": False,
        }

        self.processor = XMADataProcessor(self.config)

        yield

    def _create_test_video(self, video_path: Path, num_frames: int = 10):
        """Helper to create a test video"""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (640, 480))
        for i in range(num_frames):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            brightness = min(i * 15, 255)
            frame[:, :] = (brightness, brightness, brightness)
            out.write(frame)
        out.release()
        return video_path

    def _create_trial_with_videos(self, trial_name: str, num_frames: int = 50):
        """Helper to create a trial with cam1 and cam2 videos and CSV"""
        trial_dir = self.working_dir / "trainingdata" / trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create cam1 and cam2 videos
        self._create_test_video(trial_dir / f"{trial_name}_cam1.avi", num_frames)
        self._create_test_video(trial_dir / f"{trial_name}_cam2.avi", num_frames)

        # Create CSV with 2D points data
        data = {
            "marker1_cam1_X": [10.0 + i for i in range(num_frames)],
            "marker1_cam1_Y": [20.0 + i for i in range(num_frames)],
            "marker1_cam2_X": [30.0 + i for i in range(num_frames)],
            "marker1_cam2_Y": [40.0 + i for i in range(num_frames)],
            "marker2_cam1_X": [50.0 + i for i in range(num_frames)],
            "marker2_cam1_Y": [60.0 + i for i in range(num_frames)],
            "marker2_cam2_X": [70.0 + i for i in range(num_frames)],
            "marker2_cam2_Y": [80.0 + i for i in range(num_frames)],
        }
        df = pd.DataFrame(data)
        df.to_csv(trial_dir / f"{trial_name}.csv", index=False)

        return trial_dir

    def test_path_traversal_blocked_in_list_trials(self):
        """Does list_trials prevent directory traversal attacks?"""
        # Test various traversal attempts
        with pytest.raises(ValueError, match="(?i)security"):
            self.processor.list_trials("../outside_project")

        with pytest.raises(ValueError, match="(?i)security"):
            self.processor.list_trials("../../etc/passwd")

        with pytest.raises(ValueError, match="(?i)security"):
            self.processor.list_trials("/absolute/path")

        with pytest.raises(ValueError, match="(?i)security"):
            self.processor.list_trials("valid/../traversal")

    def test_empty_trials_directory_raises_helpful_error(self):
        """Does list_trials raise clear error for empty trials directory?"""
        # Arrange: Create empty trainingdata folder
        (self.working_dir / "trainingdata").mkdir(parents=True, exist_ok=True)

        # Act & Assert
        with pytest.raises(FileNotFoundError) as exc_info:
            self.processor.list_trials("trainingdata")

        # Check error message is helpful
        error_msg = str(exc_info.value)
        assert "trainingdata" in error_msg
        assert "no trials" in error_msg.lower()

    def test_mixed_video_and_image_folder_trials(self):
        """Can extract_frames_from_video handle mix of video files and image folders?"""
        # Arrange: trial1 with video, trial2 with image folder
        trial1_dir = self.working_dir / "trials" / "trial1"
        trial1_dir.mkdir(parents=True, exist_ok=True)
        video_path = trial1_dir / "video.avi"
        self._create_test_video(video_path, num_frames=20)

        trial2_dir = self.working_dir / "trials" / "trial2"
        img_folder = trial2_dir / "images"
        img_folder.mkdir(parents=True, exist_ok=True)
        for i in range(20):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_folder / f"img_{i:04d}.png"), img)

        output_dir = self.working_dir / "output"

        # Act: Extract from video
        result1 = self.processor.extract_frames_from_video(
            source_path=video_path,
            frame_indices=[0, 5, 10],
            output_dir=output_dir,
            output_name_base="trial1",
            mode="2D",
            camera=1,
        )

        # Extract from image folder
        result2 = self.processor.extract_frames_from_video(
            source_path=img_folder,
            frame_indices=[0, 5, 10],
            output_dir=output_dir,
            output_name_base="trial2",
            mode="2D",
            camera=1,
        )

        # Assert: Both extractions successful
        assert len(result1) == 3
        assert len(result2) == 3
        # Check naming is consistent
        assert (output_dir / "trial1_cam1_0001.png").exists()
        assert (output_dir / "trial2_cam1_0001.png").exists()

    def test_frame_extraction_with_various_compression_levels(self):
        """Do different compression levels produce valid PNGs?"""
        # Arrange
        video_path = self.working_dir / "test_video.avi"
        self._create_test_video(video_path, num_frames=10)

        # Act: Extract with different compression levels
        results = {}
        for compression in [0, 5, 9]:
            output_dir = self.working_dir / f"output_{compression}"
            self.processor.extract_frames_from_video(
                source_path=video_path,
                frame_indices=[0, 1, 2],
                output_dir=output_dir,
                output_name_base="trial1",
                mode="2D",
                camera=1,
                compression=compression,
            )
            results[compression] = output_dir / "trial1_cam1_0001.png"

        # Assert: All PNGs are valid and loadable
        for compression, png_path in results.items():
            img = cv2.imread(str(png_path))
            assert img is not None, f"Failed to load PNG with compression {compression}"

        # Assert: File sizes decrease with compression (or stay same)
        size_0 = results[0].stat().st_size
        size_5 = results[5].stat().st_size
        size_9 = results[9].stat().st_size
        assert size_9 <= size_5 <= size_0

    def test_list_trials_with_nested_suffix_path(self):
        """Does list_trials work with nested paths like 'data/experiments'?"""
        # Arrange
        nested_path = self.working_dir / "data" / "experiments"
        nested_path.mkdir(parents=True, exist_ok=True)
        (nested_path / "trial1").mkdir()
        (nested_path / "trial2").mkdir()

        # Act
        trials = self.processor.list_trials("data/experiments")

        # Assert
        assert len(trials) == 2
        trial_names = [t.name for t in trials]
        assert "trial1" in trial_names
        assert "trial2" in trial_names


if __name__ == "__main__":
    unittest.main()
