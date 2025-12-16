"""
Verify that default behaviors of deepxromm remains the same
"""

import os
from pathlib import Path
import shutil
import unittest

import cv2
import pandas as pd
from datetime import datetime as dt
from ruamel.yaml import YAML

from deepxromm import DeepXROMM

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


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
        tmp["mode"] = "rgb"
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
        tmp["mode"] = "rgb"
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
        tmp["mode"] = "rgb"
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
        tmp["mode"] = "rgb"
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

    def test_migration_from_tracking_mode_to_mode(self):
        """Does loading a config with deprecated 'tracking_mode' auto-migrate to 'mode'?"""
        yaml = YAML()

        # Modify config to use deprecated key
        with self.config.open("r") as config:
            tmp = yaml.load(config)
        del tmp["mode"]  # Remove new key if it exists
        tmp["tracking_mode"] = "2D"  # Use deprecated key
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        # Load project (should trigger migration)
        with self.assertWarns(DeprecationWarning):
            DeepXROMM.load_project(self.working_dir)

        # Verify config was migrated and saved
        with self.config.open("r") as config:
            migrated = yaml.load(config)

        self.assertIn("mode", migrated)
        self.assertNotIn("tracking_mode", migrated)
        self.assertEqual(migrated["mode"], "2D")

    def test_conflicting_mode_values_raises_error(self):
        """Does having both 'mode' and 'tracking_mode' with different values raise ValueError?"""
        yaml = YAML()

        # Create config with conflicting values
        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["mode"] = "2D"
        tmp["tracking_mode"] = "rgb"  # Different value
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        # Verify error is raised
        with self.assertRaises(ValueError) as context:
            DeepXROMM.load_project(self.working_dir)

        error_message = str(context.exception)
        self.assertIn("Conflicting values", error_message)
        self.assertIn("mode", error_message)
        self.assertIn("tracking_mode", error_message)

    def test_duplicate_mode_values_migrates_successfully(self):
        """Does having both keys with same value migrate successfully?"""
        yaml = YAML()

        # Create config with duplicate but matching values
        with self.config.open("r") as config:
            tmp = yaml.load(config)
        tmp["mode"] = "2D"
        tmp["tracking_mode"] = "2D"  # Same value
        with self.config.open("w") as fp:
            yaml.dump(tmp, fp)

        # Load project (should trigger migration with warning)
        with self.assertWarns(DeprecationWarning):
            DeepXROMM.load_project(self.working_dir)

        # Verify only 'mode' remains
        with self.config.open("r") as config:
            migrated = yaml.load(config)

        self.assertIn("mode", migrated)
        self.assertNotIn("tracking_mode", migrated)
        self.assertEqual(migrated["mode"], "2D")

    def tearDown(self):
        """Remove the created temp project"""
        shutil.rmtree(self.working_dir)
