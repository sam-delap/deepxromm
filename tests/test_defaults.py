"""
Verify that default behaviors of deepxromm remains the same
"""

import os
from pathlib import Path
import shutil
import unittest

import cv2
import pandas as pd

from deepxromm import DeepXROMM
from deepxromm.config_utilities import load_config_file, save_config_file

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class TestDefaultsPerformance(unittest.TestCase):
    """Test that the config will still be configured properly if the user only provides XMAlab input"""

    def setUp(self):
        """Create a sample project where the user only inputs XMAlab data"""
        self.working_dir = Path.cwd() / "tmp"
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
        project = deepxromm.project
        self.assertEqual(project.nframes, 1, msg=f"Actual nframes: {project.nframes}")

    def test_analyze_errors_if_no_folders_in_trials_dir(self):
        """If there are no trials to analyze, do we return an error?"""
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.analyze_videos()

    def test_autocorrect_errors_if_no_folders_in_trials_dir(self):
        """If there are no trials to autocorrect, do we return an error?"""
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.autocorrect_trials()

    def test_yaml_file_updates_nframes_after_load_if_frames_is_0(self):
        """If the user doesn't specify how many frames they want analyzed,
        does their YAML file get updated with how many are in the CSV?"""
        deepxromm_proj = DeepXROMM.load_project(self.working_dir)
        self.assertEqual(
            deepxromm_proj.project.nframes,
            1,
            msg=f"Actual nframes: {deepxromm_proj.project.nframes}",
        )

    def test_bodyparts_add_from_csv_if_not_defined(self):
        """If the user hasn't specified the bodyparts from their trial, we can pull them from the CSV"""
        deepxromm_proj = DeepXROMM.load_project(self.working_dir)
        config_obj = load_config_file(deepxromm_proj.project.path_config_file)
        self.assertEqual(config_obj["bodyparts"], ["foo", "bar", "baz"])

    def test_bodyparts_error_if_different_from_csv(self):
        """If the user specifies different bodyparts than their CSV, raise an error"""
        config_dlc = load_config_file(self.deepxromm.project.path_config_file)
        config_dlc["bodyparts"] = ["foo", "bar"]
        save_config_file(config_dlc, self.deepxromm.project.path_config_file)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def test_config_update_does_not_sort_keys(self):
        """When we update the config, does the data remain sorted according to how it is in default_config?"""
        current_config = load_config_file(self.working_dir / "project_config.yaml")
        self.deepxromm.project.update_config_file()
        updated_config = load_config_file(self.working_dir / "project_config.yaml")

        for current_key, updated_key in zip(
            current_config.keys(), updated_config.keys()
        ):
            with self.subTest(folder=current_key):
                self.assertTrue(current_key == updated_key)

    def test_migration_from_tracking_mode_to_mode(self):
        """Does loading a config with deprecated 'tracking_mode' auto-migrate to 'mode'?"""
        # Modify config to use deprecated key
        tmp = load_config_file(self.working_dir / "project_config.yaml")
        del tmp["mode"]  # Remove new key if it exists
        tmp["tracking_mode"] = "2D"  # Use deprecated key
        save_config_file(tmp, self.working_dir / "project_config.yaml")

        # Load project (should trigger migration)
        with self.assertWarns(DeprecationWarning):
            DeepXROMM.load_project(self.working_dir)

        # Verify config was migrated and saved
        migrated = load_config_file(self.working_dir / "project_config.yaml")

        self.assertIn("mode", migrated)
        self.assertNotIn("tracking_mode", migrated)
        self.assertEqual(migrated["mode"], "2D")

    def test_conflicting_mode_values_raises_error(self):
        """Does having both 'mode' and 'tracking_mode' with different values raise ValueError?"""
        # Create config with conflicting values
        tmp = load_config_file(self.working_dir / "project_config.yaml")
        tmp["mode"] = "2D"
        tmp["tracking_mode"] = "rgb"  # Different value
        save_config_file(tmp, self.working_dir / "project_config.yaml")

        # Verify error is raised
        with self.assertRaises(ValueError) as context:
            DeepXROMM.load_project(self.working_dir)

        error_message = str(context.exception)
        self.assertIn("Conflicting values", error_message)
        self.assertIn("mode", error_message)
        self.assertIn("tracking_mode", error_message)

    def test_duplicate_mode_values_migrates_successfully(self):
        """Does having both keys with same value migrate successfully?"""
        # Create config with duplicate but matching values
        tmp = load_config_file(self.working_dir / "project_config.yaml")
        tmp["mode"] = "2D"
        tmp["tracking_mode"] = "2D"  # Same value
        save_config_file(tmp, self.working_dir / "project_config.yaml")

        # Load project (should trigger migration with warning)
        with self.assertWarns(DeprecationWarning):
            DeepXROMM.load_project(self.working_dir)

        # Verify only 'mode' remains
        migrated = load_config_file(self.working_dir / "project_config.yaml")

        self.assertIn("mode", migrated)
        self.assertNotIn("tracking_mode", migrated)
        self.assertEqual(migrated["mode"], "2D")

    def tearDown(self):
        """Remove the created temp project"""
        shutil.rmtree(self.working_dir)
