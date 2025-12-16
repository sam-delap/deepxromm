"""
Test elements of the trial process that do not vary between modes
"""

import os
from pathlib import Path
import shutil
import unittest

import cv2
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from deepxromm import DeepXROMM

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


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
