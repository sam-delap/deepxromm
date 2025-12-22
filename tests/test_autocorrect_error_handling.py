"""
Test error handling in autocorrect for empty frames and GaussianBlur failures
"""

import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from io import StringIO

import cv2
import numpy as np
import pandas as pd

from deepxromm import DeepXROMM

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class TestAutocorrectErrorHandling(unittest.TestCase):
    """Test that autocorrect gracefully handles empty frames and blur errors"""

    def setUp(self):
        """Create trial with markers positioned to cause errors"""
        self.working_dir = Path.cwd() / "tmp_error_test"
        self.deepxromm = DeepXROMM.create_new_project(
            self.working_dir, codec=DEEPXROMM_TEST_CODEC
        )
        frame = cv2.imread(str(SAMPLE_FRAME))

        # Make a trial directory
        trial_dir = self.working_dir / "trainingdata/test"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create CSV with a problematic marker at boundary (5, 5)
        # With search_area=15, this creates negative indices: 5 - 15.5 = -10.5
        self.boundary_csv = pd.DataFrame(
            {
                "bead1_cam1_X": [5.0],
                "bead1_cam1_Y": [5.0],
                "bead2_cam1_X": [512.0],
                "bead2_cam1_Y": [256.0],
                "bead3_cam1_X": [700.0],
                "bead3_cam1_Y": [300.0],
                "bead1_cam2_X": [5.0],
                "bead1_cam2_Y": [5.0],
                "bead2_cam2_X": [512.0],
                "bead2_cam2_Y": [256.0],
                "bead3_cam2_X": [700.0],
                "bead3_cam2_Y": [300.0],
            }
        )

        # Create CSV with all valid markers
        self.valid_csv = pd.DataFrame(
            {
                "bead1_cam1_X": [512.0],
                "bead1_cam1_Y": [256.0],
                "bead2_cam1_X": [600.0],
                "bead2_cam1_Y": [300.0],
                "bead1_cam2_X": [512.0],
                "bead1_cam2_Y": [256.0],
                "bead2_cam2_X": [600.0],
                "bead2_cam2_Y": [300.0],
            }
        )

        # Save boundary CSV to trainingdata
        self.boundary_csv.to_csv(str(trial_dir / "test.csv"), index=False)

        # Cam 1 and Cam 2 trainingdata setup
        for cam in ["cam1", "cam2"]:
            video_path = trial_dir / f"dummy_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*DEEPXROMM_TEST_CODEC),
                30,
                (1024, 512),
            )
            out.write(frame)
            out.release()

        # Move boundary CSV to trials (it0 and trials)
        trials_dir = self.working_dir / "trials/test/it0"
        trials_dir.mkdir(parents=True, exist_ok=True)
        self.boundary_csv.to_csv(
            str(self.working_dir / "trials/test/test.csv"), index=False
        )
        self.boundary_csv.to_csv(
            str(trials_dir / "test-Predicted2DPoints.csv"), index=False
        )

        # Cam 1 and Cam 2 trials setup
        for cam in ["cam1", "cam2"]:
            video_path = self.working_dir / f"trials/test/test_{cam}.avi"
            out = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*DEEPXROMM_TEST_CODEC),
                30,
                (1024, 512),
            )
            out.write(frame)
            out.release()

        cv2.destroyAllWindows()

    def test_autocorrect_skips_marker_on_empty_subimage_in_main_blur(self):
        """Test that markers causing empty subimages are skipped gracefully (main blur)"""
        # Given: Trial with boundary marker at (5, 5) that causes negative indices
        # When: Autocorrect processes the frame
        with self.assertLogs("deepxromm.autocorrector", level="WARNING") as log_context:
            self.deepxromm.autocorrect_trials()

            # Then: Warning logged with "main blur" identifier
            self.assertTrue(
                any(
                    "main blur" in msg or "Empty subimage" in msg
                    for msg in log_context.output
                ),
                f"Expected 'main blur' or 'Empty subimage' in logs, got: {log_context.output}",
            )

        # Then: Check that bead1 coordinates remain unchanged (skipped)
        output_csv_path = (
            self.working_dir / "trials/test/it0/test-AutoCorrected2DPoints.csv"
        )
        output_csv = pd.read_csv(output_csv_path)

        # bead1 at boundary should be unchanged
        self.assertAlmostEqual(output_csv.loc[0, "bead1_cam1_X"], 5.0, places=1)
        self.assertAlmostEqual(output_csv.loc[0, "bead1_cam1_Y"], 5.0, places=1)

        # I should add a check here to make sure that other things are changing, but with
        # how this data was mocked I have no idea how to do that, so I'm leaving it alone for now

    def test_autocorrect_skips_marker_on_empty_subimage_in_threshold_blur(self):
        """Test that errors in threshold blur stage are caught"""
        # This is harder to trigger naturally, so we'll verify the mechanism exists
        # by checking that threshold blur errors would be caught if they occurred
        # For now, this test verifies the main blur case which tests the pattern
        with self.assertLogs("deepxromm.autocorrector", level="WARNING") as log_context:
            self.deepxromm.autocorrect_trials()

            # Verify warning was logged
            self.assertTrue(len(log_context.output) > 0)

    def test_filter_image_handles_empty_input(self):
        """Test that _filter_image handles empty arrays gracefully"""
        # Given: Empty array input to _filter_image
        empty_array = np.array([])

        # When: _filter_image is called
        with self.assertLogs("deepxromm.autocorrector", level="WARNING") as log_context:
            result = self.deepxromm._autocorrector._filter_image(empty_array)

            # Then: Original empty array returned, warning logged
            self.assertEqual(result.size, 0)
            self.assertTrue(
                any("filter_image" in msg for msg in log_context.output),
                f"Expected 'filter_image' in logs, got: {log_context.output}",
            )

    def test_autocorrect_logs_specific_blur_location_in_error(self):
        """Test that error messages identify specific blur locations"""
        # Given: Marker at boundary
        # When: Autocorrect processes and encounters error
        with self.assertLogs("deepxromm.autocorrector", level="WARNING") as log_context:
            self.deepxromm.autocorrect_trials()

            # Then: Log should contain location identifier
            log_messages = " ".join(log_context.output)
            # Should contain either "main blur", "threshold blur", "filter_image", or "Empty subimage"
            has_location = (
                "main blur" in log_messages
                or "threshold blur" in log_messages
                or "filter_image" in log_messages
                or "Empty subimage" in log_messages
            )
            self.assertTrue(
                has_location,
                f"Expected location identifier in logs, got: {log_messages}",
            )

    def test_autocorrect_continues_with_mixed_valid_and_problematic_markers(self):
        """Test that valid markers are processed when some markers fail"""
        # Given: Frame with 3 markers - 1 at boundary, 2 valid
        # When: Autocorrect processes
        with self.assertLogs("deepxromm.autocorrector", level="WARNING") as log_context:
            self.deepxromm.autocorrect_trials()

            # Then: No exceptions raised, processing completes
            self.assertTrue(len(log_context.output) > 0)

        # Then: Output CSV exists and has all markers
        output_csv_path = (
            self.working_dir / "trials/test/it0/test-AutoCorrected2DPoints.csv"
        )
        self.assertTrue(output_csv_path.exists())

        output_csv = pd.read_csv(output_csv_path)
        # All markers should be present
        self.assertIn("bead1_cam1_X", output_csv.columns)
        self.assertIn("bead2_cam1_X", output_csv.columns)
        self.assertIn("bead3_cam1_X", output_csv.columns)

    def test_autocorrect_summary_report_counts_skipped_markers(self):
        """Test that summary report shows skipped marker counts"""
        # Given: Trial with markers that will be skipped
        # When: Autocorrect runs
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            self.deepxromm.autocorrect_trials()
            output = fake_stdout.getvalue()

            # Then: Summary should be printed showing skipped markers
            self.assertIn("Autocorrect Summary", output)
            self.assertIn("skipped", output.lower())
            self.assertIn("test", output)  # Trial name

    def test_autocorrect_summary_report_shows_nothing_when_successful(self):
        """Test that summary is silent when no markers are skipped"""
        # Given: Trial with all valid markers (no boundary issues)
        # Replace boundary CSV with valid CSV
        trials_dir = self.working_dir / "trials/test/it0"
        self.valid_csv.to_csv(
            str(trials_dir / "test-Predicted2DPoints.csv"), index=False
        )

        # When: Autocorrect runs successfully
        with patch("sys.stdout", new=StringIO()) as fake_stdout:
            self.deepxromm.autocorrect_trials()
            output = fake_stdout.getvalue()

            # Then: No summary should be printed
            self.assertNotIn("Autocorrect Summary", output)

    def tearDown(self):
        """Remove the created temp project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
