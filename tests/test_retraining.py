import unittest
import yaml
import shutil
from pathlib import Path

from deepxromm import DeepXROMM

from .utils import set_up_project


def generic_nframes_not_updated_when_false(
    deepxromm_proj: DeepXROMM, working_dir: Path, trial_csv: Path
) -> Path:
    """Given update_nframes is set to False, when the user runs merge_datasets, nframes in config is not updated"""
    deepxromm_proj.extract_outlier_frames()

    iteration_dir = working_dir / "trials/test/it0"
    # User finds config file with outlier frames, extracts the ones they want to include
    # For this test, we'll only open the merged outliers file, but ones do exist for each camera
    with open(iteration_dir / "outliers.yaml", "r") as fp:
        outliers = yaml.safe_load(fp)

    # User edits the config file so that it only has the ones they want included in it
    outliers = outliers[:5]
    with open(iteration_dir / "outliers.yaml", "w") as fp:
        yaml.dump(outliers, fp)

    # User deposits an XMAlab-formatted CSV with the word 'outliers' in it into the 'it#' folder
    # This CSV can contain much more than just the outliers they tracked
    shutil.copy(
        trial_csv,
        str(working_dir / "trials/test/it0/outliers_tracking.csv"),
    )

    deepxromm_proj.merge_datasets(update_nframes=False)

    # Encourage the user to reload their config with all of the updates
    # This should warn if they don't do this
    deepxromm_proj = DeepXROMM.load_project(working_dir)

    # Check that nframes matches our expectations (5 initial frames)
    return deepxromm_proj.config["nframes"] == 5


def generic_test_retraining_workflow(
    deepxromm_proj: DeepXROMM, working_dir: Path, trial_csv: Path
):
    """Generic implementation of the retraining workflow that can be used across all 3 project modes"""
    deepxromm_proj.extract_outlier_frames()

    iteration_dir = working_dir / "trials/test/it0"

    # User finds config file with outlier frames, extracts the ones they want to include
    with open(iteration_dir / "outliers.yaml", "r") as fp:
        outliers = yaml.safe_load(fp)

    # User edits the config file so that it only has the ones they want included in it
    outliers = outliers[:5]
    with open(iteration_dir / "outliers.yaml", "w") as fp:
        yaml.dump(outliers, fp)

    # User deposits an XMAlab-formatted CSV with the word 'outliers' in it into the 'it#' folder
    # This CSV can contain much more than just the outliers they tracked
    shutil.copy(
        trial_csv,
        str(working_dir / "trials/test/it0/outliers_tracking.csv"),
    )

    # Merge the outliers into the user's existing dataset
    deepxromm_proj.merge_datasets()

    # Encourage the user to reload their config with all of the updates... warn them if they don't do this
    deepxromm_proj = DeepXROMM.load_project(working_dir)

    # Check that nframes matches our expectations (5 initial frames + 5 outlier frames = 10 total frames)
    assert deepxromm_proj.config["nframes"] == 10

    # Go through the rest of the retraining workflow
    deepxromm_proj.xma_to_dlc()
    deepxromm_proj.create_training_dataset()
    deepxromm_proj.train_network()
    deepxromm_proj.analyze_videos()
    deepxromm_proj.dlc_to_xma()


class TestRetraining2D(unittest.TestCase):
    """Test retraining on a 2D project"""

    def setUp(self):
        """Set up the project and go through a full workflow prior to re-training"""
        self.working_dir = Path.cwd() / "tmp"
        # Create/load deepxromm project
        self.trial_csv, self.deepxromm_proj = set_up_project(
            self.working_dir, mode="2D"
        )

        # Give a subset of frames
        self.deepxromm_proj.config["nframes"] = 5
        with open(self.working_dir / "project_config.yaml", "w") as fp:
            yaml.dump(self.deepxromm_proj.config, fp, sort_keys=False)

        # Reload project (update nframes)
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)

        # Run through a normal workflow
        self.deepxromm_proj.xma_to_dlc()
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network()
        self.deepxromm_proj.analyze_videos()
        self.deepxromm_proj.dlc_to_xma()

    def test_retraining_workflow(self):
        """Step through the retraining workflow as a user might see it"""
        generic_test_retraining_workflow(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    def test_nframes_not_updated_when_false(self):
        """Given update_nframes is set to False, when the user runs merge_datasets, nframes in config is not updated"""
        assert generic_nframes_not_updated_when_false(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    # Test that nframes is not updated when user specifies 'update_nframes=False'
    # Test that outlier collection actually adds the correct frames to a DF index in XMAlab format
    # Test that load_project() warns after retraining is initiated

    def tearDown(self):
        """Tear down the project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)


class TestRetrainingPerCam(unittest.TestCase):
    """Test retraining on a per_cam project"""

    def setUp(self):
        """Set up the project and go through a full workflow prior to re-training"""
        self.working_dir = Path.cwd() / "tmp"
        # Create/load deepxromm project
        self.trial_csv, self.deepxromm_proj = set_up_project(
            self.working_dir, mode="per_cam"
        )

        # Give a subset of frames
        self.deepxromm_proj.config["nframes"] = 5
        with open(self.working_dir / "project_config.yaml", "w") as fp:
            yaml.dump(self.deepxromm_proj.config, fp, sort_keys=False)

        # Reload project (update nframes)
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)

        # Run through a normal workflow
        self.deepxromm_proj.xma_to_dlc()
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network()
        self.deepxromm_proj.analyze_videos()
        self.deepxromm_proj.dlc_to_xma()

    def test_retraining_workflow(self):
        """Step through the retraining workflow as a user might see it"""
        generic_test_retraining_workflow(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    def test_nframes_not_updated_when_false(self):
        """Given update_nframes is set to False, when the user runs merge_datasets, nframes in config is not updated"""
        assert generic_nframes_not_updated_when_false(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    # Test that outlier collection actually adds the correct frames to a DF index in XMAlab format
    # Test that xma_to_dlc() warns if it's clear that retraining has happened for the current iteration

    def tearDown(self):
        """Tear down the project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)


class TestRetrainingRGB(unittest.TestCase):
    """Test retraining on a per_cam project"""

    def setUp(self):
        """Set up the project and go through a full workflow prior to re-training"""
        self.working_dir = Path.cwd() / "tmp"
        # Create/load deepxromm project
        self.trial_csv, self.deepxromm_proj = set_up_project(
            self.working_dir, mode="rgb"
        )

        # Give a subset of frames
        self.deepxromm_proj.config["nframes"] = 5
        with open(self.working_dir / "project_config.yaml", "w") as fp:
            yaml.dump(self.deepxromm_proj.config, fp, sort_keys=False)

        # Reload project (update nframes)
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)

        # Run through a normal workflow
        self.deepxromm_proj.xma_to_dlc()
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network()
        self.deepxromm_proj.analyze_videos()
        self.deepxromm_proj.dlc_to_xma()

    def test_retraining_workflow(self):
        """Step through the retraining workflow as a user might see it"""
        generic_test_retraining_workflow(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    def test_nframes_not_updated_when_false(self):
        """Given update_nframes is set to False, when the user runs merge_datasets, nframes in config is not updated"""
        assert generic_nframes_not_updated_when_false(
            self.deepxromm_proj, self.working_dir, self.trial_csv
        )

    # Test that outlier collection actually adds the correct frames to a DF index in XMAlab format
    # Test that load_project() warns after retraining is initiated before config reload
    # Test that init_weights get updated to use the user's snapshot from their last iteration as base weights

    def tearDown(self):
        """Tear down the project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
