import unittest
import yaml
import shutil
from pathlib import Path

from deepxromm import DeepXROMM

from .utils import set_up_project


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

        # Run through a normal workflow
        self.deepxromm_proj.xma_to_dlc()
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network()
        self.deepxromm_proj.analyze_videos()
        self.deepxromm_proj.dlc_to_xma()

    def test_retraining_workflow(self):
        """Step through the retraining workflow as a user might see it"""
        self.deepxromm_proj.extract_outlier_frames()

        iteration_dir = self.working_dir / "trials/test/it0"
        # User finds config file with outlier frames, extracts the ones they want to include
        # For this test, we'll only open the merged outliers file, but ones do exist for each camera
        with open(iteration_dir / "outliers.yaml", "r") as fp:
            outliers = yaml.safe_load(fp)

        # User edits the config file so that it only has the ones they want included in it
        with open(iteration_dir / "outliers.yaml", "w") as fp:
            yaml.dump(outliers, fp)

        # User deposits an XMAlab-formatted CSV with the word 'outliers' in it into the 'it#' folder
        # This CSV can contain much more than just the outliers they tracked
        shutil.copy(
            self.trial_csv,
            str(self.working_dir / "trials/test/it0/outliers_tracking.csv"),
        )

        # Then, we create a new dataset with the refined labels from the user
        # This will involve updating both trainingdata and the labeled dataset
        # For training data - doan update if the trial already exists as training data
        # Or create a new trial and copy the data in
        # For labeled-data - run xma_to_dlc as normal
        self.deepxromm_proj.create_refined_dataset()

        # Once we've done that, we can merge the two datasets
        # This should also update nframes
        self.deepxromm_proj.merge_datasets()

        # Encourage the user to reload their config with all of the updates
        # This should warn if they don't do this
        self.deepxromm = DeepXROMM.load_project(self.working_dir)

        # Go through the rest of the retraining workflow
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network()
        self.deepxromm_proj.analyze_videos()
        self.deepxromm_proj.dlc_to_xma()

    # def test_create_training_dataset_warns_on_noncurrent_it(self):
    #    """create_training_dataset should warn if the user doesn't reload their config after doing retraining"""
    #    assert False

    # def tearDown(self):
    #    """Tear down the project"""
    #    if self.working_dir.exists():
    #        shutil.rmtree(self.working_dir)
