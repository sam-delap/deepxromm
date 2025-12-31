import unittest
import shutil
from pathlib import Path

from deepxromm import DeepXROMM
from deepxromm.config_utilities import load_config_file, save_config_file

from .utils import set_up_project


def generic_snapshot_updated_in_pose_config(deepxromm_proj: DeepXROMM, trial_csv: Path):
    """Given a default deepxromm project, when the user merges in a dataset, then the init_weights in the pose_config.yaml should be updated"""
    # Extract outliers
    deepxromm_proj.extract_outlier_frames()
    # Copy in tracked points for outliers
    shutil.copy(
        trial_csv,
        str(
            deepxromm_proj.project.working_dir / "trials/test/it0/outliers_tracking.csv"
        ),
    )
    # Merge datasets
    deepxromm_proj.merge_datasets()

    # Reload project (update DLC iteration)
    deepxromm_proj = DeepXROMM.load_project(deepxromm_proj.project.working_dir)

    # Remove old labeled data
    shutil.rmtree(
        deepxromm_proj.project.dlc_config.path_config_file.parent
        / "labeled-data"
        / deepxromm_proj.project.dataset_name
    )

    # Create new labeled data
    deepxromm_proj.xma_to_dlc()

    # Create training dataset (should also update pose_config.yaml for nonzero iterations)
    deepxromm_proj.create_training_dataset()

    # Check if the config got updated
    pose_config_path = deepxromm_proj._network._find_pose_cfg(
        deepxromm_proj.project.dlc_config.path_config_file,
        deepxromm_proj.project.dlc_config.iteration,
    )
    pose_config = load_config_file(pose_config_path)
    return "resnet_v1_50.ckpt" not in pose_config["init_weights"]


def generic_nframes_not_updated_when_false(
    deepxromm_proj: DeepXROMM, working_dir: Path, trial_csv: Path
) -> Path:
    """Given update_nframes is set to False, when the user runs merge_datasets, nframes in config is not updated"""
    deepxromm_proj.extract_outlier_frames()

    iteration_dir = working_dir / "trials/test/it0"
    # User finds config file with outlier frames, extracts the ones they want to include
    # For this test, we'll only open the merged outliers file, but ones do exist for each camera
    outliers = load_config_file(iteration_dir / "outliers.yaml")

    # User edits the config file so that it only has the ones they want included in it
    outliers = outliers[:5]
    save_config_file(outliers, iteration_dir / "outliers.yaml")

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
    return deepxromm_proj.project.nframes == 5


def generic_test_retraining_workflow(
    deepxromm_proj: DeepXROMM, working_dir: Path, trial_csv: Path
):
    """Generic implementation of the retraining workflow that can be used across all 3 project modes"""
    deepxromm_proj.extract_outlier_frames()

    iteration_dir = working_dir / "trials/test/it0"

    # User finds config file with outlier frames, extracts the ones they want to include
    outliers = load_config_file(iteration_dir / "outliers.yaml")

    # User edits the config file so that it only has the ones they want included in it
    outliers = outliers[:5]
    save_config_file(outliers, iteration_dir / "outliers.yaml")

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
    assert deepxromm_proj.project.nframes == 10

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
        self.deepxromm_proj.project.nframes = 5
        self.deepxromm_proj.project.update_config_file()

        # Reload project (update nframes)
        self.deepxromm_proj = DeepXROMM.load_project(self.working_dir)

        # Run through a normal workflow
        self.deepxromm_proj.xma_to_dlc()
        self.deepxromm_proj.create_training_dataset()
        self.deepxromm_proj.train_network(saveiters=1)
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

    def test_snapshot_updated_in_pose_config(self):
        assert generic_snapshot_updated_in_pose_config(
            self.deepxromm_proj, self.trial_csv
        )

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
        self.deepxromm_proj.project.nframes = 5
        self.deepxromm_proj.project.update_config_file()

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

    def test_snapshot_updated_in_pose_config(self):
        assert generic_snapshot_updated_in_pose_config(
            self.deepxromm_proj, self.trial_csv
        )
        pose_config_path_2 = self.deepxromm_proj._network._find_pose_cfg(
            self.deepxromm_proj.project.path_config_file,
            self.deepxromm_proj.project.dlc_iteration,
        )
        pose_config = load_config_file(pose_config_path_2)
        assert "resnet_v1_50.ckpt" not in pose_config["init_weights"]

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
        self.deepxromm_proj.project.nframes = 5
        self.deepxromm_proj.project.update_config_file()

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

    def test_snapshot_updated_in_pose_config(self):
        assert generic_snapshot_updated_in_pose_config(
            self.deepxromm_proj, self.trial_csv
        )

    def tearDown(self):
        """Tear down the project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
