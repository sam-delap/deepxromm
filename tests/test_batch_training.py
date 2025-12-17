"""
Test of the 'train_many_projects' batch training pipeline
"""

import unittest
import os
from pathlib import Path
import yaml
import shutil

from deepxromm import DeepXROMM

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


def set_up_project(project_dir: Path, mode: str):
    """Helper function to set up a project at a given path with a given mode"""
    deepxromm_proj = DeepXROMM.create_new_project(
        project_dir, mode=mode, codec=DEEPXROMM_TEST_CODEC
    )
    trial_dir = project_dir / "trainingdata/test"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Adjust maxiters to 5 to ensure that training completes quickly
    deepxromm_proj.config["maxiters"] = 5
    with (project_dir / "project_config.yaml").open("w") as fp:
        yaml.dump(deepxromm_proj.config, fp)

    # Make vars for pathing to find files easily
    trial_csv = trial_dir / "test.csv"
    cam1_path = trial_dir / "test_cam1.avi"
    cam2_path = trial_dir / "test_cam2.avi"

    # Copy sample CSV data (use existing sample file)
    shutil.copy("trial_slice.csv", str(trial_csv))
    shutil.copy("trial_cam1_slice.avi", str(cam1_path))
    shutil.copy("trial_cam2_slice.avi", str(cam2_path))


class TestBatchTrainer(unittest.TestCase):
    def setUp(self):
        """Create trial with test data"""
        self.working_dir = Path.cwd() / "tmp"
        # Create 2D base project
        self.base_project_dir = self.working_dir / "2D"
        set_up_project(self.base_project_dir, "2D")

        # Create per_cam project
        self.per_cam_project_dir = self.working_dir / "per_cam"
        set_up_project(self.per_cam_project_dir, "per_cam")

        # Create RGB project
        self.rgb_project_dir = self.working_dir / "rgb"
        set_up_project(self.rgb_project_dir, "rgb")

    def test_train_many_projects(self):
        """Invoke train_many_projects"""
        DeepXROMM.train_many_projects(self.working_dir)

    def tearDown(self):
        """Remove the created temp project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
