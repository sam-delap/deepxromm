"""
Test of the 'train_many_projects' batch training pipeline
"""

import unittest
from pathlib import Path
import shutil

from deepxromm import DeepXROMM
from .utils import set_up_project


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
        # Invoke once to run through the whole process for each project type
        DeepXROMM.train_many_projects(self.working_dir)

        # Invoke again to ensure batch workflows complete idempotently (i.e. skips are not fatal)
        DeepXROMM.train_many_projects(self.working_dir)

    def tearDown(self):
        """Remove the created temp project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
