import unittest
from pathlib import Path
import shutil

from .utils import set_up_project, copy_mock_dlc_data_2cam


class TestTestAutocorrect(unittest.TestCase):
    """Test 'test_autocorrect' functionality"""

    def setUp(self):
        """Configure a project with sample predicted points data"""
        self.working_dir = Path.cwd() / "tmp"
        _, self.deepxromm_proj = set_up_project(self.working_dir, "2D")
        copy_mock_dlc_data_2cam(self.working_dir)
        self.deepxromm_proj.dlc_to_xma()

    def test_test_autocorrect_runs(self):
        """Test that the 'test_autocorrect' code path is working"""
        self.deepxromm_proj.project.autocorrect_settings.test_autocorrect = True
        self.deepxromm_proj.project.autocorrect_settings.trial_name = "test"
        self.deepxromm_proj.project.autocorrect_settings.cam = "cam1"
        self.deepxromm_proj.project.autocorrect_settings.frame_num = 1
        self.deepxromm_proj.project.autocorrect_settings.marker = "marker001"
        self.deepxromm_proj.project.update_config_file()

        self.deepxromm_proj.autocorrect_trials()

    def tearDown(self):
        """Destroy the project"""
        if self.working_dir.exists():
            shutil.rmtree(self.working_dir)
