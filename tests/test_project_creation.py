"""
Unit tests for deepxromm project creation
"""

import os
from pathlib import Path
import shutil
import unittest

from ruamel.yaml import YAML

from deepxromm import DeepXROMM

SAMPLE_FRAME = Path(__file__).parent / "sample_frame.jpg"
SAMPLE_FRAME_INPUT = Path(__file__).parent / "sample_frame_input.csv"
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / "sample_autocorrect_output.csv"

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


class TestProjectCreation(unittest.TestCase):
    """Tests behaviors related to XMA-DLC project creation"""

    @classmethod
    def setUpClass(cls):
        """Create a sample project"""
        super(TestProjectCreation, cls).setUpClass()
        cls.project_dir = Path.cwd() / "tmp"
        DeepXROMM.create_new_project(cls.project_dir, codec=DEEPXROMM_TEST_CODEC)

    def test_project_creates_correct_folders(self):
        """Do we have all of the correct folders?"""
        for folder in ["trainingdata", "trials"]:
            with self.subTest(folder=folder):
                self.assertTrue((self.project_dir / folder).exists())

    def test_project_creates_config_file(self):
        """Do we have a project config?"""
        self.assertTrue((self.project_dir / "project_config.yaml").exists())

    def test_project_config_has_these_variables(self):
        """Can we access each of the variables that's supposed to be in the config?"""
        yaml = YAML()
        variables = [
            "task",
            "experimenter",
            "working_dir",
            "path_config_file",
            "dataset_name",
            "nframes",
            "maxiters",
            "tracking_threshold",
            "mode",
            "swapped_markers",
            "crossed_markers",
            "search_area",
            "threshold",
            "krad",
            "gsigma",
            "img_wt",
            "blur_wt",
            "gamma",
            "cam",
            "frame_num",
            "trial_name",
            "marker",
            "test_autocorrect",
            "cam1s_are_the_same_view",
        ]

        yaml = YAML()
        config_path = self.project_dir / "project_config.yaml"
        with config_path.open() as config:
            project = yaml.load(config)
            for variable in variables:
                with self.subTest(i=variable):
                    self.assertIsNotNone(project[variable])

    @classmethod
    def tearDownClass(cls):
        """Remove the created temp project"""
        super(TestProjectCreation, cls).tearDownClass()
        shutil.rmtree(cls.project_dir)
