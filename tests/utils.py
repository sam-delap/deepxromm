import os
from pathlib import Path
import shutil

from deepxromm import DeepXROMM

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


def set_up_project(project_dir: Path, mode: str):
    """Helper function to set up a project at a given path with a given mode"""
    deepxromm_proj = DeepXROMM.create_new_project(project_dir, mode=mode)

    # Adjust maxiters to 5 to ensure that training completes quickly
    deepxromm_proj.project.dlc_config.maxiters = 5
    deepxromm_proj.project.update_config_file()

    # Make vars for pathing to find files easily
    trial_dir = project_dir / "trainingdata/test"
    trial_dir.mkdir(parents=True, exist_ok=True)
    trial_csv = trial_dir / "test.csv"
    cam1_path = trial_dir / "test_cam1.avi"
    cam2_path = trial_dir / "test_cam2.avi"

    # Copy sample CSV data for training operations (use existing sample file)
    shutil.copy("trial_slice.csv", str(trial_csv))
    shutil.copy("trial_cam1_slice.avi", str(cam1_path))
    shutil.copy("trial_cam2_slice.avi", str(cam2_path))

    # Create sample novel trial for analysis
    novel_dir = project_dir / "trials/test"
    novel_dir.mkdir(parents=True, exist_ok=True)

    # Copy videos for analysis
    shutil.copy("trial_cam1_slice.avi", str(novel_dir / "test_cam1.avi"))
    shutil.copy("trial_cam2_slice.avi", str(novel_dir / "test_cam2.avi"))

    # Reload project
    deepxromm_proj = DeepXROMM.load_project(project_dir)

    return trial_csv, deepxromm_proj


def tear_down_project(working_dir: Path):
    """Helper function for tearing down a project"""
    if working_dir.exists():
        shutil.rmtree(working_dir)
