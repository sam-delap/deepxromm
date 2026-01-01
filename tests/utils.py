import os
from pathlib import Path
import shutil

import pandas as pd

from deepxromm import DeepXROMM

DEEPXROMM_TEST_CODEC = os.environ.get("DEEPXROMM_TEST_CODEC", "avc1")


def set_up_project(project_dir: Path, mode: str):
    """Helper function to set up a project at a given path with a given mode"""
    deepxromm_proj = DeepXROMM.create_new_project(project_dir, mode=mode)

    # Is implementation-dependent... ideally I'd have a better way to check if this needs to be modified
    if "_video_codec" in vars(deepxromm_proj.project.dlc_config):
        deepxromm_proj.project.dlc_config.video_codec = DEEPXROMM_TEST_CODEC

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


def copy_mock_dlc_data_rgb(project_dir: Path) -> Path:
    """Copy in mock DLC data for analysis to avoid running train/analysis steps"""
    # Copy in mock DLC data
    rgb_df = pd.read_hdf("trial_rgbdlc.h5")
    output_dir = project_dir / "trials/test/it0"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_rgb_h5 = output_dir / "test_rgbDLC_resnet50_test_projectDec1shuffle1_100000.h5"
    mock_rgb_csv = (
        output_dir / "test_rgbDLC_resnet50_test_projectDec1shuffle1_100000.csv"
    )
    rgb_df.to_hdf(mock_rgb_h5, key="df_with_missing", mode="w")
    rgb_df.to_csv(mock_rgb_csv, na_rep="NaN")

    return mock_rgb_h5


def copy_mock_dlc_data_2cam(project_dir: Path) -> tuple[Path, Path]:
    cam1_df = pd.read_hdf("trial_cam1dlc.h5")
    cam2_df = pd.read_hdf("trial_cam2dlc.h5")
    output_dir = project_dir / "trials/test/it0"
    output_dir.mkdir(parents=True, exist_ok=True)
    mock_cam1_h5 = (
        output_dir / "test_cam1DLC_resnet50_test_projectDec1shuffle1_100000.h5"
    )
    mock_cam2_h5 = (
        output_dir / "test_cam2DLC_resnet50_test_projectDec1shuffle1_100000.h5"
    )
    mock_cam1_csv = (
        output_dir / "test_cam1DLC_resnet50_test_projectDec1shuffle1_100000.csv"
    )
    mock_cam2_csv = (
        output_dir / "test_cam2DLC_resnet50_test_projectDec1shuffle1_100000.csv"
    )
    cam1_df.to_hdf(mock_cam1_h5, key="df_with_missing", mode="w")
    cam2_df.to_hdf(mock_cam2_h5, key="df_with_missing", mode="w")
    cam1_df.to_csv(mock_cam1_csv, na_rep="NaN")
    cam2_df.to_csv(mock_cam2_csv, na_rep="NaN")

    return mock_cam1_h5, mock_cam2_h5
