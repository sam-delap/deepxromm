"""
XROMM Tools for DeepLabCut
Originally developed by J.D. Laurence-Chasen

Functions:

dlc_to_xma: convert output of DeepLabCut to XMALab format 2D points file

xma_to_dlc and analyze_xromm_videos have been migrated into the XMADataProcessor class
add_frames has been removed in favor of extract_outlier_frames and merge_datasets
"""

from pathlib import Path
import pandas as pd


def dlc_to_xma(trial: Path, iteration: int):
    # get filenames and read analyzed data
    trialname = trial.name
    iteration_folder = trial / f"it{iteration}"
    cam1data = list(iteration_folder.glob("*cam1*.h5"))
    cam2data = list(iteration_folder.glob("*cam2*.h5"))
    if not cam1data:
        raise ValueError(
            "Cannot find cam1 predicted points. Have you run deepxromm.analyze_videos() for this project?"
        )
    if not cam2data:
        raise ValueError(
            "Cannot find cam2 predicted points. Have you run deepxromm.analyze_videos() for this project?"
        )
    cam1data = pd.read_hdf(cam1data[0])
    cam2data = pd.read_hdf(cam2data[0])
    h5_save_path = iteration_folder / f"{trialname}-Predicted2DPoints.h5"
    csv_save_path = iteration_folder / f"{trialname}-Predicted2DPoints.csv"

    pointnames = list(cam1data.columns.get_level_values("bodyparts").unique())

    # make new column names
    nvar = len(pointnames)
    pointnames = [item for item in pointnames for _ in range(4)]
    post = ["_cam1_X", "_cam1_Y", "_cam2_X", "_cam2_Y"] * nvar
    cols = [m + str(n) for m, n in zip(pointnames, post)]

    # remove likelihood columns
    cam1data = cam1data.drop(cam1data.columns[2::3], axis=1)
    cam2data = cam2data.drop(cam2data.columns[2::3], axis=1)

    # replace col names with new indices
    c1cols = list(range(0, cam1data.shape[1] * 2, 4)) + list(
        range(1, cam1data.shape[1] * 2, 4)
    )
    c2cols = list(range(2, cam1data.shape[1] * 2, 4)) + list(
        range(3, cam1data.shape[1] * 2, 4)
    )
    c1cols.sort()
    c2cols.sort()
    cam1data.columns = c1cols
    cam2data.columns = c2cols

    df = pd.concat([cam1data, cam2data], axis=1).sort_index(axis=1)
    df.columns = cols
    df.to_hdf(h5_save_path, key="df_with_missing", mode="w")
    df.to_csv(csv_save_path, na_rep="NaN", index=False)
