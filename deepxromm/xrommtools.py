"""
XROMM Tools for DeepLabCut
Developed by J.D. Laurence-Chasen

Functions:

xma_to_dlc: create DeepLabCut training dataset from data tracked in XMALab
analyze_xromm_videos: Predict 2D points for novel trials
dlc_to_xma: convert output of DeepLabCut to XMALab format 2D points file
add_frames: Add new frames corrected/tracked in XMALab to an existing training dataset

"""

from pathlib import Path
import pandas as pd
import numpy as np
import cv2


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


def add_frames(
    path_config_file,
    data_path,
    iteration,
    frames,
    nnetworks=1,
    path_config_file_cam2="enterpathofcam2config",
):
    # input: config file paths, path of data to add to trainingdataset, frames-csv file where first col is trialnames and following cols are frame numbers
    # will look for 2D points file based on name (if there are multiple csv files)

    configs = [path_config_file[:-12], path_config_file_cam2[:-12]]
    cameras = [1, 2]
    subs = [
        [
            "c01",
            "c1",
            "C01",
            "C1",
            "Cam1",
            "cam1",
            "Cam01",
            "cam01",
            "Camera1",
            "camera1",
        ],
        [
            "c02",
            "c2",
            "C02",
            "C2",
            "Cam2",
            "cam2",
            "Cam02",
            "cam02",
            "Camera2",
            "camera2",
        ],
    ]
    pts = [
        "2Dpts",
        "2dpts",
        "2DPts",
        "2dPts",
        "pts2D",
        "Pts2D",
        "pts2d",
        "points2D",
        "Points2d",
        "points2d",
        "2Dpoints",
        "2dpoints",
        "2DPoints",
    ]
    corr = ["correct", "Correct", "corrected", "Corrected"]
    # read frames from csv
    if ".csv" in frames:
        f = pd.read_csv(frames, header=None)
        trialnames = list(f.iloc[:, 0])  # first row of frames file must be trialnames
        picked_frames = []
        # this is disgusting code
        for row in range(f.shape[0]):
            picked_frames.append(list(f.loc[row, 1:]))
        for count, row in enumerate(picked_frames):
            picked_frames[count] = [x for x in row if str(x) != "nan"]  # remove nans
        for count, row in enumerate(picked_frames):
            picked_frames[count] = [int(x) for x in row]  # convert to int
    else:
        raise ValueError("frames must be a .csv file with trialnames and frame numbers")

    if nnetworks == 2:
        for camera in cameras:
            dlc_dataset_path = configs[camera - 1] / "labeled-data"
            contents = list(dlc_dataset_path.glob("*"))
            if len(contents) == 1:
                dataset_name = contents[0]
                labeleddata_path = dlc_dataset_path / dataset_name
            else:
                raise ValueError(
                    "There must be only one data set in the labeled-data folder"
                )

            h5file = list(labeleddata_path.glob("*.h5"))
            csvfile = list(labeleddata_path.glob("*.csv"))
            data = pd.read_hdf(labeleddata_path / h5file[0])

            ## Extract selected frames from videos

            for trialnum, trial in enumerate(trialnames):
                # get video file
                file = []
                relnames = []
                contents = data_path / trial
                for name in contents:
                    if any(x in name for x in subs[camera - 1]):
                        file = name
                if not file:
                    raise ValueError(
                        "Cannot locate %s video file or image folder" % trial
                    )

                # if video file is actually folder of frames
                video_path = data_path / trial / file
                if video_path.is_dir():
                    imgpath = video_path
                    imgs = imgpath.glob("*")
                    relpath = Path("labeled-data" / dataset_name)
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count + 1 in frames:  # ASSUMES FRAMES PROVIDED ARE 1 index
                            image = cv2.imread(imgpath / img)
                            current_img = str(count + 1).zfill(4)
                            relname = relpath / f"{trial}_{current_img}.png"
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path / trial / f"{trial}_{current_img}.png",
                                image,
                            )
                else:
                    relpath = Path("labeled-data" / dataset_name)
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video_path)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count + 1 in frames:
                            current_img = str(count + 1).zfill(4)
                            relname = relpath / f"{trial}_{current_img}.png"
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path / trial / f"{trial}_{current_img}.png",
                                image,
                            )
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # get 2D points file / data
                # extract 2D points data
                iteration_dir = data_path / trial / f"it{iteration}"
                pointsfile = list(iteration_dir.glob("*.csv"))

                if not pointsfile:
                    raise ValueError(f"Cannot locate {trial} 2D points file")

                # if multiple csv files, look for "2Dpoints" in the name
                if len(pointsfile) > 1:
                    t = []
                    for q in pointsfile:
                        if any(x in q for x in pts):
                            t = t + [q]
                    # if there are multiple 2D points files, look for "corrected" in the name
                    if len(t) > 1:
                        for r in pointsfile:
                            if any(x in r for x in corr):
                                file = r
                    else:
                        file = t[0]
                else:
                    file = pointsfile[0]
                print(
                    f"Reading and adding the following frames from {iteration_dir}/{file}"
                )
                df = pd.read_csv(iteration_dir / file, sep=",", header=None)
                df = df.loc[1:,].reset_index(drop=True)
                print(frames)
                frames = [x - 1 for x in frames]  # account for zero index in python
                xpos = df.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                if temp_data.shape[1] > data.shape[1]:
                    raise ValueError(
                        "There are %d extra points in the corrected points file"
                        % ((temp_data.shape[1] - data.shape[1]) / 2)
                    )
                if temp_data.shape[1] < data.shape[1]:
                    raise ValueError(
                        "There are %d missing points in the corrected points file"
                        % ((data.shape[1] - temp_data.shape[1]) / 2)
                    )
                temp_data.index = relnames
                temp_data.columns = data.columns
                data = pd.concat([data, temp_data])
            data.replace(" NaN", np.nan, inplace=True)
            data.replace(" NaN ", np.nan, inplace=True)
            data.replace("NaN ", np.nan, inplace=True)
            data = data.astype("float")
            data = data.round(2)
            data = data.apply(pd.to_numeric)
            data.to_hdf(
                labeleddata_path + "/" + h5file[0], key="df_with_missing", mode="w"
            )
            data.to_csv(labeleddata_path + "/" + csvfile[0], na_rep="NaN")

    else:  # default, one network for both videos
        labeled_data_dir = config / "labeled-data"
        print(str(labeled_data_dir))
        contents = list(labeled_data_dir.glob("*"))
        if len(contents) == 1:
            dataset_name = contents[0]
            labeleddata_path = labeled_data_dir / dataset_name
        else:
            raise ValueError(
                "There must be only one data set in the labeled-data folder"
            )

        h5file = list(labeleddata_path.glob("*.h5"))
        csvfile = list(labeleddata_path.glob("*.csv"))
        data = pd.read_hdf(labeleddata_path / h5file[0])  # read old point labels

        for camera in cameras:
            ## Extract selected frames from videos
            for trialnum, trial in enumerate(trialnames):
                # get video file
                relnames = []
                file = []

                trial_path = data_path / trial
                contents = trial_path.glob("*")
                for name in contents:
                    if any(x in name for x in subs[camera - 1]):
                        file = name
                if not file:
                    raise ValueError(
                        "Cannot locate %s video file or image folder" % trial
                    )

                # if video file is actually folder of frames
                video_path = data_path / trial / file
                if video_path.is_dir():
                    imgpath = video_path
                    imgs = imgpath.glob("*")
                    relpath = Path("labeled-data" / dataset_name)
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count + 1 in frames:  # ASSUMES FRAMES PROVIDED ARE 1 index
                            image = cv2.imread(imgpath / img)
                            current_img = str(count + 1).zfill(4)
                            relname = relpath / f"{trial}_cam{camera}_{current_img}.png"
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path
                                / f"{trial}_cam{camera}_{current_img}.png"
                            )
                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    relpath = Path("labeled-data" / dataset_name)
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video_path)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count + 1 in frames:
                            current_img = str(count + 1).zfill(4)
                            relname = relpath / f"{trial}_cam{camera}_{current_img}.png"
                            relnames = relnames + [relname]
                            cv2.imwrite(
                                labeleddata_path
                                / f"{trial}_cam{camera}_{current_img}.png"
                            )
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # get 2D points file / data
                # extract 2D points data
                contents = data_path / trial / f"it{iteration}"
                pointsfile = list(contents.glob("*.csv"))

                if not pointsfile:
                    raise ValueError(f"Cannot locate {trial} 2D points file")

                # if multiple csv files, look for "2Dpoints" in the name
                if len(pointsfile) > 1:
                    t = []
                    for q in pointsfile:
                        if any(x in q for x in pts):
                            t = t + [q]
                    # if there are multiple 2D points files, look for "corrected" in the name
                    if len(t) > 1:
                        for r in pointsfile:
                            if any(x in r for x in corr):
                                pointsfile = r
                else:
                    pointsfile = pointsfile[0]
                if isinstance(pointsfile, str) != True:
                    raise ValueError(
                        f"Please check the points files in trial {trial} iteration {iteration} folder"
                    )
                df = pd.read_csv(
                    data_path / trial / f"it{iteration}" / pointsfile,
                    sep=",",
                    header=None,
                )
                df = df.loc[1:,].reset_index(drop=True)
                xpos = df.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                temp_data.index = relnames
                if temp_data.shape[1] > data.shape[1]:
                    raise ValueError(
                        "There are %d extra points in the corrected points file"
                        % ((temp_data.shape[1] - data.shape[1]) / 2)
                    )
                if temp_data.shape[1] < data.shape[1]:
                    raise ValueError(
                        "There are %d missing points in the corrected points file"
                        % ((data.shape[1] - temp_data.shape[1]) / 2)
                    )
                temp_data.columns = data.columns
                data = pd.concat([data, temp_data])
        data.replace(" NaN", np.nan, inplace=True)
        data.replace(" NaN ", np.nan, inplace=True)
        data.replace("NaN ", np.nan, inplace=True)
        data = data.astype("float")
        data = data.round(2)
        data = data.apply(pd.to_numeric)
        data.to_hdf(labeleddata_path / h5file[0], key="df_with_missing", mode="w")
        data.to_csv(labeleddata_path / csvfile[0], na_rep="NaN")

    print(
        f"Frames from {len(trialnames)} trials successfully added to training dataset"
    )
