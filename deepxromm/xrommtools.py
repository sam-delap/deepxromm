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
import random
from deeplabcut.pose_estimation_tensorflow.predict_videos import analyze_videos


# TODO: Un-nest all of J.D.'s code
def xma_to_dlc(
    path_config_file: Path,
    data_path: Path,
    dataset_name,
    scorer,
    nframes,
    path_config_file_cam2: Path | None = None,
    nnetworks=1,
):
    config = path_config_file.parent
    cameras = [1, 2]
    picked_frames = []
    dfs = []
    idx = []
    pnames = []
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
    trialnames = [
        folder
        for folder in data_path.glob("*")
        if (data_path / folder).is_dir() and not folder.name.startswith(".")
    ]

    ### PART 1: Pick frames for dataset

    for trial in trialnames:
        # Read 2D points file
        contents = list(trial.glob("*.csv"))
        if len(contents) != 1:
            raise ValueError("Found more CSVs than expected")
        filename = contents[0]
        df1 = pd.read_csv(filename, sep=",", header=None)

        # read pointnames from header row
        pointnames = df1.loc[0, ::4].astype(str).str[:-7].tolist()
        pnames.append(pointnames)

        df1 = df1.loc[1:,].reset_index(drop=True)  # remove header row

        # temp_idx = rows where fewer than half of columns are NaN
        ncol = df1.shape[1]
        temp_idx = list(df1.index.values[(~pd.isnull(df1)).sum(axis=1) >= ncol / 2])

        # randomize frames w/i each trial and append to index list
        random.shuffle(temp_idx)
        idx.append(temp_idx)
        dfs.append(df1)

    # a couple errors
    if sum(len(x) for x in idx) < nframes:
        raise ValueError("nframes is bigger than number of detected frames")

    # if pointnames aren't the same across trials
    if any(pnames[0] != x for x in pnames):
        raise ValueError("Make sure point names are consistent across trials")

        # Helper function for frame selection algorithm

    def _extract_frame_selection_loop(idx: list, nframes: int):
        """Extract the existing frame selection loop logic"""
        picked_frames = []
        count = 0
        while sum(len(x) for x in picked_frames) < nframes:
            for trialnum in range(len(idx)):
                if sum(len(x) for x in picked_frames) < nframes:
                    if count == 0:
                        picked_frames.insert(trialnum, [idx[trialnum][count]])
                    elif count < len(idx[trialnum]):
                        picked_frames[trialnum] = picked_frames[trialnum] + [
                            idx[trialnum][count]
                        ]
                count += 1
        return picked_frames

    count = 0
    while sum(len(x) for x in picked_frames) < nframes:
        for trialnum in range(len(idx)):
            if sum(len(x) for x in picked_frames) < nframes:
                if count == 0:
                    picked_frames.insert(trialnum, [idx[trialnum][count]])
                elif count < len(idx[trialnum]):
                    picked_frames[trialnum] = picked_frames[trialnum] + [
                        idx[trialnum][count]
                    ]
        count += 1

    ### Part 2: Extract images and 2D point data

    if nnetworks == 2 and path_config_file_cam2 is not None:
        configs = [path_config_file.parent, path_config_file_cam2.parent]

        for camera in cameras:
            print(f"Extracting camera {camera} trial images and 2D points...")
            relnames = []
            data = pd.DataFrame()
            # new training dataset folder
            newpath = (
                configs[camera - 1] / "labeled-data" / f"{dataset_name}_cam{camera}"
            )
            h5_save_path = newpath / f"CollectedData_{scorer}.h5"
            csv_save_path = newpath / f"CollectedData_{scorer}.csv"

            if newpath.exists():
                contents = list(newpath.glob("*"))
                if len(contents) > 0:
                    raise ValueError(
                        f"There are already data in the camera {camera} training dataset folder"
                    )
            else:
                newpath.mkdir(parents=True, exist_ok=True)  # make new folder

            for trialnum, trial_path in enumerate(trialnames):
                # get video file
                trial_name = trial_path.name
                file = None
                contents = trial_path.glob("*")
                for name in contents:
                    if any(x in name.name for x in subs[camera - 1]):
                        file = name
                if file is None:
                    raise ValueError(
                        f"Cannot locate {trial_name} video file or image folder"
                    )

                # if video file is actually folder of frames
                video_path = trial_path / file
                if video_path.is_dir():
                    imgpath = video_path
                    imgs = imgpath.glob("*")
                    frames = picked_frames[trialnum]
                    frames.sort()
                    for count, img in enumerate(imgs):
                        if count not in frames:
                            continue
                        image = cv2.imread(imgpath / img)
                        current_img = str(count + 1).zfill(4)
                        relname = str(newpath / f"{trial_name}_{current_img}.png")
                        relnames.append(relname)
                        cv2.imwrite(relname, image)
                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video_path)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count in frames:
                            current_img = str(count + 1).zfill(4)
                            relname = str(newpath / f"{trial_name}_{current_img}.png")
                            relnames.append(relname)
                            cv2.imwrite(relname, image)
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # extract 2D points data
                df1 = dfs[trialnum]
                xpos = df1.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df1.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                data = pd.concat([data, temp_data])

            ### Part 3: Complete final structure of datafiles
            dataFrame = pd.DataFrame()
            temp = np.empty(
                (
                    data.shape[0],
                    2,
                )
            )
            temp[:] = np.nan
            for i, bodypart in enumerate(pointnames):
                index = pd.MultiIndex.from_product(
                    [[scorer], [bodypart], ["x", "y"]],
                    names=["scorer", "bodyparts", "coords"],
                )
                frame = pd.DataFrame(temp, columns=index, index=relnames)
                frame.iloc[:, 0:2] = data.iloc[:, 2 * i : 2 * i + 2].values.astype(
                    float
                )
                dataFrame = pd.concat([dataFrame, frame], axis=1)
            dataFrame.replace("", np.nan, inplace=True)
            dataFrame.replace(" NaN", np.nan, inplace=True)
            dataFrame.replace(" NaN ", np.nan, inplace=True)
            dataFrame.replace("NaN ", np.nan, inplace=True)
            dataFrame.apply(pd.to_numeric)
            dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
            dataFrame.to_csv(csv_save_path, na_rep="NaN")
            print("...done.")

    else:
        relnames = []
        data = pd.DataFrame()
        # new training dataset folder
        newpath = config / "labeled-data" / dataset_name
        h5_save_path = newpath / f"CollectedData_{scorer}.h5"
        csv_save_path = newpath / f"CollectedData_{scorer}.csv"

        if newpath.exists():
            contents = list(newpath.glob("*"))
            if len(contents) > 0:
                raise ValueError(
                    f"There are already data in the training dataset folder"
                )
        else:
            newpath.mkdir(parents=True, exist_ok=True)

        for camera in cameras:
            print(f"Extracting camera {camera} trial images and 2D points...")

            for trialnum, trial in enumerate(trialnames):
                # get video file
                file = None
                trial_path = data_path / trial
                contents = trial_path.glob("*")
                for name in contents:
                    if any(x in name.name for x in subs[camera - 1]):
                        file = name
                if file is None:
                    raise ValueError(
                        f"Cannot locate {trial} video file or image folder"
                    )

                # if video file is actually folder of frames
                video_path = trial_path / file
                if video_path.is_dir():
                    imgpath = video_path
                    imgs = imgpath.glob("*")
                    relpath = Path(f"labeled-data/{dataset_name}")
                    frames = picked_frames[trialnum]
                    frames.sort()

                    for count, img in enumerate(imgs):
                        if count not in frames:
                            continue
                        current_img = str(count + 1).zfill(4)
                        image = cv2.imread(imgpath / img)
                        relname = relpath / f"{trial}_{current_img}.png"
                        relnames = relnames + [relname]
                        cv2.imwrite(newpath / f"{trial}_{current_img}.png", image)

                else:
                    # file is actually a file
                    # extract frames from video and convert to png
                    relative_path_stub = Path("labeled-data")
                    relpath = relative_path_stub / dataset_name
                    frames = picked_frames[trialnum]
                    frames.sort()
                    cap = cv2.VideoCapture(video_path)
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count in frames:
                            current_img = str(count + 1).zfill(4)
                            relname = str(
                                newpath / f"{trial.name}_cam{camera}_{current_img}.png"
                            )
                            relnames.append(relname)
                            cv2.imwrite(relname, image)
                        success, image = cap.read()
                        count += 1
                    cap.release()

                # extract 2D points data
                df1 = dfs[trialnum]
                xpos = df1.iloc[frames, 0 + (camera - 1) * 2 :: 4]
                ypos = df1.iloc[frames, 1 + (camera - 1) * 2 :: 4]
                temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)
                temp_data.columns = range(temp_data.shape[1])
                data = pd.concat([data, temp_data])

        ### Part 3: Complete final structure of datafiles
        dataFrame = pd.DataFrame()
        temp = np.empty(
            (
                data.shape[0],
                2,
            )
        )
        temp[:] = np.nan
        for i, bodypart in enumerate(pointnames):
            index = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(temp, columns=index, index=relnames)
            frame.iloc[:, 0:2] = data.iloc[:, 2 * i : 2 * i + 2].values.astype(float)
            dataFrame = pd.concat([dataFrame, frame], axis=1)
        dataFrame.replace("", np.nan, inplace=True)
        dataFrame.replace(" NaN", np.nan, inplace=True)
        dataFrame.replace(" NaN ", np.nan, inplace=True)
        dataFrame.replace("NaN ", np.nan, inplace=True)
        dataFrame.apply(pd.to_numeric)
        dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
        dataFrame.to_csv(csv_save_path, na_rep="NaN")
        print("...done.")

    print(
        "Training data extracted to projectpath/labeled-data. Now use deeplabcut.create_training_dataset"
    )


def dlc_to_xma(cam1data, cam2data, trialname, savepath):
    h5_save_path = savepath + "/" + trialname + "-Predicted2DPoints.h5"
    csv_save_path = savepath + "/" + trialname + "-Predicted2DPoints.csv"

    if isinstance(cam1data, str):  # is string
        if ".csv" in cam1data:
            cam1data = pd.read_csv(cam1data, sep=",", header=None)
            cam2data = pd.read_csv(cam2data, sep=",", header=None)
            pointnames = list(cam1data.loc[1, 1:].unique())

            # reformat CSV / get rid of headers
            cam1data = cam1data.loc[3:, 1:]
            cam1data.columns = range(cam1data.shape[1])
            cam1data.index = range(cam1data.shape[0])
            cam2data = cam2data.loc[3:, 1:]
            cam2data.columns = range(cam2data.shape[1])
            cam2data.index = range(cam2data.shape[0])

        elif ".h5" in cam1data:  # is .h5 file
            cam1data = pd.read_hdf(cam1data)
            cam2data = pd.read_hdf(cam2data)
            pointnames = list(cam1data.columns.get_level_values("bodyparts").unique())

        else:
            raise ValueError("2D point input is not in correct format")
    else:
        pointnames = list(cam1data.columns.get_level_values("bodyparts").unique())

    # make new column names
    nvar = len(pointnames)
    pointnames = [item for item in pointnames for repetitions in range(4)]
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


def analyze_xromm_videos(
    path_config_file,
    path_data_to_analyze: str,
    iteration,
    nnetworks=1,
    path_config_file_cam2=[],
):
    # assumes you have cam1 and cam2 videos as .avi in their own seperate trial folders
    # assumes all folders w/i new_data_path are trial folders
    # convert jpg stacks?

    # analyze videos
    cameras = [1, 2]
    config = path_config_file
    configs = [path_config_file, path_config_file_cam2]
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
    data_analysis_path = Path(path_data_to_analyze)
    trialnames = [
        folder
        for folder in data_analysis_path.glob("*")
        if (data_analysis_path / folder).is_dir() and not folder.name.startswith(".")
    ]

    for trialnum, trial in enumerate(trialnames):
        trialpath = data_analysis_path / trial
        contents = trialpath.glob("*")
        savepath = trialpath / f"it{iteration}"
        if savepath.exists():
            temp = savepath.glob("*")
            if temp:
                raise ValueError(
                    f"There are already predicted points in iteration {iteration} subfolders"
                )
        else:
            savepath.mkdir(parents=True, exist_ok=True)  # make new folder
        # get video file
        filename = None
        for camera in cameras:
            for name in contents:
                if any(x in name.name for x in subs[camera - 1]):
                    filename = name.name
            if filename is None:
                raise ValueError("Cannot locate %s video file or image folder" % trial)

            video = trialpath / filename
            print(video)
            # analyze video
            if nnetworks == 1:
                analyze_videos(config, [video], destfolder=savepath, save_as_csv=True)
            else:
                analyze_videos(
                    configs[camera - 1], [video], destfolder=savepath, save_as_csv=True
                )

        # get filenames and read analyzed data
        datafiles = list(savepath.glob("*.h5"))
        if not datafiles:
            raise ValueError(
                "Cannot find predicted points. Some wrong with DeepLabCut?"
            )
        cam1data = pd.read_hdf(savepath / datafiles[0])
        cam2data = pd.read_hdf(savepath / datafiles[1])
        dlc_to_xma(cam1data, cam2data, trial, savepath)


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
