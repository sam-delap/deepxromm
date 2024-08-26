"""Converts data from XMALab into the format useful for DLC training."""

import glob
import os

from subprocess import Popen

import blend_modes
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from ruamel.yaml import YAML


class XMADataProcessor:
    """Converts data from XMALab into the format useful for DLC training."""

    def __init__(self, config):
        self._config = config
        self._swap_markers = config["swapped_markers"]
        self._cross_markers = config["crossed_markers"]

    def get_bodyparts_from_xma(self, path_to_trial, mode):
        """Pulls the names of the XMAlab markers from the 2Dpoints file"""

        csv_path = [file for file in os.listdir(path_to_trial) if file[-4:] == ".csv"]
        if len(csv_path) > 1:
            raise FileExistsError(
                "Found more than 1 CSV file for trial: " + path_to_trial
            )
        if len(csv_path) <= 0:
            raise FileNotFoundError(
                "Couldn't find a CSV file for trial: " + path_to_trial
            )
        trial_csv = pd.read_csv(
            os.path.join(path_to_trial, csv_path[0]),
            sep=",",
            header=0,
            dtype="float",
            na_values="NaN",
        )
        names = trial_csv.columns.values
        if mode == "rgb":
            parts = [name.rsplit("_", 1)[0] for name in names]
            if self._swap_markers:
                parts = parts + [f"sw_{part}" for part in parts]
            if self._cross_markers:
                parts = parts + [
                    f"cx_{part}_cam1x2"
                    for part in [name.rsplit("_", 2)[0] for name in names]
                ]
        elif mode in ["2D", "per_cam"]:
            parts = [name.rsplit("_", 2)[0] for name in names]
        else:
            raise SyntaxError("Invalid value for mode parameter")

        parts_unique = []
        for part in parts:
            if part not in parts_unique:
                parts_unique.append(part)
        return parts_unique

    def make_rgb_video(self, data_path):
        """For all trials in given data path merges 2 videos into single RBG video."""
        trials = [
            folder
            for folder in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, folder))
            and not folder.startswith(".")
        ]
        for trial in trials:
            path_to_trial = f"{data_path}/{trial}"
            self._merge_rgb(path_to_trial)
            try:
                trial_name = os.path.basename(os.path.normpath(path_to_trial))
                df = pd.read_csv(f"{path_to_trial}/{trial_name}.csv")
            except FileNotFoundError as e:
                print(f"Please make sure that your trainingdata 2DPoints csv file is named {trial_name}.csv")
                raise e
            substitute_data_relpath = "labeled-data/" + self._config["dataset_name"]
            substitute_data_abspath = os.path.join(
                os.path.split(self._config["path_config_file"])[0],
                substitute_data_relpath,
            )
            df = df.dropna(how="all")
            list_of_frames = df.index + 1
            # TODO: decouple the behavior for videos vs data points/image
            # extraction. extract_matched_frames works for trials with an
            # XMAlab-formatted CSV attached (typically training data), while merge_rgb
            # works on all videos as long as there's a cam 1 and a cam2 video.
            self._extract_matched_frames_rgb(
                path_to_trial,
                substitute_data_abspath,
                list_of_frames,
            )
            self._splice_xma_to_dlc(path_to_trial)

    def find_cam_file(self, path, identifier):
        """Searches a file for a given cam video in the trail folder."""
        files = glob.glob(os.path.join(path, f"*{identifier}*.avi"))
        if files:
            result = files[0]
            print(f"Found file {result} for {identifier}")
            return files[0]
        else:
            raise FileNotFoundError(f"No video file found containing '{identifier}' in {path}")

    def list_trials(self):
        path = os.path.join(self._config["working_dir"], "trials")
        """Returns a list of trials or throws an exception if folder is empty"""
        trials = [folder for folder in os.listdir(path) if os.path.isdir(os.path.join(path, folder)) and not folder.startswith('.')]

        if len(trials) <= 0:
            raise FileNotFoundError(f'Empty trials directory found. Please put trials to be '
                                     'analyzed after training into the {path} folder')
        return trials

    def split_rgb(self, trial_path, codec='avc1'):
        '''Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos.'''
        trial_name = os.path.basename(os.path.normpath(trial_path))
        out_name = trial_name + '_split_'

        try:
            rgb_video = cv2.VideoCapture(f'{trial_path}/{trial_name}_rgb.avi')
        except FileNotFoundError as e:
            print(f'Couldn\'t find video at {trial_path}/{trial_name}_rgb.avi')
            raise e
        frame_width = int(rgb_video.get(3))
        frame_height = int(rgb_video.get(4))
        frame_rate = round(rgb_video.get(5),2)
        if codec == 'uncompressed':
            pix_format = 'gray'   ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
            cam1_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
            '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'cam1.avi'], stdin=PIPE)
            cam2_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
            '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'cam2.avi'], stdin=PIPE)
            blue_split_ffmpeg = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-r', str(int(frame_rate)),
            '-i', '-', '-vcodec', 'rawvideo','-pix_fmt',pix_format,'-r', str(int(frame_rate)), f'{trial_path}/{out_name}'+'blue.avi'], stdin=PIPE)
        else:
            if codec == 0:
                fourcc = 0
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            cam1 = cv2.VideoWriter(f'{trial_path}/{out_name}'+'cam1.avi',
                                    fourcc,
                                    frame_rate,(frame_width, frame_height))
            cam2 = cv2.VideoWriter(f'{trial_path}/{out_name}'+'cam2.avi',
                                    fourcc,
                                    frame_rate,(frame_width, frame_height))
            blue_channel = cv2.VideoWriter(f'{trial_path}/{out_name}'+'blue.avi',
                                    fourcc,
                                    frame_rate,(frame_width, frame_height))

        i = 1
        while rgb_video.isOpened():
            ret, frame = rgb_video.read()
            print(f'Current Frame: {i}')
            i = i + 1
            if ret:
                B, G, R = cv2.split(frame)
                if codec == 'uncompressed':
                    im_r = Image.fromarray(R)
                    im_g = Image.fromarray(G)
                    im_b = Image.fromarray(B)
                    im_r.save(cam1_split_ffmpeg.stdin, 'PNG')
                    im_g.save(cam2_split_ffmpeg.stdin, 'PNG')
                    im_b.save(blue_split_ffmpeg.stdin, 'PNG')
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    cam1.write(R)
                    cam2.write(G)
                    blue_channel.write(B)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        if codec == 'uncompressed':
            cam1_split_ffmpeg.stdin.close()
            cam1_split_ffmpeg.wait()
            cam2_split_ffmpeg.stdin.close()
            cam2_split_ffmpeg.wait()
            blue_split_ffmpeg.stdin.close()
            blue_split_ffmpeg.wait()
        else:
            cam1.release()
            cam2.release()
            blue_channel.release()
        rgb_video.release()
        cv2.destroyAllWindows()
        print(f"Cam1 grayscale video created at {trial_path}/{out_name}cam1.avi!")
        print(f"Cam2 grayscale video created at {trial_path}/{out_name}cam2.avi!")
        print(f"Blue channel grayscale video created at {trial_path}/{out_name}blue.avi!")

    def _merge_rgb(self, trial_path, codec="avc1", mode="difference"):
        """
        Takes the path to a trial subfolder and exports a single new video with
        cam1 video written to the red channel and cam2 video written to the
        green channel. The blue channel is, depending on the value of config
        "mode", either the difference blend between A and B, the multiply
        blend, or just a black frame.
        """
        trial_name = os.path.basename(os.path.normpath(trial_path))
        if os.path.exists(f"{trial_path}/{trial_name}_rgb.avi"):
            print("RGB video already created. Skipping.")
            return
        cam1_video_path = self.find_cam_file(trial_path, "cam1")
        cam1_video = cv2.VideoCapture(cam1_video_path)

        cam2_video_path = self.find_cam_file(trial_path, "cam2")
        cam2_video = cv2.VideoCapture(cam2_video_path)

        frame_width = int(cam1_video.get(3))
        frame_height = int(cam1_video.get(4))
        frame_rate = round(cam1_video.get(5), 2)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(
            f"{trial_path}/{trial_name}_rgb.avi",
            fourcc,
            frame_rate,
            (frame_width, frame_height),
        )
        i = 1
        while cam1_video.isOpened():
            print(f"Current Frame: {i}")
            ret_cam1, frame_cam1 = cam1_video.read()
            _, frame_cam2 = cam2_video.read()
            if ret_cam1:
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2BGRA, 4).astype(
                    np.float32
                )
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2BGRA, 4).astype(
                    np.float32
                )
                frame_cam1 = cv2.normalize(
                    frame_cam1, None, 0, 255, norm_type=cv2.NORM_MINMAX
                )
                frame_cam2 = cv2.normalize(
                    frame_cam2, None, 0, 255, norm_type=cv2.NORM_MINMAX
                )
                if mode == "difference":
                    extra_channel = blend_modes.difference(frame_cam1, frame_cam2, 1)
                elif mode == "multiply":
                    extra_channel = blend_modes.multiply(frame_cam1, frame_cam2, 1)
                else:
                    extra_channel = np.zeros((frame_width, frame_height, 3), np.uint8)
                    extra_channel = cv2.cvtColor(
                        extra_channel, cv2.COLOR_BGR2BGRA, 4
                    ).astype(np.float32)
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGRA2BGR).astype(
                    np.uint8
                )
                frame_cam1 = cv2.cvtColor(frame_cam1, cv2.COLOR_BGR2GRAY)
                frame_cam2 = cv2.cvtColor(frame_cam2, cv2.COLOR_BGR2GRAY)
                extra_channel = cv2.cvtColor(extra_channel, cv2.COLOR_BGR2GRAY)
                merged = cv2.merge((extra_channel, frame_cam2, frame_cam1))
                out.write(merged)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

            i = i + 1
        cam1_video.release()
        cam2_video.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Merged RGB video created at {trial_path}/{trial_name}_rgb.avi!")

    def _splice_xma_to_dlc(self, trial_path, outlier_mode=False):
        """Takes csv of XMALab 2D XY coordinates from 2 cameras, outputs spliced hdf+csv data for DeepLabCut"""
        substitute_data_relpath = "labeled-data/" + self._config["dataset_name"]
        substitute_data_abspath = os.path.join(
            os.path.sep.join(self._config["path_config_file"].split("\\")[:-1]),
            substitute_data_relpath,
        )
        markers = self.get_bodyparts_from_xma(trial_path, mode='2D')
        # TODO: this entire section can be solved with a creative call to
        # get_bodyparts_from_xma and some dataFrame manipulation to be
        # significantly shorter (and potentially faster?)
        try:
            trial_name = os.path.basename(os.path.normpath(trial_path))
            df = pd.read_csv(f"{trial_path}/{trial_name}.csv")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Please make sure that your trainingdata 2DPoints csv file is named {trial_name}.csv"
            ) from e
        if self._swap_markers:
            print("Creating cam1Y-cam2Y-swapped synthetic markers")
            swaps = []
            df_sw = pd.DataFrame()
            for marker in markers:
                name_x1 = marker + "_cam1_X"
                name_x2 = marker + "_cam2_X"
                name_y1 = marker + "_cam1_Y"
                name_y2 = marker + "_cam2_Y"
                swap_name_x1 = "sw_" + name_x1
                swap_name_x2 = "sw_" + name_x2
                swap_name_y1 = "sw_" + name_y1
                swap_name_y2 = "sw_" + name_y2
                df_sw[swap_name_x1] = df[name_x1]
                df_sw[swap_name_y1] = df[name_y2]
                df_sw[swap_name_x2] = df[name_x2]
                df_sw[swap_name_y2] = df[name_y1]
                swaps.extend([swap_name_x1, swap_name_y1, swap_name_x2, swap_name_y2])
            df = df.join(df_sw)
            print(swaps)
        if self._cross_markers:
            print("Creating cam1-cam2-crossed synthetic markers")
            crosses = []
            df_cx = pd.DataFrame()
            for marker in markers:
                name_x1 = marker + "_cam1_X"
                name_x2 = marker + "_cam2_X"
                name_y1 = marker + "_cam1_Y"
                name_y2 = marker + "_cam2_Y"
                cross_name_x = "cx_" + marker + "_cam1x2_X"
                cross_name_y = "cx_" + marker + "_cam1x2_Y"
                df_cx[cross_name_x] = df[name_x1] * df[name_x2]
                df_cx[cross_name_y] = df[name_y1] * df[name_y2]
                crosses.extend([cross_name_x, cross_name_y])
            df = df.join(df_cx)
            print(crosses)
        names_final = df.columns.values
        parts_final = [name.rsplit("_", 1)[0] for name in names_final]
        parts_unique_final = []
        for part in parts_final:
            if not part in parts_unique_final:
                parts_unique_final.append(part)
        print("Importing markers: ")
        print(parts_unique_final)
        with open(self._config["path_config_file"], "r") as dlc_config:
            yaml = YAML()
            dlc_proj = yaml.load(dlc_config)

        dlc_proj["bodyparts"] = parts_unique_final

        with open(self._config["path_config_file"], "w") as dlc_config:
            yaml.dump(dlc_proj, dlc_config)

        df = df.dropna(how="all")
        list_of_frames = df.index + 1
        unique_frames_set = set(list_of_frames)
        unique_frames = sorted(unique_frames_set)
        print("Importing frames: ")
        print(unique_frames)
        df["frame_index"] = [
            substitute_data_relpath
            + f"/{trial_name}_rgb_"
            + str(index).zfill(4)
            + ".png"
            for index in unique_frames
        ]
        df["scorer"] = self._config["experimenter"]
        df = df.melt(id_vars=["frame_index", "scorer"])
        new = df["variable"].str.rsplit("_", n=1, expand=True)
        df["variable"], df["coords"] = new[0], new[1]
        df = df.rename(columns={"variable": "bodyparts"})
        df["coords"] = df["coords"].str.rstrip(" ").str.lower()
        cat_type = pd.api.types.CategoricalDtype(
            categories=parts_unique_final, ordered=True
        )
        df["bodyparts"] = df["bodyparts"].str.lstrip(" ").astype(cat_type)
        newdf = df.pivot_table(
            columns=["scorer", "bodyparts", "coords"],
            index="frame_index",
            values="value",
            aggfunc="first",
            dropna=False,
        )
        newdf.index.name = None
        if not os.path.exists(substitute_data_abspath):
            os.makedirs(substitute_data_abspath)
        if outlier_mode:
            data_name = os.path.join(substitute_data_abspath, "MachineLabelsRefine.h5")
            tracked_hdf = os.path.join(
                substitute_data_abspath, ("MachineLabelsRefine_" + ".h5")
            )
        else:
            data_name = os.path.join(
                substitute_data_abspath,
                ("CollectedData_" + self._config["experimenter"] + ".h5"),
            )
            tracked_hdf = os.path.join(
                substitute_data_abspath,
                ("CollectedData_" + self._config["experimenter"] + ".h5"),
            )
        newdf.to_hdf(data_name, "df_with_missing", format="table", mode="w")
        newdf.to_hdf(tracked_hdf, "df_with_missing", format="table", mode="w")
        tracked_csv = data_name.split(".h5")[0] + ".csv"
        newdf.to_csv(tracked_csv, na_rep="NaN")
        print(
            "Successfully spliced XMALab 2D points to DLC format",
            "saved " + str(data_name),
            "saved " + str(tracked_hdf),
            "saved " + str(tracked_csv),
            sep="\n",
        )

    def _extract_matched_frames_rgb(
        self, trial_path, labeled_data_path, indices, compression=1
    ):
        """Given a list of frame indices and a project path, produce a folder (in labeled-data) of matching frame pngs per source video.
        Optionally, compress the output PNGs. Factor ranges from 0 (no compression) to 9 (most compression)"""
        extracted_frames = []
        trainingdata_path = self._config["working_dir"] + "/trainingdata"
        trial_name = os.path.basename(os.path.normpath(trial_path))
        video_path = f"{trainingdata_path}/{trial_name}/{trial_name}_rgb.avi"
        labeled_data_path = (
            os.path.split(self._config["path_config_file"])[0]
            + "/labeled-data/"
            + self._config["dataset_name"]
        )
        frames_from_vid = self._vid_to_pngs(
            video_path,
            indices,
            labeled_data_path,
            name_from_folder=True,
            compression=compression,
        )
        extracted_frames.append(frames_from_vid)
        print("Extracted " + str(len(indices)) + f" matching frames from {video_path}")

    def _vid_to_pngs(
        self,
        video_path,
        indices_to_match,
        output_dir=None,
        name_from_folder=True,
        compression=0,
    ):
        """Takes a list of frame numbers and exports matching frames from a video as pngs.
        Optionally, compress the output PNGs. Factor ranges from 0 (no compression) to 9 (most compression)"""
        frame_index = 1
        last_frame_to_analyze = max(indices_to_match)
        png_list = []
        if name_from_folder:
            out_name = os.path.splitext(os.path.basename(video_path))[0]
        else:
            out_name = "img"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print("===== try this")
        cap = cv2.VideoCapture(video_path)
        print("===== end this")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is False:
                break
            if frame_index > last_frame_to_analyze:
                break
            if frame_index not in indices_to_match:
                frame_index += 1
                continue

            print(f"Extracting frame {frame_index}")
            png_name = out_name + "_" + str(frame_index).zfill(4) + ".png"
            png_path = os.path.join(output_dir, png_name)
            png_list.append(png_path)
            cv2.imwrite(png_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, compression])
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()
        return png_list
