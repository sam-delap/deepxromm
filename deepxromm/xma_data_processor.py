"""
Standardized data format conversion for deepxromm projects
"""

from pathlib import Path

from subprocess import Popen, PIPE

import cv2
from PIL import Image
import numpy as np
import pandas as pd
import random
import yaml

from deepxromm.xrommtools import dlc_to_xma, get_marker_names, get_marker_and_cam_names
from deepxromm.logging import logger
from deepxromm.trial import Trial
from deepxromm.config_utilities import load_config_file, save_config_file
from deepxromm.dlc_config import DlcConfig


class XMADataProcessor:
    """Converts data from XMALab into the format useful for DLC training."""

    def __init__(self, project, dlc_config: DlcConfig):
        self.project = project
        self.dlc_config = dlc_config
        self.mode = dlc_config.mode
        self.path_config_file = dlc_config.path_config_file
        self.working_dir = project.working_dir
        self.nframes = project.nframes
        self.dataset_name = project.dataset_name
        self.experimenter = project.experimenter
        self.video_codec = project.video_codec
        if dlc_config.mode == "per_cam":
            self.path_config_file_2 = dlc_config.path_config_file_2
        # TODO: Move this into project functionality, or somewhere that doesn't require referencing this way
        self._swap_markers = False
        self._cross_markers = False
        if self.mode == "rgb":
            self._swap_markers = dlc_config.swapped_markers
            self._cross_markers = dlc_config.crossed_markers

    def dlc_to_xma(self):
        """Convert DLC-formatted training output into XMAlab-formatted data"""
        mode = self.mode
        trials = self.project.list_trials()

        if mode in ["2D", "per_cam"]:
            correct_function_signature = dlc_to_xma
        elif mode == "rgb":
            correct_function_signature = self._split_dlc_to_xma
        else:
            raise AttributeError(f"Unsupported mode: {mode}")

        with self.path_config_file.open("r") as dlc_config:
            dlc_proj = yaml.safe_load(dlc_config)

        iteration = int(dlc_proj["iteration"])
        for trial in trials:
            correct_function_signature(trial, iteration)

    def _split_dlc_to_xma(self, trial_path: Path, iteration: int, save_hdf=True):
        """Takes the output from RGB deeplabcut and splits it into XMAlab-readable output"""
        trial = Trial(trial_path)
        current_files = trial_path.glob("*")
        logger.debug(f"Current files in directory {current_files}")
        # Use the CSV from the user's training data to fetch bodyparts
        sample_trial = Trial(self.project.list_trials("trainingdata")[0])
        trial_csv_path = sample_trial.find_trial_csv()
        bodyparts_xy = get_marker_names(trial_csv_path)
        rgb_parts = get_marker_and_cam_names(trial_csv_path)

        iteration_folder_name = f"it{iteration}"
        # Assumes that the project itself doesn't have rgbDLC in this format in it
        csv_path = trial.find_trial_csv(iteration_folder_name, "rgbDLC")
        xma_csv_path = (
            trial.trial_path
            / iteration_folder_name
            / f"{trial.trial_name}-Predicted2DPoints.csv"
        )

        df = pd.read_csv(csv_path, skiprows=1)
        df.index = df["bodyparts"]
        df = df.drop(columns=df.columns[df.loc["coords"] == "likelihood"])
        df = df.drop(
            columns=[
                column
                for column in df.columns
                if column not in rgb_parts
                and column not in [f"{bodypart}.1" for bodypart in rgb_parts]
            ]
        )
        df.columns = bodyparts_xy
        df = df.drop(index="coords")
        df.to_csv(xma_csv_path, index=False)
        logger.info(
            "Successfully split DLC format to XMALab 2D points; saved "
            + str(xma_csv_path)
        )
        if save_hdf:
            tracked_hdf = xma_csv_path.with_suffix(".h5")
            logger.debug(f"Tracked hdf stored at {str(tracked_hdf)}")
            df.to_hdf(
                tracked_hdf, "df_with_missing", format="table", mode="w", nan_rep="NaN"
            )

    def find_trial_csv(self, trial_path: Path, identifier: str | None = None) -> Path:
        """
        Takes the path to a trial and returns the path to a trial CSV.
        Errors if there is not exactly 1 trial CSV in a trial folder.
        """
        if identifier is not None:
            csv_path = list(trial_path.glob(f"*{identifier}*.csv"))
        else:
            csv_path = list(trial_path.glob("*.csv"))

        if len(csv_path) > 1:
            logger.error(csv_path)
            raise FileExistsError(
                f"Found more than 1 CSV file with identifier {identifier} for trial: {trial_path}"
            )
        if len(csv_path) <= 0:
            logger.debug(f"Current files in {str(trial_path)}")
            logger.debug(list(trial_path.glob("*")))
            raise FileNotFoundError(
                f"Couldn't find a CSV file with identifier {identifier} for trial: {trial_path}"
            )

        return csv_path[0]

    def xma_to_dlc_rgb(self, suffix: str, picked_frames: list[list[int]]):
        """Convert XMAlab input into RGB-ready DLC input"""
        trials = self.project.list_trials(suffix=suffix)
        for idx, trial_path in enumerate(trials):
            list_of_frames = picked_frames[idx]
            substitute_data_relpath = Path("labeled-data") / self.dataset_name
            substitute_data_abspath = self.path_config_file / substitute_data_relpath
            self._extract_matched_frames_rgb(
                trial_path,
                substitute_data_abspath,
                list_of_frames,
            )
            self._splice_xma_to_dlc(trial_path, list_of_frames)

    def find_cam_file(self, path: Path, identifier: str):
        """Searches a file for a given cam video in the trail folder."""
        files = list(path.glob(f"*{identifier}*.avi"))
        logger.debug(files)
        if files:
            result = files[0]
            logger.debug(f"Found file {result} for {identifier}")
            return files[0]
        else:
            raise FileNotFoundError(
                f"No video file found containing '{identifier}' in {path}"
            )

    def split_rgb(self, trial_path: Path, codec=None):
        """Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos."""
        # Use provided codec, otherwise fall back to config value
        if codec is None:
            codec = self.video_codec

        trial_name = trial_path.name
        out_name = trial_name + "_split_"

        rgb_video_path = trial_path / f"{trial_name}_rgb.avi"
        try:
            rgb_video = cv2.VideoCapture(rgb_video_path)
        except FileNotFoundError as e:
            logger.error(f"Couldn't find video at {rgb_video_path}")
            raise e
        frame_width = int(rgb_video.get(3))
        frame_height = int(rgb_video.get(4))
        frame_rate = round(rgb_video.get(5), 2)

        if codec == "uncompressed":
            pix_format = "gray"  ##change to 'yuv420p' for color or 'gray' for grayscale. 'pal8' doesn't play on macs
            cam1_split_ffmpeg = Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-r",
                    str(int(frame_rate)),
                    "-i",
                    "-",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    pix_format,
                    "-r",
                    str(int(frame_rate)),
                    f"{trial_path}/{out_name}" + "cam1.avi",
                ],
                stdin=PIPE,
            )
            cam2_split_ffmpeg = Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-r",
                    str(int(frame_rate)),
                    "-i",
                    "-",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    pix_format,
                    "-r",
                    str(int(frame_rate)),
                    f"{trial_path}/{out_name}" + "cam2.avi",
                ],
                stdin=PIPE,
            )
            blue_split_ffmpeg = Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "png",
                    "-r",
                    str(int(frame_rate)),
                    "-i",
                    "-",
                    "-vcodec",
                    "rawvideo",
                    "-pix_fmt",
                    pix_format,
                    "-r",
                    str(int(frame_rate)),
                    f"{trial_path}/{out_name}" + "blue.avi",
                ],
                stdin=PIPE,
            )
        else:
            if codec == 0:
                fourcc = 0
            else:
                fourcc = cv2.VideoWriter_fourcc(*codec)
            cam1 = cv2.VideoWriter(
                f"{trial_path}/{out_name}" + "cam1.avi",
                fourcc,
                frame_rate,
                (frame_width, frame_height),
            )
            cam2 = cv2.VideoWriter(
                f"{trial_path}/{out_name}" + "cam2.avi",
                fourcc,
                frame_rate,
                (frame_width, frame_height),
            )
            blue_channel = cv2.VideoWriter(
                f"{trial_path}/{out_name}" + "blue.avi",
                fourcc,
                frame_rate,
                (frame_width, frame_height),
            )

            # Verify VideoWriters opened successfully
            if not cam1.isOpened():
                raise RuntimeError(
                    f"Failed to create cam1 video writer with codec '{codec}'"
                )
            if not cam2.isOpened():
                raise RuntimeError(
                    f"Failed to create cam2 video writer with codec '{codec}'"
                )
            if not blue_channel.isOpened():
                raise RuntimeError(
                    f"Failed to create blue channel video writer with codec '{codec}'"
                )

        i = 1
        while rgb_video.isOpened():
            ret, frame = rgb_video.read()
            if i == 1 or i % 50 == 0:
                logger.info(f"Current Frame: {i}")
            i = i + 1
            if ret:
                B, G, R = cv2.split(frame)
                if codec == "uncompressed":
                    im_r = Image.fromarray(R)
                    im_g = Image.fromarray(G)
                    im_b = Image.fromarray(B)
                    # TODO - add error handling/assertions
                    im_r.save(cam1_split_ffmpeg.stdin, "PNG")
                    im_g.save(cam2_split_ffmpeg.stdin, "PNG")
                    im_b.save(blue_split_ffmpeg.stdin, "PNG")
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    cam1.write(R)
                    cam2.write(G)
                    blue_channel.write(B)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            else:
                break
        if codec == "uncompressed":
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
        logger.info(f"Cam1 grayscale video created at {trial_path}/{out_name}cam1.avi!")
        logger.info(f"Cam2 grayscale video created at {trial_path}/{out_name}cam2.avi!")
        logger.info(
            f"Blue channel grayscale video created at {trial_path}/{out_name}blue.avi!"
        )

    def _splice_xma_to_dlc(self, trial_path: Path, list_of_frames: list[int]):
        """Takes csv of XMALab 2D XY coordinates from 2 cameras, outputs spliced hdf+csv data for DeepLabCut"""
        dlc_path = self.path_config_file.parent
        trial = Trial(trial_path)
        substitute_data_abspath = dlc_path / "labeled-data" / self.dataset_name
        trial_csv_path = trial.find_trial_csv()
        markers = get_marker_names(trial_csv_path)

        # Add salt to the training data, if desired
        df = pd.read_csv(trial_csv_path)
        if self._swap_markers:
            logger.info("Creating cam1Y-cam2Y-swapped synthetic markers")
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
            logger.debug(swaps)
        if self._cross_markers:
            logger.info("Creating cam1-cam2-crossed synthetic markers")
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
            logger.debug(crosses)
        parts_unique_final = self.dlc_config.get_bodyparts(trial_csv_path)
        logger.debug("Importing markers: ")
        logger.debug(parts_unique_final)
        dlc_proj = load_config_file(self.path_config_file)
        dlc_proj["bodyparts"] = parts_unique_final
        save_config_file(dlc_proj, self.dlc_config.path_config_file)

        # Ensure that all frames were unique originally
        unique_frames_set = set(list_of_frames)
        assert len(list_of_frames) == len(unique_frames_set)
        unique_frames = sorted(unique_frames_set)
        logger.debug("Importing frames: ")
        logger.debug(unique_frames)

        # Cut data down to just the frames that we've picked
        df = df.loc[unique_frames, :]
        df["frame_index"] = [
            str(
                (
                    substitute_data_abspath
                    / f"{trial.trial_name}_rgb_{str(index + 1).zfill(4)}.png"
                ).relative_to(dlc_path)
            )
            for index in unique_frames
        ]
        df["scorer"] = self.experimenter
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
        substitute_data_abspath.mkdir(parents=True, exist_ok=True)
        data_name = "CollectedData_" + self.experimenter + ".h5"
        tracked_hdf = substitute_data_abspath / data_name

        newdf.to_hdf(tracked_hdf, "df_with_missing", format="table", mode="w")
        tracked_csv = tracked_hdf.with_suffix(".csv")
        newdf.to_csv(tracked_csv, na_rep="NaN")
        logger.info("Successfully spliced XMALab 2D points to DLC format")
        logger.info(f"HDF saved to: {tracked_hdf}")
        logger.info(f"CSV saved to: {tracked_csv}")

    def _extract_matched_frames_rgb(
        self, trial_path, labeled_data_path, indices, compression=1
    ):
        """Given a list of frame indices and a project path, produce a folder (in labeled-data) of matching frame pngs per source video.
        Optionally, compress the output PNGs. Factor ranges from 0 (no compression) to 9 (most compression)
        """
        extracted_frames = []
        trainingdata_path = self.working_dir / "trainingdata"
        trial_name = trial_path.name
        video_path = trainingdata_path / trial_name / f"{trial_name}_rgb.avi"
        dlc_path = self.path_config_file.parent
        labeled_data_path = dlc_path / "labeled-data" / self.dataset_name
        if len(indices) < int(self.nframes):
            raise ValueError("nframes is bigger than number of detected frames")
        frames_from_vid = self.extract_frames_from_video(
            source_path=video_path,
            frame_indices=indices,
            output_dir=labeled_data_path,
            output_name_base=video_path.parent.name,  # Gets trial name from filename
            mode="rgb",
            compression=compression,
        )
        extracted_frames.append(frames_from_vid)
        logger.info(f"Extracted {len(indices)} matching frames from {video_path}")

    def _extract_frame_selection_loop(self, idx: list, nframes: int):
        """Extract the existing frame selection algorithm (preserve existing algorithm)

        Args:
            idx: List of frame indices for each trial
            nframes: Number of frames to select

        Returns:
            List of selected frame indices for each trial
        """
        picked_frames = []

        # Check if we have enough frames
        if sum(len(x) for x in idx) < nframes:
            raise ValueError("nframes is bigger than number of detected frames")

        # Pick frames to extract (NOTE this is random currently)
        # current code iteratively picks one frame at a time from each shuffled trial until # of picked_frames hits nframes
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

    def extract_frames_from_video(
        self,
        source_path: Path,
        frame_indices: list[int],
        output_dir: Path,
        output_name_base: str,
        mode: str = "2D",
        camera: int | None = None,
        compression: int = 0,
    ) -> list[str]:
        """
        Extract specified frames from video or image folder.

        Unified interface for all frame extraction needs across different modes.

        Args:
            source_path: Path to video file or image directory
            frame_indices: List of frame numbers to extract (0-indexed)
            output_dir: Directory to save extracted frames
            output_name_base: Base name for output files (e.g., trial name)
            mode: Extraction mode - determines filename format
                  "rgb": {output_name_base}_rgb_{frame}.png
                  "2D": {output_name_base}_cam{camera}_{frame}.png
                  "per_cam": {output_name_base}_{frame}.png
            camera: Camera number (required for 2D mode)
            compression: PNG compression level (0=none, 9=max)

        Returns:
            List of absolute file paths to extracted frames (as strings)

        Raises:
            ValueError: If mode is invalid or required parameters are missing
            FileNotFoundError: If source_path doesn't exist
        """
        # Validate mode
        valid_modes = ["2D", "per_cam", "rgb"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Validate camera parameter for 2D mode
        if mode == "2D" and camera is None:
            raise ValueError("Camera parameter is required for 2D mode")

        # Validate compression level
        if not (0 <= compression <= 9):
            raise ValueError(f"Compression must be between 0 and 9, got {compression}")

        # Validate source exists
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")

        # Create output directory if needed
        output_dir.mkdir(parents=True, exist_ok=True)

        # Route to appropriate handler based on source type
        if source_path.is_dir():
            return self._extract_from_image_folder(
                source_path,
                frame_indices,
                output_dir,
                output_name_base,
                mode,
                camera,
            )
        else:
            return self._extract_from_video_file(
                source_path,
                frame_indices,
                output_dir,
                output_name_base,
                mode,
                camera,
                compression,
            )

    def _extract_from_video_file(
        self,
        video_path: Path,
        frame_indices: list[int],
        output_dir: Path,
        output_name_base: str,
        mode: str,
        camera: int | None,
        compression: int,
    ) -> list[str]:
        """
        Extract frames from video file.

        Args:
            video_path: Path to video file
            frame_indices: List of 0-indexed frame numbers to extract
            output_dir: Directory to save frames
            output_name_base: Base name for output files
            mode: Extraction mode ("2D", "per_cam", or "rgb")
            camera: Camera number (for 2D mode)
            compression: PNG compression level (0-9)

        Returns:
            List of absolute file paths as strings
        """
        frame_indices_set = set(frame_indices)
        last_frame_to_analyze = max(frame_indices)
        extracted_paths = []

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index > last_frame_to_analyze:
                break

            if frame_index in frame_indices_set:
                # Print progress with human-readable (1-indexed) frame number
                if frame_index % 50 == 0:
                    logger.info(f"Extracting frame {frame_index + 1}")

                # Build filename based on mode (use 1-based indexing for file names)
                frame_str = str(frame_index + 1).zfill(4)
                if mode == "rgb":
                    filename = f"{output_name_base}_rgb_{frame_str}.png"
                elif mode == "2D":
                    filename = f"{output_name_base}_cam{camera}_{frame_str}.png"
                else:  # per_cam
                    filename = f"{output_name_base}_{frame_str}.png"

                output_path = output_dir / filename
                cv2.imwrite(
                    str(output_path), frame, [cv2.IMWRITE_PNG_COMPRESSION, compression]
                )
                extracted_paths.append(str(output_path.absolute()))

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        return extracted_paths

    def _extract_from_image_folder(
        self,
        img_folder: Path,
        frame_indices: list[int],
        output_dir: Path,
        output_name_base: str,
        mode: str,
        camera: int | None,
    ) -> list[str]:
        """
        Extract frames from image folder.

        Args:
            img_folder: Path to folder containing images
            frame_indices: List of 0-indexed frame numbers to extract
            output_dir: Directory to save frames
            output_name_base: Base name for output files
            mode: Extraction mode ("2D", "per_cam", or "rgb")
            camera: Camera number (for 2D mode)

        Returns:
            List of absolute file paths as strings
        """
        extracted_paths = []
        imgs = sorted(list(img_folder.glob("*")))
        frame_indices_sorted = sorted(frame_indices)

        for frame_idx in frame_indices_sorted:
            if frame_idx >= len(imgs):
                logger.warning(
                    f"Frame index {frame_idx} out of range (folder has {len(imgs)} images)"
                )
                continue

            # Print progress with human-readable (1-indexed) frame number
            if frame_idx % 50 == 0:
                logger.info(f"Extracting frame {frame_idx + 1}")

            # Load image
            img_path = imgs[frame_idx]
            image = cv2.imread(str(img_path))

            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Build filename based on mode
            # Note: For image folders, we use frame_idx + 1 for human readability
            frame_str = str(frame_idx + 1).zfill(4)
            if mode == "rgb":
                filename = f"{output_name_base}_rgb_{frame_str}.png"
            elif mode == "2D":
                filename = f"{output_name_base}_cam{camera}_{frame_str}.png"
            else:  # per_cam
                filename = f"{output_name_base}_{frame_str}.png"

            output_path = output_dir / filename
            cv2.imwrite(str(output_path), image)
            extracted_paths.append(str(output_path.absolute()))

        return extracted_paths

    def build_dlc_dataframe(
        self,
        data: pd.DataFrame,
        scorer: str,
        relnames: list[str],
        pointnames: list[str],
    ) -> pd.DataFrame:
        """
        Build DeepLabCut MultiIndex DataFrame from raw coordinate data.

        Args:
            data: DataFrame with coordinates (columns are alternating x,y for each point)
            scorer: Name of the scorer/experimenter
            relnames: List of absolute image file paths (for index)
            pointnames: List of body part names

        Returns:
            DataFrame with MultiIndex columns (scorer, bodyparts, coords)

        Raises:
            ValueError: If data dimensions don't match expected structure
        """
        # Validate inputs
        expected_cols = len(pointnames) * 2  # x,y for each point
        if data.shape[1] != expected_cols:
            raise ValueError(
                f"Data has {data.shape[1]} columns but expected {expected_cols} "
                f"for {len(pointnames)} body parts (2 coords each)"
            )

        if data.shape[0] != len(relnames):
            raise ValueError(
                f"Data has {data.shape[0]} rows but {len(relnames)} image names provided"
            )

        # Build MultiIndex DataFrame
        dataFrame = pd.DataFrame()
        temp = np.empty((data.shape[0], 2))
        temp[:] = np.nan

        for idx, relname in enumerate(relnames):
            if "_cam2" in relname and self.mode == "per_cam":
                relnames[idx] = str(
                    Path(relname).relative_to(self.path_config_file_2.parent)
                )
            else:
                relnames[idx] = str(
                    Path(relname).relative_to(self.path_config_file.parent)
                )

        for i, bodypart in enumerate(pointnames):
            index = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"],
            )
            frame = pd.DataFrame(temp, columns=index, index=relnames)
            frame.iloc[:, 0:2] = data.iloc[:, 2 * i : 2 * i + 2].values.astype(float)
            dataFrame = pd.concat([dataFrame, frame], axis=1)

        # Clean various NaN representations
        dataFrame.replace("", np.nan, inplace=True)
        dataFrame.replace(" NaN", np.nan, inplace=True)
        dataFrame.replace(" NaN ", np.nan, inplace=True)
        dataFrame.replace("NaN ", np.nan, inplace=True)
        dataFrame = dataFrame.apply(pd.to_numeric, errors="coerce")

        return dataFrame

    def save_dlc_dataset(
        self,
        data: pd.DataFrame,
        scorer: str,
        relnames: list[str],
        pointnames: list[str],
        output_dir: Path,
    ) -> None:
        """
        Create complete DeepLabCut dataset with DataFrame and save files.

        Builds the multi-index DataFrame structure and saves to both
        HDF5 and CSV formats expected by DeepLabCut.

        Args:
            data: DataFrame with raw coordinate data
            scorer: Name of scorer/experimenter
            relnames: List of absolute image file paths
            pointnames: List of body part names
            output_dir: Directory to save output files
        """
        # Build the DataFrame
        dataFrame = self.build_dlc_dataframe(data, scorer, relnames, pointnames)

        # Save files
        h5_save_path = output_dir / f"CollectedData_{scorer}.h5"
        csv_save_path = output_dir / f"CollectedData_{scorer}.csv"

        dataFrame.to_hdf(h5_save_path, key="df_with_missing", mode="w")
        dataFrame.to_csv(csv_save_path, na_rep="NaN")

    def read_trial_csv_with_validation(
        self,
        trialnames: list[Path],
    ) -> tuple[list[pd.DataFrame], list[list[int]], list[str]]:
        """
        Read trial CSVs and validate point name consistency.

        Args:
            trialnames: List of trial directory paths

        Returns:
            Tuple of (dataframes, valid_frame_indices, pointnames)
            - dataframes: List of DataFrames (one per trial, header row removed)
            - valid_frame_indices: List of lists containing valid frame indices per trial
            - pointnames: List of point/marker names

        Raises:
            ValueError: If CSVs have inconsistent point names or structure
            FileNotFoundError: If CSV file is missing or multiple CSVs found
        """

        dfs = []
        idx = []
        pnames = []

        for trial in trialnames:
            # Find CSV file
            contents = list(trial.glob("*.csv"))
            if len(contents) != 1:
                csv_list = [f.name for f in contents]
                raise FileNotFoundError(
                    f"Expected exactly 1 CSV file in {trial.name}, "
                    f"but found {len(contents)}: {csv_list}"
                )
            filename = contents[0]

            # Read CSV with header=None to match xrommtools pattern
            df1 = pd.read_csv(filename, sep=",", header=None)

            # Extract point names from header row (every 4th column, strip suffix)
            pointnames = df1.loc[0, ::4].astype(str).str[:-7].tolist()
            pnames.append(pointnames)

            # Remove header row
            df1 = df1.loc[1:,].reset_index(drop=True)

            # Find valid frames (rows where >= 50% of columns are non-NaN)
            ncol = df1.shape[1]
            temp_idx = list(df1.index.values[(~pd.isnull(df1)).sum(axis=1) >= ncol / 2])

            # Randomize frames within each trial
            random.shuffle(temp_idx)
            idx.append(temp_idx)
            dfs.append(df1)

        # Validate consistent point names across trials
        if any(pnames[0] != x for x in pnames):
            # Build detailed error message showing differences
            differences = []
            for i, pname_list in enumerate(pnames):
                if pname_list != pnames[0]:
                    trial_name = trialnames[i].name
                    diff_points = set(pname_list) ^ set(pnames[0])
                    differences.append(f"  {trial_name}: differs by {diff_points}")

            error_msg = (
                "Point names are not consistent across trials.\n"
                f"Reference trial ({trialnames[0].name}): {pnames[0]}\n"
                "Differences found in:\n" + "\n".join(differences)
            )
            raise ValueError(error_msg)

        return dfs, idx, pnames[0]

    def extract_2d_points_for_camera(
        self,
        trial_csv_df: pd.DataFrame,
        camera: int,
        picked_frames: list[int],
    ) -> pd.DataFrame:
        """
        Extract 2D point data for specific camera from trial CSV.

        Args:
            trial_csv_df: DataFrame loaded from trial CSV (header row already removed)
            camera: Camera number (1 or 2)
            picked_frames: List of frame indices to extract

        Returns:
            DataFrame with x,y coordinates for each point, sorted by column index
            Note: Columns are NOT renamed - caller should handle column naming

        Raises:
            ValueError: If camera number is not 1 or 2
            IndexError: If picked_frames contains invalid indices
        """
        # Validate camera number
        if camera not in [1, 2]:
            raise ValueError(f"Camera must be 1 or 2, got {camera}")

        # Validate frame indices
        max_frame = trial_csv_df.shape[0] - 1
        invalid_frames = [f for f in picked_frames if f > max_frame or f < 0]
        if invalid_frames:
            raise IndexError(
                f"Frame indices {invalid_frames} are out of bounds. "
                f"Valid range: 0-{max_frame}"
            )

        # Extract x and y positions for this camera
        # Camera 1: columns 0, 1, 4, 5, 8, 9, ... (offset 0)
        # Camera 2: columns 2, 3, 6, 7, 10, 11, ... (offset 2)
        xpos = trial_csv_df.iloc[picked_frames, 0 + (camera - 1) * 2 :: 4]
        ypos = trial_csv_df.iloc[picked_frames, 1 + (camera - 1) * 2 :: 4]

        # Concatenate and sort by column index
        temp_data = pd.concat([xpos, ypos], axis=1).sort_index(axis=1)

        return temp_data
