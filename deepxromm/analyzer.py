"""Primary interface for analyzing the XROMM data using trained network"""

import math
from pathlib import Path

from itertools import combinations
import cv2
import deeplabcut
import imagehash
import pandas as pd
from PIL import Image

from deepxromm.logging import logger
from deepxromm.dlc_config import DlcConfigFactory
from deepxromm.trial import Trial
from deepxromm.xrommtools import get_marker_and_cam_names


class Analyzer:
    """Analyzes XROMM videos using trained network."""

    def __init__(self, project):
        self.working_dir = project.working_dir
        self.project_config = project.project_config_path
        self.dlc_config = DlcConfigFactory.load_existing_config(
            project.mode, project.path_config_file
        )
        self._project = project
        self._trials_path = self.working_dir / "trials"

    def analyze_videos(self, **kwargs):
        """Analyze videos with a pre-existing network"""
        trials = self._project.list_trials()

        # Establish project vars
        iteration = self.dlc_config.iteration

        mode = self._project.mode
        if mode in ["2D", "per_cam"]:
            self._analyze_xromm_videos(iteration)

        elif mode == "rgb":
            for trial_path in trials:
                trial = Trial(trial_path)
                trial.make_rgb_video(codec=self._project.video_codec, **kwargs)
                trial = trial_path.name
                current_files = trial_path.glob("*")
                logger.debug(f"Current files in directory {current_files}")
                video_path = trial_path / f"{trial}_rgb.avi"
                destfolder = trial_path / f"it{iteration}"
                deeplabcut.analyze_videos(
                    str(self.dlc_config.path_config_file),
                    str(
                        video_path
                    ),  # DLC relies on .endswith to determine suffix, so this needs to be a string
                    destfolder=destfolder,
                    save_as_csv=True,
                )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def _analyze_xromm_videos(self, iteration: int) -> None:
        """Analyze all novel videos in the 'trials' folder of a deepxromm project"""
        # assumes you have cam1 and cam2 videos as .avi in their own seperate trial folders
        # assumes all folders w/i new_data_path are trial folders

        # analyze videos
        cameras = [1, 2]
        trials = self._project.list_trials()
        mode = self._project.mode

        for trialpath in trials:
            trial = Trial(trialpath)
            savepath = trialpath / f"it{iteration}"
            if savepath.exists():
                temp = savepath.glob("*Predicted2DPoints.csv")
                if temp:
                    logger.warning(
                        f"There are already predicted points in iteration {iteration} subfolders... skipping point prediction"
                    )
                    return
            else:
                savepath.mkdir(parents=True, exist_ok=True)  # make new folder
            # get video file
            for camera in cameras:
                # Error handling handled by find_cam_file helper
                video = trial.find_cam_file(identifier=f"cam{camera}")
                # analyze video
                if mode == "2D":
                    deeplabcut.analyze_videos(
                        str(self._project.path_config_file),
                        [
                            str(video)
                        ],  # DLC uses endswith filtering for suffixes for some reason
                        destfolder=savepath,
                        save_as_csv=True,
                    )
                else:
                    configs = [
                        str(self._project.path_config_file),
                        str(self._project.path_config_file_2),
                    ]
                    deeplabcut.analyze_videos(
                        configs[camera - 1],
                        [
                            str(video)
                        ],  # DLC uses endswith filtering for suffixes for some reason
                        destfolder=savepath,
                        save_as_csv=True,
                    )

    def analyze_video_similarity_project(self):
        """Analyze all videos in a project and take their average similar. This is dangerous, as it will assume that all cam1/cam2 pairs match
        or don't match!"""
        similarity_score = {}
        trial_combos = combinations(self._project.list_trials(), 2)
        for trial1, trial2 in trial_combos:
            similarity_score[(trial1, trial2)] = self.analyze_video_similarity_trial(
                [trial1, trial2]
            )
        return similarity_score

    def analyze_video_similarity_trial(self, trials: list[Path]):
        """Analyze the average similarity between trials using image hashing"""
        cameras = ["cam1", "cam2"]
        videos = {
            (trial_path, cam): cv2.VideoCapture(
                Trial(trial_path).find_cam_file(identifier=cam)
            )
            for trial_path in trials
            for cam in cameras
        }

        # Compare hashes based on the camera views configuration
        if self._project.cam1s_are_the_same_view:
            cam1_diff, noc1 = self._compare_two_videos(
                videos[(trials[0], "cam1")], videos[(trials[1], "cam1")]
            )
            cam2_diff, noc2 = self._compare_two_videos(
                videos[(trials[0], "cam2")], videos[(trials[1], "cam2")]
            )
        else:
            cam1_diff, noc1 = self._compare_two_videos(
                videos[(trials[0], "cam1")], videos[(trials[1], "cam2")]
            )
            cam2_diff, noc2 = self._compare_two_videos(
                videos[(trials[0], "cam2")], videos[(trials[1], "cam1")]
            )

        # Calculate the average difference for each camera view
        cam1_avg_diff = cam1_diff / noc1 if noc1 > 0 else 0
        cam2_avg_diff = cam2_diff / noc2 if noc2 > 0 else 0

        # Calculate the overall trial average difference
        trial_avg_diff = (cam1_avg_diff + cam2_avg_diff) / 2

        # Note: The number of comparisons (noc) grows with video size, which could potentially affect the similarity measure,
        # making larger videos appear more similar than they actually are. This aspect may need further consideration.
        return trial_avg_diff

    def analyze_marker_similarity_project(self):
        """Analyze all videos in a project and get their average rhythmicity. This assumes that all cam1/2 pairs are either the same or different!"""
        marker_similarity = {}

        trial_perms = combinations(self._project.list_trials(), 2)
        logger.debug(f"Trial permutations for project: {trial_perms}")
        for trial1, trial2 in trial_perms:
            marker_similarity[(trial1, trial2)] = abs(
                self.analyze_marker_similarity_trial(trial1, trial2)
            )
        return marker_similarity

    def analyze_marker_similarity_trial(self, trial1_path: Path, trial2_path: Path):
        """Analyze marker similarity for a pair of trials using the distance formula."""
        # Get a list of markers that each trial have in common
        trial1 = Trial(trial1_path)
        trial2 = Trial(trial2_path)
        bodyparts1_csv_path = trial1.find_trial_csv()
        bodyparts2_csv_path = trial2.find_trial_csv()
        bodyparts1 = get_marker_and_cam_names(bodyparts1_csv_path)
        bodyparts2 = get_marker_and_cam_names(bodyparts2_csv_path)
        markers_in_common = [marker for marker in bodyparts1 if marker in bodyparts2]
        logger.debug(f"Markers in common for similarity analysis: {markers_in_common}")

        # Analyze intermarker distances for each marker in common
        trial1_csv = pd.read_csv(bodyparts1_csv_path)
        trial2_csv = pd.read_csv(bodyparts2_csv_path)
        avg_distances = []
        for marker in markers_in_common:
            # Debug logging
            x1_vals = trial1_csv[f"{marker}_X"]
            logger.debug(f"Trial1 values for {marker}_X: {x1_vals}")
            x2_vals = trial2_csv[f"{marker}_X"]
            logger.debug(f"Trial2 values for {marker}_X: {x2_vals}")

            # Get mean positions for each marker
            avg_x1, avg_y1 = (
                trial1_csv[f"{marker}_X"].mean(),
                trial1_csv[f"{marker}_Y"].mean(),
            )
            logger.debug(
                f"Average trial1 position for marker {marker}: ({avg_x1}, {avg_y1})"
            )
            avg_x2, avg_y2 = (
                trial2_csv[f"{marker}_X"].mean(),
                trial2_csv[f"{marker}_Y"].mean(),
            )
            logger.debug(
                f"Average trial2 position for marker {marker}: ({avg_x2}, {avg_y2})"
            )

            # Calculate the distance between the average positions for this marker in the two trials
            distance = math.sqrt((avg_x2 - avg_x1) ** 2 + (avg_y2 - avg_y1) ** 2)
            avg_distances.append(distance)

        logger.debug(
            f"Avg distances for trials {trial1_path.name} and {trial2_path.name}: {avg_distances}"
        )
        # Calculate the mean of the distances to get an overall similarity measure
        marker_similarity = (
            sum(avg_distances) / len(avg_distances) if avg_distances else 0
        )

        return marker_similarity

    def get_max_dissimilarity_for_trial(self, trial_path: Path, window):
        """Calculate the dissimilarity within the trial given the frame sliding window."""
        trial = Trial(trial_path)
        video1 = cv2.VideoCapture(trial.find_cam_file(identifier="cam1"))
        video2 = cv2.VideoCapture(trial.find_cam_file(identifier="cam2"))

        hashes1 = self._hash_trial_video(video1)
        hashes2 = self._hash_trial_video(video2)

        return self._find_dissimilar_regions(hashes1, hashes2, window)

    def _compare_two_videos(self, video1, video2):
        """Compare two videos using image hashing"""
        hashes1 = self._hash_trial_video(video1)
        hashes2 = self._hash_trial_video(video2)

        video1_frames = len(hashes1)
        video2_frames = len(hashes2)
        # Thought: maybe this should be computed alongside the for loop?
        noc = math.perm(
            video1_frames + video2_frames, 2
        )  # Might need revision based on actual comparison logic

        logger.debug(f"Video 1 frames: {video1_frames}")
        logger.debug(f"Video 2 frames: {video2_frames}")

        logger.info("Comparing hashes between videos")
        hash_dif = sum(hash1 - hash2 for hash1 in hashes1 for hash2 in hashes2)

        return hash_dif, noc

    def _hash_trial_video(self, video):
        """Generate image hashes for a single video"""
        video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video frames: {video_frames}")

        hashes = []
        logger.info("Creating hashes for video")
        for i in range(video_frames):
            if i % 50 == 0:
                logger.info(f"Current frame: {i}")
            ret, frame = video.read()
            if not ret:
                # Should this throw an error?
                cv2.destroyAllWindows()
                raise IOError("Error reading video frame")
            hashes.append(imagehash.phash(Image.fromarray(frame)))

        return hashes

    def _find_dissimilar_regions(self, hashes1, hashes2, window):
        """Find the region of maximum dissimilarity given 2 lists of hashes and a sliding window (how many frames)"""
        start_frame_vid1 = 0
        start_frame_vid2 = 0
        max_hash_dif_vid1 = 0
        max_hash_dif_vid2 = 0
        hash_dif_vid1 = 0
        hash_dif_vid2 = 0

        for slider in range(0, len(hashes1) // window):
            logger.debug(f"Current start frame {slider * window}")
            hash_dif_vid1, hash_dif_vid2 = self._compare_hash_sets(
                hashes1[slider * window : (slider + 1) * window],
                hashes2[slider * window : (slider + 1) * window],
            )

            logger.debug(f"Current hash diff (vid 1): {hash_dif_vid1}")
            logger.debug(f"Current hash diff (vid 2): {hash_dif_vid2}")
            if hash_dif_vid1 > max_hash_dif_vid1:
                max_hash_dif_vid1 = hash_dif_vid1
                start_frame_vid1 = slider * window

            if hash_dif_vid2 > max_hash_dif_vid2:
                max_hash_dif_vid2 = hash_dif_vid2
                start_frame_vid2 = slider * window

            logger.info(f"Max hash diff (vid 1): {max_hash_dif_vid1}")
            logger.info(f"Max hash diff (vid 2): {max_hash_dif_vid2}")

            logger.info(f"Start frame (vid 1): {start_frame_vid1}")
            logger.info(f"Start frame (vid 2): {start_frame_vid2}")

        return start_frame_vid1, start_frame_vid2

    def _compare_hash_sets(self, hashes1, hashes2):
        """Compares two sets of image hashes to find dissimilarities"""
        hash1_dif = 0
        hash2_dif = 0

        logger.debug(f"Hash set 1 {hashes1[0]}")
        logger.debug(f"Hash set 2 {hashes2[0]}")
        # Compares all possible combinations of images
        for combination in combinations(hashes1, 2):
            hash1_dif = hash1_dif + (combination[0] - combination[1])

        for combination in combinations(hashes2, 2):
            hash2_dif = hash2_dif + (combination[0] - combination[1])

        return hash1_dif, hash2_dif
