"""Performs XMAlab-style autocorrection on the trials and videos"""

import logging
import math
from pathlib import Path
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .xma_data_processor import XMADataProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="autocorrecter.log",
    encoding="utf-8",
    level=os.environ.get("DEEPXROMM_LOG_LEVEL", "INFO").upper(),
)


class Autocorrector:
    """Performs XMAlab-style autocorrection on the trials and videos"""

    def __init__(self, config):
        self.working_dir = Path(config["working_dir"])
        self._trials_path = self.working_dir / "trials"
        self._data_processor = XMADataProcessor(config)
        self._config = config
        self._dlc_config_path = Path(config["path_config_file"])
        self._skip_stats = {}  # Track skipped markers: {trial_name: count}

    def autocorrect_trials(self):
        """Do XMAlab-style autocorrect on the tracked beads for all trials"""
        self._skip_stats = {}  # Reset for each run
        trials = self._data_processor.list_trials()

        # Establish project vars
        with self._dlc_config_path.open("r") as dlc_config:
            dlc = yaml.safe_load(dlc_config)

        iteration = dlc["iteration"]

        for trial in trials:
            # Find the appropriate pointsfile
            trial_name = trial.name
            iteration_folder = trial / f"it{iteration}"
            trial_csv_location = self._data_processor.find_trial_csv(
                iteration_folder, "Predicted2DPoints"
            )
            csv = pd.read_csv(trial_csv_location)
            out_name = iteration_folder / f"{trial_name}-AutoCorrected2DPoints.csv"

            if self._config["test_autocorrect"]:
                cams = [self._config["cam"]]
            else:
                cams = ["cam1", "cam2"]  # Assumes 2-camera setup

            # For each camera
            for cam in cams:
                csv = self._autocorrect_video(cam, self._trials_path / trial_name, csv)

            # Print when autocorrect finishes
            if not self._config["test_autocorrect"]:
                print(f"Done! Saving to {out_name}")
                csv.to_csv(out_name, index=False)

        # Print summary report
        self._print_skip_summary()

    def _autocorrect_video(self, cam, trial_path, csv):
        """Run the autocorrect function on a single video within a single trial"""
        # Find the raw video
        video = cv2.VideoCapture(self._data_processor.find_cam_file(trial_path, cam))
        if not video.isOpened():
            raise FileNotFoundError(f"Couldn't find a video at file path: {trial_path}")

        if self._config["test_autocorrect"]:
            video.set(1, self._config["frame_num"] - 1)
            ret, frame = video.read()
            if ret is False:
                raise IOError("Error reading video frame")
            self._autocorrect_frame(
                trial_path, frame, cam, self._config["frame_num"], csv
            )
            return csv

        # For each frame of video
        print(f"Total frames in video: {video.get(cv2.CAP_PROP_FRAME_COUNT)}")

        for frame_index in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Load frame
            if frame_index % 50 == 0:
                print(f"Current Frame: {frame_index + 1}")
            ret, frame = video.read()
            if ret is False:
                raise IOError("Error reading video frame")
            csv = self._autocorrect_frame(trial_path, frame, cam, frame_index, csv)
        return csv

    def _autocorrect_frame(self, trial_path, frame, cam, frame_index, csv):
        """Run the autocorrect function for a single frame (no output)"""
        # For each marker in the frame
        if self._config["test_autocorrect"]:
            parts_unique = [self._config["marker"]]
        else:
            with open(self._dlc_config_path) as dlc_config:
                dlc = yaml.safe_load(dlc_config)
            iteration = dlc["iteration"]
            iteration_path = trial_path / f"it{iteration}"
            trial_csv_path = self._data_processor.find_trial_csv(
                iteration_path, "Predicted2DPoints"
            )
            parts_unique = self._data_processor.get_bodyparts_from_xma(
                trial_csv_path, mode="2D"
            )
        for part in parts_unique:
            # Find point and offsets
            x_float = csv.loc[frame_index, part + "_" + cam + "_X"]
            y_float = csv.loc[frame_index, part + "_" + cam + "_Y"]
            search_area_with_offset = self._config["search_area"] + 0.5
            x_start = int(x_float - search_area_with_offset)
            y_start = int(y_float - search_area_with_offset)
            x_end = int(x_float + search_area_with_offset)
            y_end = int(y_float + search_area_with_offset)

            subimage = frame[y_start:y_end, x_start:x_end]

            # Validate subimage before processing
            if subimage.size == 0 or subimage.shape[0] == 0 or subimage.shape[1] == 0:
                logger.warning(
                    f"Empty subimage for marker '{part}' at ({x_float:.1f}, {y_float:.1f}) "
                    f"in frame {frame_index} on {cam}. Skipping autocorrect."
                )
                self._increment_skip_count(trial_path.name)
                continue

            try:
                subimage_filtered = self._filter_image(
                    subimage,
                    self._config["krad"],
                    self._config["gsigma"],
                    self._config["img_wt"],
                    self._config["blur_wt"],
                    self._config["gamma"],
                )

                subimage_float = subimage_filtered.astype(np.float32)
                radius = int(1.5 * 5 + 0.5)  # 5 might be too high
                sigma = radius * math.sqrt(2 * math.log(255)) - 1
                subimage_blurred = cv2.GaussianBlur(
                    subimage_float, (2 * radius + 1, 2 * radius + 1), sigma
                )

                subimage_diff = subimage_float - subimage_blurred
            except cv2.error as e:
                logger.warning(
                    f"Skipping autocorrect for marker '{part}' in frame {frame_index} "
                    f"on {cam} at ({x_float:.1f}, {y_float:.1f}) (main blur) "
                    f"[subimage: {subimage.shape}]: {str(e)}"
                )
                self._increment_skip_count(trial_path.name)
                continue

            subimage_diff
            subimage_diff = cv2.normalize(
                subimage_diff, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

            # Median
            subimage_median = cv2.medianBlur(subimage_diff, 3)

            # LUT
            subimage_median_filtered = self._filter_image(subimage_median, krad=3)

            # Thresholding
            subimage_median_threshold = cv2.cvtColor(
                subimage_median_filtered, cv2.COLOR_BGR2GRAY
            )
            min_val, _, _, _ = cv2.minMaxLoc(subimage_median_threshold)
            thres = (
                0.5 * min_val
                + 0.5 * np.mean(subimage_median_threshold)
                + self._config["threshold"] * 0.01 * 255
            )
            _, subimage_threshold = cv2.threshold(
                subimage_median_threshold, thres, 255, cv2.THRESH_BINARY_INV
            )

            # Gaussian blur
            try:
                subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3, 3), 1.3)
            except cv2.error as e:
                logger.warning(
                    f"Skipping autocorrect for marker '{part}' in frame {frame_index} "
                    f"on {cam} at ({x_float:.1f}, {y_float:.1f}) (threshold blur): {str(e)}"
                )
                self._increment_skip_count(trial_path.name)
                continue

            # Find contours
            contours, _ = cv2.findContours(
                subimage_gaussthresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
                offset=(x_start, y_start),
            )
            contours_im = [contour - [x_start, y_start] for contour in contours]

            # Find closest contour
            dist = 1000
            best_index = -1
            detected_centers = {}
            for i, cnt in enumerate(contours):
                detected_center, _ = cv2.minEnclosingCircle(cnt)
                dist_tmp = math.sqrt(
                    (x_float - detected_center[0]) ** 2
                    + (y_float - detected_center[1]) ** 2
                )
                detected_centers[round(dist_tmp, 4)] = detected_center
                if dist_tmp < dist:
                    best_index = i
                    dist = dist_tmp

            # Fix how this displays, because this logic does not track
            if self._config["test_autocorrect"]:
                print("Raw")
                self._show_crop(subimage, 15)

                print("Filtered")
                self._show_crop(subimage_filtered, 15)

                print(f"Blurred: {sigma}")
                self._show_crop(subimage_blurred, 15)

                print("Diff (Float - blurred)")
                self._show_crop(subimage_diff, 15)

                print("Median")
                self._show_crop(subimage_median, 15)

                print("Median filtered")
                self._show_crop(subimage_median_filtered, 15)

                print("Threshold")
                self._show_crop(subimage_threshold, 15)

                print("Gaussian")
                self._show_crop(subimage_gaussthresh, 15)

                print("Best Contour")
                detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
                self._show_crop(
                    subimage,
                    15,
                    contours=[contours_im[best_index]],
                    detected_marker=detected_center_im,
                )

            # Save center of closest contour to CSV
            if best_index >= 0:
                detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
                csv.loc[frame_index, part + "_" + cam + "_X"] = detected_center[0]
                csv.loc[frame_index, part + "_" + cam + "_Y"] = detected_center[1]
        return csv

    def _filter_image(
        self, image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.10
    ):
        """Filter the image to make it easier to see the bead"""
        krad = krad * 2 + 1
        # Gaussian blur
        try:
            image_blur = cv2.GaussianBlur(image, (krad, krad), gsigma)
        except cv2.error as e:
            logger.warning(f"GaussianBlur failed in filter_image: {str(e)}")
            return image  # Return original unchanged
        # Add to original
        image_blend = cv2.addWeighted(image, img_wt, image_blur, blur_wt, 0)
        lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)])
        image_gamma = image_blend.copy()
        im_type = len(image_gamma.shape)
        if im_type == 2:
            image_gamma = lut[image_gamma]
        elif im_type == 3:
            image_gamma[:, :, 0] = lut[image_gamma[:, :, 0]]
            image_gamma[:, :, 1] = lut[image_gamma[:, :, 1]]
            image_gamma[:, :, 2] = lut[image_gamma[:, :, 2]]
        return image_gamma

    def _show_crop(self, src, center, scale=5, contours=None, detected_marker=None):
        """Display a visual of the marker and Python's projected center"""
        if len(src.shape) < 3:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        image = src.copy().astype(np.uint8)
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        if contours:
            overlay = image.copy()
            scaled_contours = [contour * scale for contour in contours]
            cv2.drawContours(overlay, scaled_contours, -1, (255, 0, 0), 2)
            image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
        cv2.drawMarker(
            image,
            (center * scale, center * scale),
            color=(0, 255, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=10,
            thickness=1,
        )
        if detected_marker:
            cv2.drawMarker(
                image,
                (int(detected_marker[0] * scale), int(detected_marker[1] * scale)),
                color=(255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=1,
            )
        plt.imshow(image)
        plt.show()

    def _increment_skip_count(self, trial_name):
        """Track skipped marker for summary reporting"""
        if trial_name not in self._skip_stats:
            self._skip_stats[trial_name] = 0
        self._skip_stats[trial_name] += 1

    def _print_skip_summary(self):
        """Print summary of skipped markers during autocorrection"""
        total_skipped = sum(self._skip_stats.values())

        if total_skipped == 0:
            return  # Silent when no issues

        print("\n=== Autocorrect Summary ===")
        print(f"Total markers skipped: {total_skipped}")

        if len(self._skip_stats) > 0:
            print("Breakdown by trial:")
            for trial_name, count in self._skip_stats.items():
                print(f"  - {trial_name}: {count} marker(s) skipped")

        print("Check autocorrecter.log for details")
