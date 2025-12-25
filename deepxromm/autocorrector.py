"""Performs XMAlab-style autocorrection on the trials and videos"""

from dataclasses import dataclass
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from deepxromm.logging import logger
from deepxromm.trial import Trial
from deepxromm.xrommtools import get_marker_names
from deepxromm.config_utilities import load_config_file


@dataclass
class AutocorrectParams:
    """Image processor config for autocorrect"""

    _search_area: int = 15
    threshold: int = 8
    krad: int = 17
    gsigma: int = 10
    img_wt: float = 3.6
    blur_wt: float = -2.9
    gamma: float = 0.1
    _test_autocorrect: bool = False
    trial_name: str = "your_trial_here"
    cam: str = "cam1"
    frame_num: int = 1
    marker: str = "your_marker_here"

    @property
    def search_area(self):
        """Search radius around the marker to search within"""
        return self._search_area

    @search_area.setter
    def search_area(self, value: int):
        """Sets a minimum of 10 for the search area"""
        if value >= 10:
            self._search_area = value
            return

        self._search_area = 10

    @property
    def test_autocorrect(self) -> bool:
        """Whether or not to test the autocorrect setup"""
        if not self._test_autocorrect:
            return self._test_autocorrect

        if self.trial_name == "your_trial_here":
            raise SyntaxError("Please specify a trial to test autocorrect() with")
        if self.marker == "your_marker_here":
            raise SyntaxError("Please specify a marker to test autocorrect() with")
        return self._test_autocorrect

    @test_autocorrect.setter
    def test_autocorrect(self, value):
        self._test_autocorrect = value


class Autocorrector:
    """Performs XMAlab-style autocorrection on the trials and videos"""

    def __init__(self, project):
        self.project = project
        self.autocorrect_settings = project.autocorrect_settings
        self.working_dir = project.working_dir
        self._trials_path = self.working_dir / "trials"
        self._dlc_config_path = project.path_config_file
        self._skip_stats = {}  # Track skipped markers: {trial_name: count}

    def autocorrect_trials(self):
        """Do XMAlab-style autocorrect on the tracked beads for all trials"""
        self._skip_stats = {}  # Reset for each run
        trials = self.project.list_trials()

        # Establish project vars
        yaml = YAML()
        with self._dlc_config_path.open("r") as dlc_config:
            dlc = yaml.load(dlc_config)

        iteration = dlc["iteration"]

        for trial_path in trials:
            # Find the appropriate pointsfile
            trial = Trial(trial_path)
            iteration_folder = trial_path / f"it{iteration}"
            trial_csv_location = trial.find_trial_csv(
                suffix=f"it{iteration}", identifier="Predicted2DPoints"
            )
            csv = pd.read_csv(trial_csv_location)
            out_name = (
                iteration_folder / f"{trial.trial_name}-AutoCorrected2DPoints.csv"
            )

            if self.autocorrect_settings.test_autocorrect:
                cams = self.project.cam
            else:
                cams = ["cam1", "cam2"]  # Assumes 2-camera setup

            # For each camera
            for cam in cams:
                csv = self._autocorrect_video(
                    cam, trial_path, csv, self.autocorrect_settings
                )

            # Print when autocorrect finishes
            if not self.autocorrect_settings.test_autocorrect:
                print(f"Done! Saving to {out_name}")
                csv.to_csv(out_name, index=False)

        # Print summary report
        self._print_skip_summary()

    def _autocorrect_video(
        self, cam, trial_path, csv, autocorrect_settings: AutocorrectParams
    ):
        """Run the autocorrect function on a single video within a single trial"""
        # Find the raw video
        trial = Trial(trial_path)
        video = cv2.VideoCapture(trial.find_cam_file(identifier=cam))
        if not video.isOpened():
            raise FileNotFoundError(f"Couldn't find a video at file path: {trial_path}")

        if autocorrect_settings.test_autocorrect:
            video.set(1, self.autocorrect_settings.frame_num - 1)
            ret, frame = video.read()
            if ret is False:
                raise IOError("Error reading video frame")
            self._autocorrect_frame(
                trial_path,
                frame,
                cam,
                autocorrect_settings.frame_num,
                csv,
                autocorrect_settings,
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
            csv = self._autocorrect_frame(
                trial_path, frame, cam, frame_index, csv, autocorrect_settings
            )
        return csv

    def _autocorrect_frame(
        self,
        trial_path,
        frame,
        cam,
        frame_index,
        csv,
        autocorrect_settings: AutocorrectParams,
    ):
        """Run the autocorrect function for a single frame (no output)"""
        # For each marker in the frame
        trial = Trial(trial_path)
        if autocorrect_settings.test_autocorrect:
            parts_unique = [autocorrect_settings.marker]
        else:
            dlc = load_config_file(self._dlc_config_path)
            iteration = dlc["iteration"]
            trial_csv_path = trial.find_trial_csv(
                suffix=f"{iteration}", identifier="Predicted2DPoints"
            )
            parts_unique = get_marker_names(trial_csv_path)
        for part in parts_unique:
            # Find point and offsets
            x_float = csv.loc[frame_index, f"{part}_{cam}_X"]
            y_float = csv.loc[frame_index, f"{part}_{cam}_Y"]
            search_area_with_offset = autocorrect_settings.search_area + 0.5
            x_start = int(x_float - search_area_with_offset)
            y_start = int(y_float - search_area_with_offset)
            x_end = int(x_float + search_area_with_offset)
            y_end = int(y_float + search_area_with_offset)

            subimage = frame[y_start:y_end, x_start:x_end]

            # Validate subimage before processing
            if subimage.size == 0 or subimage.shape[0] == 0 or subimage.shape[1] == 0:
                logger.warning(
                    f"Empty subimage for marker '{part}' at ({x_float:.1f}, {y_float:.1f}) "
                    f"in frame {frame_index + 1} on {cam}. Skipping autocorrect."
                )
                self._increment_skip_count(trial_path.name)
                continue

            try:
                subimage_filtered = self._filter_image(
                    subimage,
                    autocorrect_settings.krad,
                    autocorrect_settings.gsigma,
                    autocorrect_settings.img_wt,
                    autocorrect_settings.blur_wt,
                    autocorrect_settings.gamma,
                )

                subimage_float = subimage_filtered.astype(np.float32)
                radius = int(1.5 * 5 + 0.5)  # Isn't this just always 8?
                sigma = radius * math.sqrt(2 * math.log(255)) - 1
                subimage_blurred = cv2.GaussianBlur(
                    subimage_float, (2 * radius + 1, 2 * radius + 1), sigma
                )

                subimage_diff = subimage_float - subimage_blurred
            except cv2.error as e:
                logger.warning(
                    f"Skipping autocorrect for marker '{part}' in frame {frame_index + 1} "
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
                + autocorrect_settings.threshold * 0.01 * 255
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
            if autocorrect_settings.test_autocorrect:
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
            else:
                print(
                    f"Couldn't find better contour for {part} in {cam} video at {frame_index + 1} frame"
                )
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
