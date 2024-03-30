"""Performs XMAlab-style autocorrection on the trials and videos"""

import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ruamel.yaml import YAML

from .xma_data_processor import XMADataProcessor


class Autocorrector:
    """Performs XMAlab-style autocorrection on the trials and videos"""

    def __init__(self, config):
        self._trials_path = os.path.join(config["working_dir"], "trials")
        self._data_processor = XMADataProcessor(config)
        self._config = config
        self._dlc_config = config['path_config_file']

    def autocorrect_trial(self):
        '''Do XMAlab-style autocorrect on the tracked beads'''
        trials = self._data_processor.list_trials()

        # Establish project vars
        yaml = YAML()
        with open(self._dlc_config) as dlc_config:
            dlc = yaml.load(dlc_config)

        iteration = dlc['iteration']

        for trial in trials:
            # Find the appropriate pointsfile
            trial_path = os.path.join(self._trials_path, trial)
            iteration_path = os.path.join(trial_path, f'it{iteration}')

            try:
                csv = pd.read_csv(os.path.join(iteration_path, f'{trial}-Predicted2DPoints.csv'))
            except FileNotFoundError as e:
                print(f'Could not find predicted 2D points file. Please check the it{iteration} folder for trial {trial}')
                raise e
            out_name = os.path.join(iteration_path, f'{trial}-AutoCorrected2DPoints.csv')

            if self._config['test_autocorrect']:
                cams = [self._config['cam']]
            else:
                cams = ['cam1', 'cam2']

            # For each camera
            for cam in cams:
                csv = self._autocorrect_video(cam, trial_path, csv)

            # Print when autocorrect finishes
            if not self._config['test_autocorrect']:
                print(f'Done! Saving to {out_name}')
                csv.to_csv(out_name, index=False)

    def _autocorrect_video(self, cam, trial_path, csv):
        '''Run the autocorrect function on a single video within a single trial'''
        # Find the raw video
        video = cv2.VideoCapture(self._data_processor.find_cam_file(trial_path, cam))
        if not video.isOpened():
            raise FileNotFoundError(f'Couldn\'t find a video at file path: {trial_path}') from None

        if self._config['test_autocorrect']:
            video.set(1, self._config['frame_num'] - 1)
            ret, frame = video.read()
            if ret is False:
                raise IOError('Error reading video frame')
            self._autocorrect_frame(trial_path, frame, cam, self._config['frame_num'], csv)
            return csv

        # For each frame of video
        print(f'Total frames in video: {video.get(cv2.CAP_PROP_FRAME_COUNT)}')

        for frame_index in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
            # Load frame
            print(f'Current Frame: {frame_index + 1}')
            ret, frame = video.read()
            if ret is False:
                raise IOError('Error reading video frame')
            csv = self._autocorrect_frame(trial_path, frame, cam, frame_index, csv)
        return csv

    def _autocorrect_frame(self, trial_path, frame, cam, frame_index, csv):
        '''Run the autocorrect function for a single frame (no output)'''
        # For each marker in the frame
        if self._config['test_autocorrect']:
            parts_unique = [self._config['marker']]
        else:
            parts_unique = self._data_processor.get_bodyparts_from_xma(trial_path, mode='2D')
        for part in parts_unique:
            # Find point and offsets
            x_float = csv.loc[frame_index, part + '_' + cam + '_X']
            y_float = csv.loc[frame_index, part + '_' + cam + '_Y']
            search_area_with_offset = self._config['search_area'] + 0.5
            x_start = int(x_float - search_area_with_offset)
            y_start = int(y_float - search_area_with_offset)
            x_end = int(x_float + search_area_with_offset)
            y_end = int(y_float + search_area_with_offset)

            subimage = frame[y_start:y_end, x_start:x_end]

            subimage_filtered = self._filter_image(
                subimage, self._config['krad'], self._config['gsigma'], self._config['img_wt'],
                self._config['blur_wt'], self._config['gamma'])

            subimage_float = subimage_filtered.astype(np.float32)
            radius = int(1.5 * 5 + 0.5) #5 might be too high
            sigma = radius * math.sqrt(2 * math.log(255)) - 1
            subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, 2 * radius + 1), sigma)

            subimage_diff = subimage_float-subimage_blurred
            subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

            # Median
            subimage_median = cv2.medianBlur(subimage_diff, 3)

            # LUT
            subimage_median = self._filter_image(subimage_median, krad=3)

            # Thresholding
            subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
            min_val, _, _, _ = cv2.minMaxLoc(subimage_median)
            thres = 0.5 * min_val + 0.5 * np.mean(subimage_median) + self._config['threshold'] * 0.01 * 255
            _, subimage_threshold =  cv2.threshold(subimage_median, thres, 255, cv2.THRESH_BINARY_INV)

            # Gaussian blur
            subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3,3), 1.3)

            # Find contours
            contours, _ = cv2.findContours(subimage_gaussthresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x_start,y_start))
            contours_im = [contour-[x_start, y_start] for contour in contours]

            # Find closest contour
            dist = 1000
            best_index = -1
            detected_centers = {}
            for i, cnt in enumerate(contours):
                detected_center, _ = cv2.minEnclosingCircle(cnt)
                dist_tmp = math.sqrt((x_float - detected_center[0])**2 + (y_float - detected_center[1])**2)
                detected_centers[round(dist_tmp, 4)] = detected_center
                if dist_tmp < dist:
                    best_index = i
                    dist = dist_tmp

            if self._config['test_autocorrect']:
                print('Raw')
                self._show_crop(subimage, 15)

                print('Filtered')
                self._show_crop(subimage_filtered, 15)

                print(f'Blurred: {sigma}')
                self._show_crop(subimage_blurred, 15)

                print('Diff (Float - blurred)')
                self._show_crop(subimage_diff, 15)

                print('Median')
                self._show_crop(subimage_median, 15)

                print('Median filtered')
                self._show_crop(subimage_median, 15)

                print('Threshold')
                self._show_crop(subimage_threshold, 15)

                print('Gaussian')
                self._show_crop(subimage_threshold, 15)

                print('Best Contour')
                detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
                self._show_crop(subimage, 15, contours = [contours_im[best_index]], detected_marker = detected_center_im)

            # Save center of closest contour to CSV
            if best_index >= 0:
                detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
                csv.loc[frame_index, part + '_' + cam + '_X']  = detected_center[0]
                csv.loc[frame_index, part + '_' + cam + '_Y']  = detected_center[1]
        return csv

    def _filter_image(self, image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.10):
        '''Filter the image to make it easier to see the bead'''
        krad = krad*2+1
        # Gaussian blur
        image_blur = cv2.GaussianBlur(image, (krad, krad), gsigma)
        # Add to original
        image_blend = cv2.addWeighted(image, img_wt, image_blur, blur_wt, 0)
        lut = np.array([((i/255.0)**gamma)*255.0 for i in range(256)])
        image_gamma = image_blend.copy()
        im_type = len(image_gamma.shape)
        if im_type == 2:
            image_gamma = lut[image_gamma]
        elif im_type == 3:
            image_gamma[:,:,0] = lut[image_gamma[:,:,0]]
            image_gamma[:,:,1] = lut[image_gamma[:,:,1]]
            image_gamma[:,:,2] = lut[image_gamma[:,:,2]]
        return image_gamma

    def _show_crop(self, src, center, scale=5, contours=None, detected_marker=None):
        '''Display a visual of the marker and Python's projected center'''
        if len(src.shape) < 3:
            src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
        image = src.copy().astype(np.uint8)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        if contours:
            overlay = image.copy()
            scaled_contours = [contour*scale for contour in contours]
            cv2.drawContours(overlay, scaled_contours, -1, (255,0,0),2)
            image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
        cv2.drawMarker(image, (center*scale, center*scale), color = (0,255,255), markerType = cv2.MARKER_CROSS, markerSize = 10, thickness = 1)
        if detected_marker:
            cv2.drawMarker(image,
            (int(detected_marker[0]*scale),
            int(detected_marker[1]*scale)),
            color = (255,0,0),
            markerType = cv2.MARKER_CROSS,
            markerSize = 10,
            thickness = 1)
        plt.imshow(image)
        plt.show()
