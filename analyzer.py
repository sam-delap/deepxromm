"""Primary interface for analyzing the XROMM data using trained network"""

import os
import math

from itertools import combinations
import cv2
import deeplabcut
import imagehash
import pandas as pd
from PIL import Image
from ruamel.yaml import YAML

import xrommtools
from xma_data_processor import XMADataProcessor


class Analyzer:
    """Analyzes XROMM videos using trained network."""

    def __init__(self, config):
        self._trials_path = os.path.join(config["working_dir"], "trials")
        self._data_processor = XMADataProcessor(config)
        self._config = config
        self._dlc_config = config['path_config_file']

    def analyze_videos(self):
        '''Analyze videos with a pre-existing network'''

        # Error if trials directory is empty
        trials = [folder for folder in os.listdir(self._trials_path) if os.path.isdir(os.path.join(self._trials_path, folder)) and not folder.startswith('.')]
        if len(trials) <= 0:
            raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {self._trials_path} folder')

        # Establish project vars
        yaml = YAML()
        with open(self._dlc_config) as dlc_config:
            dlc = yaml.load(dlc_config)
        iteration = dlc['iteration']

        mode = self._config['tracking_mode']
        if mode == '2D':
            xrommtools.analyze_xromm_videos(self._dlc_config, self._trials_path, iteration)
        elif mode == 'per_cam':
            xrommtools.analyze_xromm_videos(path_config_file=self._dlc_config,
                                            path_config_file_cam2=self._config['path_config_file_2'],
                                            path_data_to_analyze=self._trials_path,
                                            iteration=iteration,
                                            nnetworks=2)
        else:
            for trial in trials:
                trial_path =  os.path.join(self._trials_path, trial)
                video_path = f'{trial_path}/{trial}_rgb.avi'
                if not os.path.exists(video_path):
                    self._data_processor.make_rgb_video(trial_path)
                destfolder = f'{trial_path}/it{iteration}/'
                deeplabcut.analyze_videos(self._dlc_config, video_path, destfolder=destfolder, save_as_csv=True)
                self._split_dlc_to_xma(trial)

    def analyze_video_similarity_project(self):
        '''Analyze all videos in a project and take their average similar. This is dangerous, as it will assume that all cam1/cam2 pairs match
        or don't match!'''
        similarity_score = {}
        list_of_trials = [folder for folder in os.listdir(self._trials_path) if os.path.isdir(os.path.join(self._trials_path, folder)) and not folder.startswith('.')]
        yaml = YAML()

        trial_perms = combinations(list_of_trials, 2)
        for trial1, trial2 in trial_perms:
            self._config['trial_1_name'] = trial1
            self._config['trial_2_name'] = trial2
            with open(os.path.join(self._config['working_dir'], 'project_config.yaml'), 'w') as file:
                yaml.dump(self._config, file)
            # TODO: pass the trial names directly instead of writing into config
            similarity_score[(trial1, trial2)] = self.analyze_video_similarity_trial()
        return similarity_score

    def analyze_video_similarity_trial(self):
        '''Analyze the average similarity between trials using image hashing'''
        # Find videos for each trial
        trial1 = self._config['trial_1_name']
        trial2 = self._config['trial_2_name']
        trial1_cam1 = cv2.VideoCapture(os.path.join(self._trials_path, trial1, trial1 + '_cam1.avi'))
        trial2_cam1 = cv2.VideoCapture(os.path.join(self._trials_path, trial2, trial2 + '_cam1.avi'))
        trial1_cam2 = cv2.VideoCapture(os.path.join(self._trials_path, trial1, trial1 + '_cam2.avi'))
        trial2_cam2 = cv2.VideoCapture(os.path.join(self._trials_path, trial2, trial2 + '_cam2.avi'))
        # Compare hashes
        if self._config['cam1s_are_the_same_view']:
            cam1_dif, noc1 = self._compare_two_videos(trial1_cam1, trial2_cam1)
            cam2_dif, noc2 = self._compare_two_videos(trial1_cam2, trial2_cam2)
        else:
            cam1_dif, noc1 = self._compare_two_videos(trial1_cam1, trial2_cam2)
            cam2_dif, noc2 = self._compare_two_videos(trial1_cam2, trial2_cam1)
        return (cam1_dif + cam2_dif) / (noc1 + noc2)

    def analyze_marker_similarity_project(self):
        '''Analyze all videos in a project and get their average rhythmicity. This assumes that all cam1/2 pairs are either the same or different!'''
        marker_similarity = {}
        list_of_trials = [folder for folder in os.listdir(self._trials_path) if os.path.isdir(os.path.join(self._trials_path, folder)) and not folder.startswith('.')]
        yaml = YAML()

        trial_perms = combinations(list_of_trials, 2)
        for trial1, trial2 in trial_perms:
            self._config['trial_1_name'] = trial1
            self._config['trial_2_name'] = trial2
            with open(os.path.join(self._config['working_dir'], 'project_config.yaml'), 'w') as file:
                yaml.dump(self._config, file)
            # TODO: pass the trial names directly instead of writing into config
            marker_similarity[(trial1, trial2)] = abs(self.analyze_marker_similarity_trial())
        return marker_similarity

    def analyze_marker_similarity_trial(self):
        '''Analyze marker similarity for a pair of trials. Returns the mean difference for paired marker positions (X - X, Y - Y for each marker)'''
        # Find CSVs for each trial
        trial1 = self._config['trial_1_name']
        trial2 = self._config['trial_2_name']
        trial1_path = os.path.join(self._trials_path, trial1)
        trial2_path = os.path.join(self._trials_path, trial2)
        # Get a list of markers that each trial have in commmon
        # Marker similarity is always in rgb mode.
        bodyparts1 = self._data_processor.get_bodyparts_from_xma(trial1_path, mode='rgb')
        bodyparts2 = self._data_processor.get_bodyparts_from_xma(trial2_path, mode='rgb')

        markers_in_common = [marker for marker in bodyparts1 if marker in bodyparts2]
        bodyparts_xy = [f'{marker}_X' for marker in markers_in_common] + [f'{marker}_Y' for marker in markers_in_common]
        trial1_csv = pd.read_csv(os.path.join(trial1_path, trial1 + '.csv'))
        trial2_csv = pd.read_csv(os.path.join(trial2_path, trial2 + '.csv'))

        marker_similarity = sum((trial1_csv[marker] - trial2_csv[marker]).sum() / (len(trial1_csv[marker]) + len(trial2_csv[marker])) for marker in bodyparts_xy) / len(bodyparts_xy)

        return marker_similarity

    def get_max_dissimilarity_for_trial(self, trial_path, window):
        '''Calculate the dissimilarity within the trial given the frame sliding window.'''
        trial_name = os.path.basename(trial_path)
        video1 = cv2.VideoCapture(os.path.join(trial_path, f'{trial_name}_cam1.avi'))
        video2 = cv2.VideoCapture(os.path.join(trial_path, f'{trial_name}_cam2.avi'))

        hashes1, hashes2 = self._hash_trial_videos(video1, video2)
        return self._find_dissimilar_regions(hashes1, hashes2, window)

    def find_dissimilar_regions(self, hashes1, hashes2, window):
        '''Find the region of maximum dissimilarity given 2 lists of hashes and a sliding window (how many frames)'''
        start_frame_vid1 = 0
        start_frame_vid2 = 0
        max_hash_dif_vid1 = 0
        max_hash_dif_vid2 = 0
        hash_dif_vid1 = 0
        hash_dif_vid2 = 0

        for slider in range(0, len(hashes1) // window):
            print(f'Current start frame {slider * window}')
            hash_dif_vid1, hash_dif_vid2 = self._compare_hash_sets(hashes1[slider * window:(slider + 1) * window], hashes2[slider * window:(slider + 1) * window])

            print(f'Current hash diff (vid 1): {hash_dif_vid1}')
            print(f'Current hash diff (vid 2): {hash_dif_vid2}')
            if hash_dif_vid1 > max_hash_dif_vid1:
                max_hash_dif_vid1 = hash_dif_vid1
                start_frame_vid1 = slider * window

            if hash_dif_vid2 > max_hash_dif_vid2:
                max_hash_dif_vid2 = hash_dif_vid2
                start_frame_vid2 = slider * window

            print(f'Max hash diff (vid 1): {max_hash_dif_vid1}')
            print(f'Max hash diff (vid 2): {max_hash_dif_vid2}')

            print(f'Start frame (vid 1): {start_frame_vid1}')
            print(f'Start frame (vid 2): {start_frame_vid2}')

        return start_frame_vid1, start_frame_vid2

    def _split_dlc_to_xma(self, trial, save_hdf=True):
        '''Takes the output from RGB deeplabcut and splits it into XMAlab-readable output'''
        bodyparts_xy = []
        yaml = YAML()
        with open(self._dlc_config) as dlc_config:
            dlc = yaml.load(dlc_config)
        iteration = dlc['iteration']
        trial_path = os.path.join(self._trials_path, trial)

        rgb_parts = self._data_processor.get_bodyparts_from_xma(trial_path, mode='rgb')
        for part in rgb_parts:
            bodyparts_xy.append(part+'_X')
            bodyparts_xy.append(part+'_Y')

        csv_path = [file for file in os.listdir(f'{trial_path}/it{iteration}') if '.csv' in file and '-2DPoints' not in file]
        if len(csv_path) > 1:
            raise FileExistsError('Found more than 1 data CSV for RGB trial. Please remove CSVs from older analyses from this folder before analyzing.')
        if len(csv_path) < 1:
            raise FileNotFoundError(f'Couldn\'t find data CSV for trial {trial}. Something wrong with DeepLabCut?')

        csv_path = csv_path[0]
        xma_csv_path = f'{trial_path}/it{iteration}/{trial}-Predicted2DPoints.csv'

        df = pd.read_csv(f'{trial_path}/it{iteration}/{csv_path}', skiprows=1)
        df.index = df['bodyparts']
        df = df.drop(columns=df.columns[[df.loc['coords'] == 'likelihood']])
        df = df.drop(columns=[column for column in df.columns if column not in rgb_parts and column not in [f'{bodypart}.1' for bodypart in rgb_parts]])
        df.columns = bodyparts_xy
        df = df.drop(index='coords')
        df.to_csv(xma_csv_path, index=False)
        print("Successfully split DLC format to XMALab 2D points; saved "+str(xma_csv_path))
        if save_hdf:
            tracked_hdf = os.path.splitext(csv_path)[0]+'.h5'
            df.to_hdf(tracked_hdf, 'df_with_missing', format='table', mode='w', nan_rep='NaN')

    def _compare_two_videos(self, video1, video2):
        '''Do an image hashing between two videos'''
        video1_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        video2_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
        noc = math.perm(video1_frames + video2_frames, 2)
        print(f'Video 1 frames: {video1_frames}')
        print(f'Video 2 frames: {video2_frames}')
        hash_dif = 0

        hashes1 = []
        print ('Creating hashes for video 1')
        for i in range(video1_frames):
            print(f'Current frame (video 1): {i}')
            ret, frame1 = video1.read()
            if not ret:
                print('Error reading video 1 frame')
                cv2.destroyAllWindows()
                break
            hashes1.append(imagehash.phash(Image.fromarray(frame1)))

        print('Creating hashes for video 2')
        hashes2 = []
        for j in range(video2_frames):
            print(f'Current frame (video 2): {j}')
            ret, frame2 = video2.read()
            if not ret:
                print('Error reading video 2 frame')
                cv2.destroyAllWindows()
                break
            hashes2.append(imagehash.phash(Image.fromarray(frame2)))
        print('Comparing hashes between videos')
        for hash1 in hashes1:
            for hash2 in hashes2:
                hash_dif = hash_dif + (hash1 - hash2)
        return hash_dif, noc

    def _hash_trial_videos(self, video1, video2):
        '''Do an image hashing between two videos'''
        video1_frames = int(video1.get(cv2.CAP_PROP_FRAME_COUNT))
        video2_frames = int(video2.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Video 1 frames: {video1_frames}')
        print(f'Video 2 frames: {video2_frames}')

        hashes1 = []
        print ('Creating hashes for video 1')
        for i in range(video1_frames):
            print(f'Current frame (video 1): {i}')
            ret, frame1 = video1.read()
            if not ret:
                print('Error reading video 1 frame')
                cv2.destroyAllWindows()
                break
            hashes1.append(imagehash.phash(Image.fromarray(frame1)))

        print('Creating hashes for video 2')
        hashes2 = []
        for j in range(video2_frames):
            print(f'Current frame (video 2): {j}')
            ret, frame2 = video2.read()
            if not ret:
                print('Error reading video 2 frame')
                cv2.destroyAllWindows()
                break
            hashes2.append(imagehash.phash(Image.fromarray(frame2)))

        return hashes1, hashes2

    def _find_dissimilar_regions(self, hashes1, hashes2, window):
        '''Find the region of maximum dissimilarity given 2 lists of hashes and a sliding window (how many frames)'''
        start_frame_vid1 = 0
        start_frame_vid2 = 0
        max_hash_dif_vid1 = 0
        max_hash_dif_vid2 = 0
        hash_dif_vid1 = 0
        hash_dif_vid2 = 0

        for slider in range(0, len(hashes1) // window):
            print(f'Current start frame {slider * window}')
            hash_dif_vid1, hash_dif_vid2 = self._compare_hash_sets(hashes1[slider * window:(slider + 1) * window], hashes2[slider * window:(slider + 1) * window])

            print(f'Current hash diff (vid 1): {hash_dif_vid1}')
            print(f'Current hash diff (vid 2): {hash_dif_vid2}')
            if hash_dif_vid1 > max_hash_dif_vid1:
                max_hash_dif_vid1 = hash_dif_vid1
                start_frame_vid1 = slider * window

            if hash_dif_vid2 > max_hash_dif_vid2:
                max_hash_dif_vid2 = hash_dif_vid2
                start_frame_vid2 = slider * window

            print(f'Max hash diff (vid 1): {max_hash_dif_vid1}')
            print(f'Max hash diff (vid 2): {max_hash_dif_vid2}')

            print(f'Start frame (vid 1): {start_frame_vid1}')
            print(f'Start frame (vid 2): {start_frame_vid2}')

        return start_frame_vid1, start_frame_vid2

    def _compare_hash_sets(self, hashes1, hashes2):
        '''Compares two sets of image hashes to find dissimilarities'''
        hash1_dif = 0
        hash2_dif = 0

        print(f'Hash set 1 {hashes1[0]}')
        print(f'Hash set 2 {hashes2[0]}')
        # Compares all possible combinations of images
        for combination in combinations(hashes1, 2):
            hash1_dif = hash1_dif + (combination[0] - combination[1])

        for combination in combinations(hashes2, 2):
            hash2_dif = hash2_dif + (combination[0] - combination[1])

        return hash1_dif, hash2_dif
    