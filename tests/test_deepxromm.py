'''Unit tests for XROMM-DLC'''
from pathlib import Path
import shutil
import unittest
from datetime import datetime as dt

import cv2
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from ruamel.yaml import YAML

from deepxromm import DeepXROMM


SAMPLE_FRAME = Path(__file__).parent / 'sample_frame.jpg'
SAMPLE_FRAME_INPUT = Path(__file__).parent / 'sample_frame_input.csv'
SAMPLE_AUTOCORRECT_OUTPUT = Path(__file__).parent / 'sample_autocorrect_output.csv'


class TestProjectCreation(unittest.TestCase):
    '''Tests behaviors related to XMA-DLC project creation'''
    @classmethod
    def setUpClass(cls):
        '''Create a sample project'''
        super(TestProjectCreation, cls).setUpClass()
        cls.project_dir = Path.cwd() / 'tmp'
        DeepXROMM.create_new_project(cls.project_dir)

    def test_project_creates_correct_folders(self):
        '''Do we have all of the correct folders?'''
        for folder in ["trainingdata", "trials"]:
            with self.subTest(folder=folder):
                self.assertTrue((self.project_dir / folder).exists())

    def test_project_creates_config_file(self):
        '''Do we have a project config?'''
        self.assertTrue((self.project_dir / 'project_config.yaml').exists())

    def test_project_config_has_these_variables(self):
        '''Can we access each of the variables that's supposed to be in the config?'''
        yaml = YAML()
        variables = ['task',
        'experimenter',
        'working_dir',
        'path_config_file',
        'dataset_name',
        'nframes',
        'maxiters',
        'tracking_threshold',
        'tracking_mode',
        'swapped_markers',
        'crossed_markers',
        'search_area',
        'threshold',
        'krad',
        'gsigma',
        'img_wt',
        'blur_wt',
        'gamma',
        'cam',
        'frame_num',
        'trial_name',
        'marker',
        'test_autocorrect',
        'cam1s_are_the_same_view']

        yaml = YAML()
        config_path = self.project_dir / 'project_config.yaml'
        with config_path.open() as config:
            project = yaml.load(config)
            for variable in variables:
                with self.subTest(i=variable):
                    self.assertIsNotNone(project[variable])

    @classmethod
    def tearDownClass(cls):
        '''Remove the created temp project'''
        super(TestProjectCreation, cls).tearDownClass()
        shutil.rmtree(cls.project_dir)

class TestDefaultsPerformance(unittest.TestCase):
    '''Test that the config will still be configured properly if the user only provides XMAlab input'''
    def setUp(self):
        '''Create a sample project where the user only inputs XMAlab data'''
        self.working_dir = Path.cwd() / 'tmp'
        self.config = self.working_dir / 'project_config.yaml'
        self.deepxromm = DeepXROMM.create_new_project(self.working_dir)
        frame = cv2.imread(str(SAMPLE_FRAME))

        # Make a trial directory
        (self.working_dir / 'trainingdata/dummy').mkdir(parents=True, exist_ok=True)

        # Cam 1
        video_path_1 = self.working_dir / 'trainingdata/dummy/dummy_cam1.avi'
        out = cv2.VideoWriter(str(video_path_1), cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024, 512))
        out.write(frame)
        out.release()

        # Cam 2
        video_path_2 = self.working_dir / 'trainingdata/dummy/dummy_cam2.avi'
        out = cv2.VideoWriter(str(video_path_2), cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024, 512))
        out.write(frame)
        out.release()

        # CSV
        df = pd.DataFrame({'foo_cam1_X': 0, 'foo_cam1_Y': 0, 'foo_cam2_X': 0, 'foo_cam2_Y': 0,
                        'bar_cam1_X': 0, 'bar_cam1_Y': 0, 'bar_cam2_X': 0, 'bar_cam2_Y': 0,
                        'baz_cam1_X': 0, 'baz_cam1_Y': 0, 'baz_cam2_X': 0, 'baz_cam2_Y': 0}, index=[1])
        csv_path = self.working_dir / 'trainingdata/dummy/dummy.csv'
        df.to_csv(str(csv_path), index=False)
        cv2.destroyAllWindows()

    def test_can_find_frames_from_csv(self):
        '''Can I accurately find the number of frames in the video if the user doesn't tell me?'''
        print(list(self.working_dir.iterdir()))
        deepxromm = DeepXROMM.load_project(self.working_dir)
        config = deepxromm.config
        self.assertEqual(config['nframes'], 1, msg=f"Actual nframes: {config['nframes']}")

    def test_analyze_errors_if_no_folders_in_trials_dir(self):
        '''If there are no trials to analyze, do we return an error?'''
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.analyze_videos()

    def test_autocorrect_errors_if_no_folders_in_trials_dir(self):
        '''If there are no trials to autocorrect, do we return an error?'''
        with self.assertRaises(FileNotFoundError):
            self.deepxromm.autocorrect_trial()

    def test_warn_users_if_nframes_doesnt_match_csv(self):
        '''If the number of frames in the CSV doesn't match the number of frames specified, do I issue a warning?'''
        yaml = YAML()
        with self.config.open() as config:
            tmp = yaml.load(config)

        # Modify the number of frames (similar to how a user would)
        tmp['nframes'] = 2
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            DeepXROMM.load_project(self.working_dir)

    def test_yaml_file_updates_nframes_after_load_if_frames_is_0(self):
        '''If the user doesn't specify how many frames they want analyzed,
        does their YAML file get updated with how many are in the CSV?'''
        yaml = YAML()
        DeepXROMM.load_project(self.working_dir)
        with self.config.open() as config:
            tmp = yaml.load(config)
        self.assertEqual(tmp['nframes'], 1, msg=f"Actual nframes: {tmp['nframes']}")

    def test_warn_if_user_has_tracked_less_than_threshold_frames(self):
        '''If the user has tracked less than threshold % of their trial,
        do I give them a warning?'''
        DeepXROMM.load_project(self.working_dir)

        # Increase the number of frames in the video to 100 so I can test this
        frame = cv2.imread(str(SAMPLE_FRAME))
        video_path = self.working_dir / 'trainingdata/dummy/dummy_cam1.avi'
        out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024, 512))
        for _ in range(100):
            out.write(frame)
        out.release()

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            DeepXROMM.load_project(self.working_dir)

    def test_bodyparts_add_from_csv_if_not_defined(self):
        '''If the user hasn't specified the bodyparts from their trial, we can pull them from the CSV'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        DeepXROMM.load_project(self.working_dir)
        config_path = Path(self.working_dir) / f'tmp-NA-{date}' / 'config.yaml'

        with config_path.open() as dlc_config:
            config_obj = yaml.load(dlc_config)
        self.assertEqual(config_obj['bodyparts'], ['foo', 'bar', 'baz'])

    def test_bodyparts_add_from_csv_in_3d(self):
        '''If the user wants to do 3D tracking, we output the desired list of bodyparts'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        with self.config.open('r') as config:
            tmp = yaml.load(config)
        tmp['tracking_mode'] = 'rgb'
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f'tmp-NA-{date}' / 'config.yaml'
        yaml = YAML()
        with path_to_config.open('r') as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(config_obj['bodyparts'], ['foo_cam1', 'foo_cam2', 'bar_cam1', 'bar_cam2', 'baz_cam1', 'baz_cam2'])

    def test_bodyparts_add_synthetic(self):
        '''Can we add swapped markers?'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open('r') as config:
            tmp = yaml.load(config)
        tmp['tracking_mode'] = 'rgb'
        tmp['swapped_markers'] = True
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f'tmp-NA-{date}' / 'config.yaml'
        yaml = YAML()
        with path_to_config.open('r') as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(config_obj['bodyparts'],
                        ['foo_cam1',
                         'foo_cam2',
                         'bar_cam1',
                         'bar_cam2',
                         'baz_cam1',
                         'baz_cam2',
                         'sw_foo_cam1',
                         'sw_foo_cam2',
                         'sw_bar_cam1',
                         'sw_bar_cam2',
                         'sw_baz_cam1',
                         'sw_baz_cam2'])

    def test_bodyparts_add_crossed(self):
        '''Can we add crossed markers?'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open('r') as config:
            tmp = yaml.load(config)
        tmp['tracking_mode'] = 'rgb'
        tmp['crossed_markers'] = True
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f'tmp-NA-{date}' / 'config.yaml'
        yaml = YAML()
        with path_to_config.open('r') as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(config_obj['bodyparts'], ['foo_cam1', 'foo_cam2', 'bar_cam1', 'bar_cam2', 'baz_cam1', 'baz_cam2', 'cx_foo_cam1x2', 'cx_bar_cam1x2', 'cx_baz_cam1x2'])

    def test_bodyparts_add_synthetic_and_crossed(self):
        '''Can we add both swapped and crossed markers?'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")

        with self.config.open('r') as config:
            tmp = yaml.load(config)
        tmp['tracking_mode'] = 'rgb'
        tmp['swapped_markers'] = True
        tmp['crossed_markers'] = True
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)
        DeepXROMM.load_project(self.working_dir)

        path_to_config = self.working_dir / f'tmp-NA-{date}' / 'config.yaml'
        yaml = YAML()
        with path_to_config.open('r') as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(config_obj['bodyparts'],
                        ['foo_cam1',
                         'foo_cam2',
                         'bar_cam1',
                         'bar_cam2',
                         'baz_cam1',
                         'baz_cam2',
                         'sw_foo_cam1',
                         'sw_foo_cam2',
                         'sw_bar_cam1',
                         'sw_bar_cam2',
                         'sw_baz_cam1',
                         'sw_baz_cam2',
                         'cx_foo_cam1x2',
                         'cx_bar_cam1x2',
                         'cx_baz_cam1x2'])

    def test_bodyparts_error_if_different_from_csv(self):
        '''If the user specifies different bodyparts than their CSV, raise an error'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        path_to_config = self.working_dir / f'tmp-NA-{date}' / 'config.yaml'
        yaml = YAML()
        with path_to_config.open('r') as dlc_config:
            config_dlc = yaml.load(dlc_config)

        config_dlc['bodyparts'] = ['foo', 'bar']

        with path_to_config.open('w') as dlc_config:
            yaml.dump(config_dlc, dlc_config)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def test_autocorrect_error_if_trial_not_set(self):
        '''If the user doesn't specify a trial to test autocorrect with, do we error?'''
        yaml = YAML()
        with self.config.open('r') as config:
            tmp = yaml.load(config)

        tmp['test_autocorrect'] = True
        tmp['marker'] = 'foo'
        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def test_autocorrect_error_if_marker_not_set(self):
        '''If the user doesn't specify a marker to test autocorrect with, do we error?'''
        yaml = YAML()
        
        with self.config.open('r') as config:
            tmp = yaml.load(config)

        tmp['test_autocorrect'] = True
        tmp['trial_name'] = 'test'

        with self.config.open('w') as fp:
            yaml.dump(tmp, fp)

        with self.assertRaises(SyntaxError):
            DeepXROMM.load_project(self.working_dir)

    def tearDown(self):
        '''Remove the created temp project'''
        shutil.rmtree(self.working_dir)

class TestSampleTrial(unittest.TestCase):
    '''Test function behaviors using a frame from an actual trial'''
    def setUp(self):
        '''Create trial'''
        self.working_dir = Path.cwd() / 'tmp'
        self.deepxromm = DeepXROMM.create_new_project(self.working_dir)
        frame = cv2.imread(str(SAMPLE_FRAME))

        # Make a trial directory
        trial_dir = self.working_dir / 'trainingdata/test'
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Move sample frame input to trainingdata
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(trial_dir / 'test.csv'))

        # Cam 1 and Cam 2 trainingdata setup
        for cam in ['cam1', 'cam2']:
            video_path = trial_dir / f'dummy_{cam}.avi'
            out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024, 512))
            out.write(frame)
            out.release()

        # Move sample frame input to trials (it0 and trials)
        trials_dir = self.working_dir / 'trials/test/it0'
        trials_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(self.working_dir / 'trials/test/test.csv'))
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(trials_dir / 'test-Predicted2DPoints.csv'))

        # Cam 1 and Cam 2 trials setup
        for cam in ['cam1', 'cam2']:
            video_path = self.working_dir / f'trials/test/test_{cam}.avi'
            out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024, 512))
            out.write(frame)
            out.release()

        cv2.destroyAllWindows()

    def test_autocorrect_is_working(self):
        '''Make sure that autocorrect still works properly after making changes'''
        # Run autocorrect on the sample frame
        self.deepxromm.autocorrect_trial()

        # Load CSVs
        function_output_path = self.working_dir / 'trials/test/it0/test-AutoCorrected2DPoints.csv'
        function_output = pd.read_csv(str(function_output_path), dtype='float64')
        sample_output = pd.read_csv(str(SAMPLE_AUTOCORRECT_OUTPUT), dtype='float64')

        # Drop cam2 markers and check for changes
        columns_to_drop = function_output.columns[function_output.columns.str.contains('cam2', case=False)]
        function_output.drop(columns_to_drop, axis=1, inplace=True)
        function_output = function_output.round(6)
        sample_output = sample_output.round(6)

        try:
            # Use assert_frame_equal to check if the data frames are the same
            assert_frame_equal(function_output, sample_output, check_exact=False, rtol=1e-6, atol=1e-6)
        except AssertionError as e:
            print(f"Autocorrector diff: {function_output.compare(sample_output)}")
            raise e

    def test_image_hashing_identical_trials_returns_0(self):
        '''Make sure the image hashing function is working properly'''
        # Create an identical second trial
        frame_path = self.working_dir.parent / 'sample_frame.jpg'
        frame = cv2.imread(str(frame_path))
        trial_dir = self.working_dir / 'trials/test2_same'
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create videos
        for cam in ['cam1', 'cam2']:
            video_path = trial_dir / f'test2_same_{cam}.avi'
            out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024, 512))
            out.write(frame)
            out.release()

        # Do similarity comparison
        similarity = self.deepxromm.analyze_video_similarity_project()
        # Since both videos are the same, the image similarity output should be 0
        self.assertEqual(sum(similarity.values()), 0)

    def test_image_hashing_different_trials_returns_nonzero(self):
        '''Image hashing different videos returns nonzero answer'''
        # Create a different second trial
        frame = np.zeros((480, 480, 3), np.uint8)
        trial_dir = self.working_dir / 'trials/test2_diff'
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Create videos
        for cam in ['cam1', 'cam2']:
            video_path = trial_dir / f'test2_diff_{cam}.avi'
            out = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'DIVX'), 30, (480, 480))
            out.write(frame)
            out.release()

        # Do similarity comparison
        similarity = self.deepxromm.analyze_video_similarity_project()
        # Since the videos are different, should return nonzero answer
        self.assertNotEqual(sum(similarity.values()), 0)

    def test_marker_similarity_returns_0_if_identical(self):
        '''Check that identical data has a similarity value of 0'''
        # Move sample data into test trial
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(self.working_dir / 'trials/test/test.csv'))

        # Move sample data into test2 trial
        (self.working_dir / 'trials/test2').mkdir(parents=True, exist_ok=True)
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(self.working_dir / 'trials/test2/test2.csv'))

        # Do cross-correlation
        marker_similarity = self.deepxromm.analyze_marker_similarity_project()
        self.assertEqual(sum(marker_similarity.values()), 0)

    def test_marker_similarity_returns_not_0_if_different(self):
        '''Check that different data has a similarity value of not 0'''
        # Move sample data into test trial
        shutil.copy(str(SAMPLE_FRAME_INPUT), str(self.working_dir / 'trials/test/test.csv'))

        # Move autocorrect data into test2 trial
        (self.working_dir / 'trials/test2').mkdir(parents=True, exist_ok=True)
        shutil.copy(str(SAMPLE_AUTOCORRECT_OUTPUT), str(self.working_dir / 'trials/test2/test2.csv'))

        # Do cross-correlation
        marker_similarity = self.deepxromm.analyze_marker_similarity_project()
        self.assertNotEqual(sum(marker_similarity.values()), 0)

    def tearDown(self):
        '''Remove the created temp project'''
        project_path = Path.cwd() / 'tmp'
        shutil.rmtree(project_path)

if __name__ == "__main__":
    unittest.main()
