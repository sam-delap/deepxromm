'''Unit tests for XROMM-DLC'''
import unittest
import os
import shutil
from datetime import datetime as dt
import pandas as pd
import cv2
from ruamel.yaml import YAML
import samtools as sam


class TestProjectCreation(unittest.TestCase):
    '''Tests behaviors related to XMA-DLC project creation'''
    @classmethod
    def setUpClass(cls):
        '''Create a sample project'''
        super(TestProjectCreation, cls).setUpClass()
        sam.create_new_project(os.path.join(os.getcwd(), 'tmp'))

    def test_project_creates_correct_folders(self):
        '''Do we have all of the correct folders?'''
        for folder in ["trainingdata", "trials", "XMA_files"]:
            with self.subTest(i=folder):
                self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'tmp', folder)))

    def test_project_creates_config_file(self):
        '''Do we have a project config?'''
        self.assertTrue(os.path.exists(os.path.join(os.getcwd(), 'tmp/project_config.yaml')))

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
        'search_area',
        'threshold',
        'krad',
        'gsigma',
        'img_wt',
        'blur_wt',
        'gamma']


        with open(os.path.join(os.getcwd(), 'tmp/project_config.yaml')) as config:
            project = yaml.load(config)
            for variable in variables:
                with self.subTest(i=variable):
                    self.assertIsNotNone(project[variable])

    @classmethod
    def tearDownClass(cls):
        '''Remove the created temp project'''
        super(TestProjectCreation, cls).tearDownClass()
        shutil.rmtree(os.path.join(os.getcwd(), 'tmp'))

class TestDefaultsPerformance(unittest.TestCase):
    '''Test that the config will still be configured properly if the user only provides XMAlab input'''
    def setUp(self):
        '''Create a sample project where the user only inputs XMAlab data'''
        self.working_dir = os.path.join(os.getcwd(), 'tmp')
        sam.create_new_project(self.working_dir)
        frame = cv2.imread('sample_frame.jpg')

        # Make a trial directory
        os.mkdir(os.path.join(self.working_dir, 'trainingdata/dummy'))

        # Cam 1
        out = cv2.VideoWriter('tmp/trainingdata/dummy/dummy_cam1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024,512))
        out.write(frame)
        out.release()

        # Cam 2
        out = cv2.VideoWriter('tmp/trainingdata/dummy/dummy_cam2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024,512))
        out.write(frame)
        out.release()

        # CSV
        df = pd.DataFrame({'foo_cam1_X': 0,
        'foo_cam1_Y': 0,
        'foo_cam2_X': 0,
        'foo_cam2_Y': 0,
        'bar_cam1_X': 0,
        'bar_cam1_Y': 0,
        'bar_cam2_X': 0,
        'bar_cam2_Y': 0,
        'baz_cam1_X': 0,
        'baz_cam1_Y': 0,
        'baz_cam2_X': 0,
        'baz_cam2_Y': 0
        }, index=[1])

        df.to_csv('tmp/trainingdata/dummy/dummy.csv', index=False)

    def test_can_find_frames_from_csv(self):
        '''Can I accurately find the number of frames in the video if the user doesn't tell me?'''
        project = sam.load_project(self.working_dir)
        self.assertEqual(project['nframes'], 1, msg=f"Actual nframes: {project['nframes']}")

    def test_analyze_errors_if_no_folders_in_trials_dir(self):
        '''If there are no trials to analyze, do we return an error?'''
        with self.assertRaises(FileNotFoundError):
            sam.analyze_videos(self.working_dir)

    def test_autocorrect_errors_if_no_folders_in_trials_dir(self):
        '''If there are no trials to autocorrect, do we return an error?'''
        with self.assertRaises(FileNotFoundError):
            sam.autocorrect_trial(self.working_dir)

    def test_warn_users_if_nframes_doesnt_match_csv(self):
        '''If the number of frames in the CSV doesn't match the number of frames specified, do I issue a warning?'''
        yaml = YAML()

        # Modify the number of frames (similar to how a user would)
        with open(os.path.join(self.working_dir, 'project_config.yaml'), 'r') as config:
            tmp = yaml.load(config)
        tmp['nframes'] = 2
        with open(os.path.join(self.working_dir, 'project_config.yaml'), 'w') as fp:
            yaml.dump(tmp, fp)

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            sam.load_project(self.working_dir)

    def test_yaml_file_updates_nframes_after_load_if_frames_is_0(self):
        '''If the user doesn't specify how many frames they want analyzed,
        does their YAML file get updated with how many are in the CSV?'''
        yaml = YAML()

        sam.load_project(self.working_dir)
        with open(os.path.join(self.working_dir, 'project_config.yaml'), 'r') as config:
            tmp = yaml.load(config)

        self.assertEqual(tmp['nframes'], 1, msg=f"Actual nframes: {tmp['nframes']}")

    def test_warn_if_user_has_tracked_less_than_threshold_frames(self):
        '''If the user has tracked less than threshold % of their trial,
        do I give them a warning?'''
        sam.load_project(self.working_dir)

        # Increase the number of frames in the video to 100 so I can test this
        frame = cv2.imread('sample_frame.jpg')
        out = cv2.VideoWriter('tmp/trainingdata/dummy/dummy_cam1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (1024,512))

        for _ in range(100):
            out.write(frame)

        out.release()

        # Check that the user is warned
        with self.assertWarns(UserWarning):
            sam.load_project(self.working_dir)

    def test_bodyparts_add_from_csv_if_not_defined(self):
        '''If the user hasn't specified the bodyparts from their trial,
        we can pull them from the CSV'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        sam.load_project(self.working_dir)
        path_to_config = '/tmp-NA-' + date + '/config.yaml'

        with open(self.working_dir + path_to_config) as dlc_config:
            config_obj = yaml.load(dlc_config)

        self.assertEqual(config_obj['bodyparts'], ['foo', 'bar', 'baz'])

    def test_bodyparts_error_if_different_from_csv(self):
        '''If the user specifies different bodyparts than their CSV, raise an error'''
        yaml = YAML()
        date = dt.today().strftime("%Y-%m-%d")
        path_to_config = '/tmp-NA-' + date + '/config.yaml'

        with open(self.working_dir + path_to_config, 'r') as dlc_config:
            config_dlc = yaml.load(dlc_config)

        config_dlc['bodyparts'] = ['foo', 'bar']

        with open(self.working_dir + path_to_config, 'w') as dlc_config:
            yaml.dump(config_dlc, dlc_config)

        with self.assertRaises(SyntaxError):
            sam.load_project(self.working_dir)

    def tearDown(self):
        '''Remove the created temp project'''
        shutil.rmtree(os.path.join(os.getcwd(), 'tmp'))

class TestSampleTrial(unittest.TestCase):
    '''Test function behaviors using a frame from an actual trial'''
    def setUp(self):
        '''Create trial'''
        self.working_dir = os.path.join(os.getcwd(), 'tmp')
        sam.create_new_project(self.working_dir)
        frame = cv2.imread('sample_frame.jpg')

        # Make a trial directory
        os.mkdir(os.path.join(self.working_dir, 'trainingdata/test'))

        # Move sample frame input to trainingdata
        shutil.copy('sample_frame_input.csv', f'{self.working_dir}/trainingdata/test/test.csv')

        # Cam 1 trainingdata
        out = cv2.VideoWriter('tmp/trainingdata/test/test_cam1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024,512))
        out.write(frame)
        out.release()

        # Cam 2 trainingdata
        out = cv2.VideoWriter('tmp/trainingdata/test/test_cam2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024,512))
        out.write(frame)
        out.release()

        # Move sample frame input to trials (it0 and trials)
        os.makedirs(f'{self.working_dir}/trials/test/it0', exist_ok=True)
        shutil.copy('sample_frame_input.csv', f'{self.working_dir}/trials/test/test.csv')
        shutil.copy('sample_frame_input.csv', f'{self.working_dir}/trials/test/it0/test-Predicted2DPoints.csv')

        # Cam 1 trials
        out = cv2.VideoWriter('tmp/trials/test/test_cam1.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024,512))
        out.write(frame)
        out.release()

        # Cam 2 trials
        out = cv2.VideoWriter('tmp/trials/test/test_cam2.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, (1024,512))
        out.write(frame)
        out.release()

    def test_autocorrect_is_working(self):
        '''Make sure that autocorrect still works properly after making changes'''
        # Run autocorrect on the sample frame
        sam.autocorrect_trial(self.working_dir)

        # Load CSVs
        function_output = pd.read_csv(f'{self.working_dir}/trials/test/it0/test-AutoCorrected2DPoints.csv', dtype='float64')
        sample_output = pd.read_csv('sample_autocorrect_output.csv', dtype='float64')

        # Drop cam2 markers
        columns_to_drop = function_output.columns[function_output.columns.str.contains('cam2', case = False)]
        function_output.drop(columns_to_drop, axis = 1, inplace = True)

        # Round the pixel measurements to the nearest millionth (pixel measurements get a bit imprecise beyond this)
        function_output = function_output.round(6)
        sample_output = sample_output.round(6)

        # Make sure the output hasn't changed
        self.assertTrue(function_output.equals(sample_output))

    def tearDown(self):
        '''Remove the created temp project'''
        shutil.rmtree(os.path.join(os.getcwd(), 'tmp'))

if __name__ == "__main__":
    unittest.main()
