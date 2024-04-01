'''A Complete Set of User-Friendly Tools for DeepLabCut-XMAlab marker tracking'''
# Import packages
import math
import os
import warnings
from subprocess import PIPE, Popen

import cv2
import deeplabcut
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from ruamel.yaml import YAML

from analyzer import Analyzer
from network import Network
from xma_data_processor import XMADataProcessor


def create_new_project(working_dir=os.getcwd(), experimenter='NA', mode='2D'):
    '''Create a new xrommtools project'''
    if not os.path.exists(working_dir):
        os.mkdir(working_dir)
    dirs = ["trainingdata", "trials"]
    for folder in dirs:
        if not os.path.exists(f'{working_dir}/{folder}'):
            os.mkdir(f'{working_dir}/{folder}')

    # Create a fake video to pass into the deeplabcut workflow
    frame = np.zeros((480, 480, 3), np.uint8)
    out = cv2.VideoWriter(f'{working_dir}/dummy.avi',
                          cv2.VideoWriter_fourcc(*'DIVX'),
                          15,
                          (480,480))
    out.write(frame)
    out.release()

    # Create a new project
    yaml = YAML()
    task = os.path.basename(working_dir)
    path_config_file = deeplabcut.create_new_project(task, experimenter,
                                                    [os.path.join(working_dir,
                                                                  "dummy.avi")],
                                                    working_dir + os.path.sep,
                                                     copy_videos=True)

    if isinstance(path_config_file, str):
        template = f"""
    task: {task}
    experimenter: {experimenter}
    working_dir: {working_dir}
    path_config_file: {path_config_file}
    dataset_name: MyData
    nframes: 0
    maxiters: 150000
    tracking_threshold: 0.1 # Fraction of total frames included in training sample
    tracking_mode: 2D
    swapped_markers: false
    crossed_markers: false

# Image Processing Vars
    search_area: 15
    threshold: 8
    krad: 17
    gsigma: 10
    img_wt: 3.6
    blur_wt: -2.9
    gamma: 0.1

# Autocorrect() Testing Vars

    trial_name: your_trial_here
    cam: cam1
    frame_num: 1
    marker: your_marker_here
    test_autocorrect: false # Set to true if you want to see autocorrect's output in Jupyter

# Video Similarity Analysis Vars
    cam1s_are_the_same_view: true    
        """

        tmp = yaml.load(template)

        if mode == 'per_cam':
            task_2 = f'{task}_cam2'
            path_config_file_2 = deeplabcut.create_new_project(task_2,
                                                               experimenter,
                                                               [os.path.join(working_dir,
                                                                            "dummy.avi")],
                                                               working_dir
                                                               + os.path.sep,
                                                               copy_videos=True)
            tmp['path_config_file_2'] = path_config_file_2

        with open(f"{working_dir}/project_config.yaml", 'w') as config:
            yaml.dump(tmp, config)

        try:
            os.rmdir(path_config_file[:path_config_file.find("config")] + os.path.join("labeled-data","dummy"))
        except FileNotFoundError:
            pass

        try:
            os.remove(os.path.join(path_config_file[:path_config_file.find("config")], "videos", "dummy.avi"))
        except FileNotFoundError:
            pass
    
    os.remove(f'{working_dir}/dummy.avi')


def load_project(working_dir=os.getcwd()):
    '''Load an existing project (only used internally/in testing)'''
    # Open the config
    with open(os.path.join(working_dir, "project_config.yaml"), 'r') as config_file:
        yaml = YAML()
        project = yaml.load(config_file)

    experimenter = str(project['experimenter'])
    project['experimenter'] = experimenter
    if project['dataset_name'] == 'MyData':
        warnings.warn('Default project name in use', SyntaxWarning)

    training_data_path = os.path.join(project['working_dir'], "trainingdata")
    trial = [folder for folder in os.listdir(training_data_path) if os.path.isdir(os.path.join(training_data_path, folder)) and not folder.startswith('.')][0] 
    trial_path = os.path.join(training_data_path, trial)
    # Load trial CSV
    try:     
        trial_csv = pd.read_csv(os.path.join(trial_path, f'{trial}.csv'))
    except FileNotFoundError as e:
        print(f'Please make sure that your trainingdata 2DPoints csv file is named {trial}.csv')
        raise e

    # Give search_area a minimum of 10
    project['search_area'] = int(project['search_area'] + 0.5) if project['search_area'] >= 10 else 10

    # Drop untracked frames (all NaNs)
    trial_csv = trial_csv.dropna(how='all')

    # Make sure there aren't any partially tracked frames
    if trial_csv.isna().sum().sum() > 0:
        raise AttributeError(f'Detected {len(trial_csv) - len(trial_csv.dropna())} partially tracked frames. \
    Please ensure that all frames are completely tracked')

    # Check/set the default value for tracked frames
    if project['nframes'] <= 0:
        project['nframes'] = len(trial_csv)

    elif project['nframes'] != len(trial_csv):
        warnings.warn('Project nframes tracked does not match 2D Points file. \
        If this is intentional, ignore this message')

    data_processor = XMADataProcessor(config=project)
    # Check the current nframes against the threshold value * the number of frames in the cam1 video
    trial_path = os.path.join(training_data_path, trial)
    cam1_video_path = data_processor.find_cam_file(trial_path, "cam1")
    video = cv2.VideoCapture(cam1_video_path)

    if project['nframes'] < int(video.get(cv2.CAP_PROP_FRAME_COUNT)) * project['tracking_threshold']:
        tracking_threshold = project['tracking_threshold']
        warnings.warn(f'Project nframes is less than the recommended {tracking_threshold * 100}% of the total frames')

    # Check DLC bodyparts (marker names)
    default_bodyparts = ['bodypart1', 'bodypart2', 'bodypart3', 'objectA']

    path_to_trial = os.path.join(working_dir, 'trainingdata', trial)
    bodyparts = get_bodyparts_from_xma(
        project,
        path_to_trial,
        mode=project['tracking_mode'])
    
    dlc_config_loader = YAML()
    with open(project['path_config_file'], 'r') as dlc_config:
        dlc_yaml = dlc_config_loader.load(dlc_config)

    if dlc_yaml['bodyparts'] == default_bodyparts:
        dlc_yaml['bodyparts'] = bodyparts
    elif dlc_yaml['bodyparts'] != bodyparts:
        raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

    with open(project['path_config_file'], 'w') as dlc_config:
        yaml.dump(dlc_yaml, dlc_config)

    # Check DLC bodyparts (marker names) for config 2 if needed
    if project['tracking_mode'] == 'per_cam':
        dlc_config_loader = YAML()
        if 'path_config_file_2' not in project.keys(): 
            print('Couldn\'t find a reference to a second config file')
            print('Make sure you run create_new_project with mode=\'per_cam\'')
            raise KeyError('path_config_file_2')
        with open(project['path_config_file_2'], 'r') as dlc_config:
            dlc_yaml = dlc_config_loader.load(dlc_config)
        # Better conditional logic could definitely be had to reduce function calls here
        if dlc_yaml['bodyparts'] == default_bodyparts:
            dlc_yaml['bodyparts'] = get_bodyparts_from_xma(trial_path, 
                                                           project['tracking_mode'],
                                                           project['swapped_markers'],
                                                           project['crossed_markers'])

        elif dlc_yaml['bodyparts'] != get_bodyparts_from_xma(trial_path,
                                                             project['tracking_mode'],
                                                             project['swapped_markers'],
                                                             project['crossed_markers']):
            raise SyntaxError('XMAlab CSV marker names are different than DLC bodyparts.')

        with open(project['path_config_file_2'], 'w') as dlc_config:
            yaml.dump(dlc_yaml, dlc_config)

    # Check test_autocorrect params for defaults
    if project['test_autocorrect']:
        if project['trial_name'] == 'your_trial_here':
            raise SyntaxError('Please specify a trial to test autocorrect() with')
        if project['marker'] == 'your_marker_here':
            raise SyntaxError('Please specify a marker to test autocorrect() with')

    # Update changed attributes to match in the file
    with open(os.path.join(working_dir, 'project_config.yaml'), 'w') as file:
        yaml.dump(project, file)

    return project

def train_network(working_dir=os.getcwd()):
    '''Starts training the network using xrommtools-compatible data in the working directory.'''
    project = load_project(working_dir)
    network = Network(config=project)
    network.train()

def analyze_videos(working_dir=os.getcwd()):
    '''Analyze videos with a pre-existing network'''
    project = load_project(working_dir)
    analyzer = Analyzer(config=project)
    analyzer.analyze_videos()

def autocorrect_trial(working_dir=os.getcwd()): #try 0.05 also
    '''Do XMAlab-style autocorrect on the tracked beads'''
    # Open the config
    project = load_project(working_dir)

    # Error if trials directory is empty
    new_data_path = os.path.join(working_dir, 'trials')
    trials = [folder for folder in os.listdir(new_data_path) if os.path.isdir(os.path.join(new_data_path, folder)) and not folder.startswith('.')]
    if len(trials) <= 0:
        raise FileNotFoundError(f'Empty trials directory found. Please put trials to be analyzed after training into the {working_dir}/trials folder')

    # Establish project vars
    yaml = YAML()
    with open(project['path_config_file']) as dlc_config:
        dlc = yaml.load(dlc_config)

    iteration = dlc['iteration']

    # For each trial
    for trial in trials:
        # Find the appropriate pointsfile
        try:
            csv = pd.read_csv(new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-Predicted2DPoints.csv')
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find predicted 2D points file. Please check the it{iteration} folder for trial {trial}') from None
        out_name = new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-AutoCorrected2DPoints.csv'

        if project['test_autocorrect']:
            cams = [project['cam']]
        else:
            cams = ['cam1', 'cam2']

        # For each camera
        for cam in cams:
            csv = autocorrect_video(cam, trial, csv, project, new_data_path)

        # Print when autocorrect finishes
        if not project['test_autocorrect']:
            print(f'Done! Saving to {out_name}')
            csv.to_csv(out_name, index=False)

def autocorrect_video(cam, trial, csv, project, new_data_path):
    '''Run the autocorrect function on a single video within a single trial'''
    # Find the raw video
    video_path = new_data_path + '/' + trial + '/' + trial + '_' + cam + '.avi'
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f'Couldn\'t find a video at file path: {video_path}') from None

    if project['test_autocorrect']:
        video.set(1, project['frame_num'] - 1)
        ret, frame = video.read()
        if ret is False:
            raise IOError('Error reading video frame')
        path_to_trial = os.path.join(new_data_path, project['trial_name'])
        autocorrect_frame(path_to_trial, frame, project['cam'], project['frame_num'], csv, project)
        return csv

    # For each frame of video
    print(f'Total frames in video: {video.get(cv2.CAP_PROP_FRAME_COUNT)}')

    for frame_index in range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
        # Load frame
        print(f'Current Frame: {frame_index + 1}')
        ret, frame = video.read()
        if ret is False:
            raise IOError('Error reading video frame')
        csv = autocorrect_frame(f'{new_data_path}/{trial}', frame, cam, frame_index, csv, project)
    return csv

def autocorrect_frame(path_to_trial, frame, cam, frame_index, csv, project):
    '''Run the autocorrect function for a single frame (no output)'''
    # For each marker in the frame
    parts_unique = get_bodyparts_from_xma(project, path_to_trial, mode='2D')
    if project['test_autocorrect']:
        parts_unique = [project['marker']]
    else:
        parts_unique = get_bodyparts_from_xma(project, path_to_trial, mode='2D')
    for part in parts_unique:
        # Find point and offsets
        x_float = csv.loc[frame_index, part + '_' + cam + '_X']
        y_float = csv.loc[frame_index, part + '_' + cam + '_Y']
        x_start = int(x_float-project['search_area']+0.5)
        y_start = int(y_float-project['search_area']+0.5)
        x_end = int(x_float+project['search_area']+0.5)
        y_end = int(y_float+project['search_area']+0.5)

        subimage = frame[y_start:y_end, x_start:x_end]

        subimage_filtered = filter_image(subimage, project['krad'], project['gsigma'], project['img_wt'], project['blur_wt'], project['gamma'])

        subimage_float = subimage_filtered.astype(np.float32)
        radius = int(1.5 * 5 + 0.5) #5 might be too high
        sigma = radius * math.sqrt(2 * math.log(255)) - 1
        subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, 2 * radius + 1), sigma)

        subimage_diff = subimage_float-subimage_blurred
        subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)

        # Median
        subimage_median = cv2.medianBlur(subimage_diff, 3)

        # LUT
        subimage_median = filter_image(subimage_median, krad=3)

        # Thresholding
        subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
        min_val, _, _, _ = cv2.minMaxLoc(subimage_median)
        thres = 0.5 * min_val + 0.5 * np.mean(subimage_median) + project['threshold'] * 0.01 * 255
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

        if project['test_autocorrect']:
            print('Raw')
            show_crop(subimage, 15)

            print('Filtered')
            show_crop(subimage_filtered, 15)

            print(f'Blurred: {sigma}')
            show_crop(subimage_blurred, 15)

            print('Diff (Float - blurred)')
            show_crop(subimage_diff, 15)

            print('Median')
            show_crop(subimage_median, 15)

            print('Median filtered')
            show_crop(subimage_median, 15)

            print('Threshold')
            show_crop(subimage_threshold, 15)

            print('Gaussian')
            show_crop(subimage_threshold, 15)

            print('Best Contour')
            detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
            show_crop(subimage, 15, contours = [contours_im[best_index]], detected_marker = detected_center_im)

        # Save center of closest contour to CSV
        if best_index >= 0:
            detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
            csv.loc[frame_index, part + '_' + cam + '_X']  = detected_center[0]
            csv.loc[frame_index, part + '_' + cam + '_Y']  = detected_center[1]
    return csv

def filter_image(image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.10):
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

def show_crop(src, center, scale=5, contours=None, detected_marker=None):
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

def get_bodyparts_from_xma(project, path_to_trial, mode):
    '''Pull the names of the XMAlab markers from the 2Dpoints file'''
    data_processor = XMADataProcessor(config=project)
    return data_processor.get_bodyparts_from_xma(path_to_trial, mode)

def split_rgb(trial_path, codec='avc1'):
    '''Takes a RGB video with different grayscale data written to the R, G, and B channels and splits it back into its component source videos.'''
    trial_name = os.path.basename(os.path.normpath(trial_path))
    out_name = trial_name+'_split_'

    try:
        rgb_video = cv2.VideoCapture(f'{trial_path}/{trial_name}_rgb.avi')
    except FileNotFoundError as e:
        raise FileNotFoundError(f'Couldn\'t find video at {trial_path}/{trial_name}_rgb.avi') from e
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

def analyze_video_similarity_project(working_dir):
    '''Analyze all videos in a project and take their average similar. This is dangerous, as it will assume that all cam1/cam2 pairs match
    or don't match!'''
    project = load_project(working_dir)
    analyzer = Analyzer(config=project)
    return analyzer.analyze_video_similarity_project()

def analyze_video_similarity_trial(working_dir):
    '''Analyze the average similarity between trials using image hashing'''
    project = load_project(working_dir)
    analyzer = Analyzer(config=project)
    return analyzer.analyze_video_similarity_trial()

def get_max_dissimilarity_for_trial(trial_path, window):
    '''Calculate the dissimilarity within the trial given the frame sliding window.'''
    project = load_project()
    analyzer = Analyzer(config=project)
    return analyzer.get_max_dissimilarity_for_trial(trial_path, window)

def analyze_marker_similarity_project(working_dir):
    '''Analyze all videos in a project and get their average rhythmicity. This assumes that all cam1/2 pairs are either the same or different!'''
    project = load_project(working_dir)
    analyzer = Analyzer(config=project)
    return analyzer.analyze_marker_similarity_project()

def analyze_marker_similarity_trial(working_dir):
    '''Analyze marker similarity for a pair of trials. Returns the mean difference for paired marker positions (X - X, Y - Y for each marker)'''
    project = load_project(working_dir)
    analyzer = Analyzer(config=project)
    return analyzer.analyze_marker_similarity_trial()

def train_many_projects(parent_dir):
    '''Train and analyze multiple SDLC_XMALAB projects given a parent folder'''
    for folder in os.listdir(parent_dir):
        project_path = os.path.join(parent_dir, folder)
        if os.path.isdir(project_path):
            train_network(project_path)
            analyze_videos(project_path)
