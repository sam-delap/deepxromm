# Import packages
import os
import deeplabcut
from deeplabcut.utils import xrommtools
import pandas as pd
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np
from ruamel.yaml import YAML

def create_new_project(working_dir=os.getcwd(), experimenter='NA'):
    '''Create a new xrommtools project'''
    try:
        os.chdir(working_dir)
    except FileNotFoundError:
        os.mkdir(working_dir)
        os.chdir(working_dir)
    dirs = ["trainingdata", "trials", "XMA_files"]
    for dir in dirs:
        try:
            os.mkdir(dir)
        except FileExistsError:
            continue

    config = open("project_config.yaml", 'w')

    # Create a fake video to pass into the deeplabcut workflow
    frame = np.zeros((480, 480, 3), np.uint8)
    out = cv2.VideoWriter('dummy.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, (480,480))
    out.write(frame)
    out.release()

    # Create a new project
    yaml = YAML()
    if '\\' in working_dir:
        task = working_dir.split("\\")[len(working_dir.split("\\")) - 1]
    else:
        task = working_dir.split('/')[len(working_dir.split('/')) - 1]
    
    path_config_file = deeplabcut.create_new_project(task, experimenter, [working_dir + "\\dummy.avi"], working_dir + "\\", copy_videos=True)
    template = f"""
    task: {task}
    experimenter: {experimenter}
    working_dir: {working_dir}
    path_config_file: {path_config_file}
    dataset_name: MyData
    nframes: 0
    """

    tmp = yaml.load(template)

    yaml.dump(tmp, config)
    config.close()

    try:
        os.rmdir(path_config_file[:path_config_file.find("config")] + "labeled-data\\dummy")
    except FileNotFoundError:
        pass

    try:
        os.remove(path_config_file[:path_config_file.find("config")] + "\\videos\\dummy.avi")
    except FileNotFoundError:
        pass

    try:
        os.remove("dummy.avi")
    except FileNotFoundError:
        pass

def train_network(working_dir=os.getcwd()):
    '''Start training xrommtools-compatible data'''
    # Open the config
    try:
        config_file = open(working_dir + "\\project_config.yaml", 'r')
    except FileNotFoundError:
        raise FileNotFoundError('Make sure that the current directory has a project already created in it.')
    yaml = YAML()
    project = yaml.load(config_file)

    # Establish project vars
    path_config_file = project['path_config_file']
    data_path = working_dir + "/trainingdata"
    dataset_name = project['dataset_name']
    experimenter = str(project['experimenter'])
    nframes = project['nframes']
    maxiters = 150000 # ADD TO CONFIG

    # ADD ABILITY TO PULL NFRAMES FROM VIDEO
    # ADD ABILITY TO CHECK VIDEO NFRAMES AGAINST CSV, DROPNAS 
    # ADD ABILITY TO PULL BODYPARTS FROM CSV
    if dataset_name == 'MyData':
        raise Exception("Please specify a name for this dataset in the config file")
    if nframes == 0:
        raise Exception("Please specify the number of frames in the training dataset")

    try:
        xrommtools.xma_to_dlc(path_config_file, data_path, dataset_name, experimenter, nframes)
    except UnboundLocalError:
        pass
    deeplabcut.create_training_dataset(path_config_file)
    deeplabcut.train_network(path_config_file, maxiters=maxiters)

def analyze_videos(working_dir=os.getcwd()):
    '''Analyze videos with a pre-existing network'''
    # Open the config
    try:
        config_file = open(working_dir + "/project_config.yaml", 'r')
    except FileNotFoundError:
        raise FileNotFoundError('Make sure that the current directory has a project already created in it.')
    yaml = YAML()
    project = yaml.load(config_file)

    
    # Establish project vars
    path_config_file = project['path_config_file']
    new_data_path = working_dir + "/trials"
    try:
        dlc_config = open(path_config_file)
    except FileNotFoundError:
        raise FileNotFoundError('Oops! Looks like there\'s no deeplabcut config file inside of your deeplabcut directory.')
    dlc = yaml.load(dlc_config)
    iteration = dlc['iteration']

    xrommtools.analyze_xromm_videos(path_config_file, new_data_path, iteration)

def autocorrect(working_dir,likelihood_cutoff=0.01, search_area=15, mask_size=5, threshold=8): #try 0.05 also
    '''Do XMAlab-style autocorrect on the tracked beads'''
    # Open the config
    try:
        config_file = open(working_dir + "\\project_config.yaml", 'r')
    except FileNotFoundError:
        raise FileNotFoundError('Make sure that the current directory has a project already created in it.')
    yaml = YAML()
    project = yaml.load(config_file)
    
    # Establish project vars
    path_config_file = project['path_config_file']
    new_data_path = working_dir + "/trials"
    try:
        dlc_config = open(path_config_file)
    except FileNotFoundError:
        raise FileNotFoundError('Oops! Looks like there\'s no deeplabcut config file inside of your deeplabcut directory.')
    dlc = yaml.load(dlc_config)
    iteration = dlc['iteration']
    search_area = int(search_area + 0.5) if search_area >= 10 else 10

    # For each trial
    for trial in os.listdir(new_data_path):
        # Find the appropriate pointsfile
        try:
            hdf = pd.read_hdf(new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-Predicted2DPoints.h5')
        except FileNotFoundError:
            raise FileNotFoundError(f'Could not find predicted 2D points file. Please check the it{iteration} folder for trial {trial}')
        out_name = new_data_path + '/' + trial + '/' + 'it' + str(iteration) + '/' + trial + '-AutoCorrected2DPoints.csv'
        # For each camera
        for cam in ['cam1','cam2']:
            # Find the raw video
            try:
                video = cv2.VideoCapture(new_data_path + '/' + trial + '/' + trial + '_' + cam + '.avi')
            except FileNotFoundError:
                raise FileNotFoundError(f'Please make sure that your {cam} video file is named {trial}_{cam}.avi')
            # For each frame of video
            print(video.get(cv2.CAP_PROP_FRAME_COUNT)) # DEBUG
             # DEBUG switch to a single frame
            for frame_index in [0]: #range(int(video.get(cv2.CAP_PROP_FRAME_COUNT))):
                # Load frame
                ret, frame = video.read()
                if ret == False:
                    raise IOError('Error reading video frame')
                cv2.imwrite('sample_frame.jpg', frame)
                frame = filter_image(frame, krad=10)

                # For each marker in the frame
                parts_unique = getBodypartsFromXmaExport(working_dir)
                for part in parts_unique:
                    # Find point and offsets
                    x_float = hdf.loc[frame_index, part + '_' + cam + '_X']
                    y_float = hdf.loc[frame_index, part + '_' + cam + '_Y']
                    x_start = int(x_float-search_area+0.5)
                    y_start = int(y_float-search_area+0.5)
                    x_end = int(x_float+search_area+0.5)
                    y_end = int(y_float+search_area+0.5)

                    # Preprocessing
                    # Blurred image
                    radius = int(1.5 * 2 + 0.5) #5 might be too high
                    sigma = radius * math.sqrt(2 * math.log(255)) - 1
                    subimage_blurred = cv2.GaussianBlur(subimage_float, (2 * radius + 1, radius + 1), sigma)
                    # Subtract Background
                    subimage_diff = subimage_float-subimage_blurred
                    subimage_diff = cv2.normalize(subimage_diff, None, 0,255,cv2.NORM_MINMAX).astype(np.uint8)
                    # Median
                    subimage_median = cv2.medianBlur(subimage_diff, 3)
                    # LUT
                    subimage_median = filter_image(subimage_median, krad=3)
                    # Crop image to marker vicinity
                    subimage = frame[y_start:y_end, x_start:x_end]

                    # Convert To float
                    subimage_float = subimage.astype(np.float32)

                    # Convert to grayscale
                    gray = cv2.cvtColor(subimage_float, cv2.COLOR_BGR2GRAY)
                    
                    # Edge detection
                    edges = cv2.Canny(gray, threshold1=30, threshold2=100)

                    # Gaussian blur
                    gaussian = cv2.GaussianBlur(edges, (3,3), 1.3)

                    # Thresholding
                    ret, subimage_threshold =  cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                    
                    

                    # # Thresholding
                    # # Canny
                    # subimage_median = cv2.cvtColor(subimage_median, cv2.COLOR_BGR2GRAY)
                    # minVal, _, _, _ = cv2.minMaxLoc(subimage_median)
                    # thres = 0.5 * minVal + 0.5 * np.mean(subimage_median) + threshold * 0.01 * 255
                    # edges = cv2.Canny(subimage_median, threshold1=thres, threshold2=255)
                    
                    # # ret, subimage_threshold =  cv2.threshold(subimage_median, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

                    # # # Gaussian blur
                    # # subimage_gaussthresh = cv2.GaussianBlur(subimage_threshold, (3,3), 1.3)

                    # # # Thresholding
                    # # minVal, _, _, _ = cv2.minMaxLoc(subimage_gaussthresh)
                    # # thres = 0.5 * minVal + 0.5 * np.mean(subimage_gaussthresh) + (threshold -2) * 0.01 * 255
                    # # ret, subimage_threshold2 =  cv2.threshold(subimage_gaussthresh, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    
                    # # # Gaussian blur
                    # # subimage_gaussthresh2 = cv2.GaussianBlur(subimage_threshold2, (3,3), 1.3)

                    # Find contours
                    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(x_start,y_start))
                    # DEBUG contours
                    # print("Detected "+str(len(contours))+" contours in "+str(search_area)+"*"+str(search_area)+" neighborhood of marker "+part+' in Camera '+cam[-1])
                    contours_im = [contour-[x_start, y_start] for contour in contours]

                    # Find closest contour
                    dist = 1000
                    best_index = -1
                    detected_centers = {}
                    for i in range(len(contours)):
                        detected_center, circle_radius = cv2.minEnclosingCircle(contours[i])
                        distTmp = math.sqrt((x_float - detected_center[0])**2 + (y_float - detected_center[1])**2)
                        detected_centers[round(distTmp, 4)] = detected_center
                        if distTmp < dist:
                            best_index = i
                            dist = distTmp
                    if best_index >= 0:
                        detected_center, _ = cv2.minEnclosingCircle(contours[best_index])
                        detected_center_im, _ = cv2.minEnclosingCircle(contours_im[best_index])
                        show_crop(subimage, center=search_area, contours = [contours_im[best_index]], detected_marker = detected_center_im)
                        show_crop(gray, center=search_area, contours=contours_im,detected_marker = detected_center_im)
                        show_crop(edges, center=search_area, contours=contours_im,detected_marker = detected_center_im)
                        show_crop(gaussian, center=search_area, contours = [contours_im[best_index]], detected_marker = detected_center_im)
                        show_crop(subimage_threshold, center=search_area, contours = [contours_im[best_index]], detected_marker = detected_center_im)
                        hdf.loc[frame_index, part + '_' + cam + '_X']  = detected_center[0]
                        hdf.loc[frame_index, part + '_' + cam + '_Y']  = detected_center[1]   
                
            print('done! saving...')
            # ADD delete first column (index column) before saving as a CSV
            hdf.to_csv(out_name, index=False)

def filter_image(image, krad=17, gsigma=10, img_wt=3.6, blur_wt=-2.9, gamma=0.30):
    '''Filter the image to make it easier for python to see'''
    krad = krad*2+1
    image_blur = cv2.GaussianBlur(image, (krad, krad), gsigma)
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
        cv2.drawMarker(image, (int(detected_marker[0]*scale),int(detected_marker[1]*scale)),color = (255,0,0), markerType = cv2.MARKER_CROSS, markerSize = 10, thickness = 1)
    plt.imshow(image)
    plt.show()

# play with first pass filter params
# play with thresholding
# filter contours for area then circularity
# try blobdetector

def getBodypartsFromXmaExport(working_dir):
    '''Pull the names of the XMAlab markers from the 2Dpoints file'''
     # Open the config
    try:
        config_file = open(working_dir + "\\project_config.yaml", 'r')
    except FileNotFoundError:
        raise FileNotFoundError('Make sure that the current directory has a project already created in it.')
    yaml = YAML()
    project = yaml.load(config_file)
    
    # Establish project vars
    path_config_file = project['path_config_file']
    data_path = working_dir + "/trainingdata"
    try:
        dlc_config = open(path_config_file)
    except FileNotFoundError:
        raise FileNotFoundError('Oops! Looks like there\'s no deeplabcut config file inside of your deeplabcut directory.')
    
    # ADD support for more than one trial
    for trial in os.listdir(data_path):
        csv_path = [file for file in os.listdir(data_path + '/' + trial) if file[-4:] == '.csv']
        if len(csv_path) > 1:
            raise FileExistsError('Found more than 1 CSV file for trial: ' + trial)
        df = pd.read_csv(data_path + '/' + trial + '/' + csv_path[0], sep=',',header=0, dtype='float',na_values='NaN')
        names = df.columns.values
        parts = [name.rsplit('_',2)[0] for name in names]
        parts_unique = []
        for part in parts:
            if not part in parts_unique:
                parts_unique.append(part)
        return parts_unique