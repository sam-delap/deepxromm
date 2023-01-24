import os
def tif_to_avi(working_dir=os.getcwd()):
    '''Converts training data tif files to AVI videos'''
    XMA_dir = working_dir + "/trainingdata"
    try:
        os.chdir(XMA_dir)
    except FileNotFoundError:
        if os.path.isdir(working_dir + "/trainingdata"):
            raise Exception("Please create a trainingdata directory inside your parent directory")
    # For each trial
    for dir in os.listdir():
        # Change to the trials directory
        try:
            os.chdir(dir)
        except NotADirectoryError:
            raise Exception("Please separate your training_images directory into per-trial folders")
        try:
            os.chdir("training_images")
        except:
            raise Exception("Please create a training_images directory inside your trial directory")
        # Assuming all directories within a trial are for undistorted trial images, find how many tifdirs there are
        tifdirs = [f for f in os.listdir() if os.path.isdir(f)]
        if len(tifdirs) == 0:
            raise Exception("Please export a set of undistorted trial images into this directory")

        for tifdir in tifdirs:
            # Find filename pattern
            os.chdir(tifdir)
            filename = os.listdir('.')[0]
            index = filename.find('.')
            pattern = filename[:index+1]
            # Use ffmpeg
            index2 = filename.find('_cam')
            index3 = filename.find('_UND')
            cmd = "ffmpeg -i " + pattern + "%04d.tif -r 30 -b:v 9653K " + "..\\\\..\\\\..\\\\" + pattern[:index2] + "\\\\" + pattern[:index3] + ".avi"
            print(cmd)
            os.system(cmd)
            os.chdir("../")
