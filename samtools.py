import os

def tif_to_avi(filepath):
    # Rename all files to be of the form
    for i, tiffile in [p for p in os.listdir(filepath) if p.endswith('.tif')]:
        if i == 0:
            ind = tiffile.find('cam')
            cam = tiffile[ind:ind+4]
        newName = cam + "frame" + str(i) + '.tif'
        os.rename(tiffile, newName)
    
    os.system("ffmpeg -i " + filepath + "/" + cam + "/frame%d.tif -vcodec ffv1" + os.path.abspath(filepath) + ".avi")

if __name__ == "__main__":
    tif_to_avi(".")