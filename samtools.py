import os

def tif_to_avi(filepath):
    # Rename all files to be of the form
    for i, tiffile in [p for p in os.listdir(filepath) if p.endswith('.tif')]:
        newName = "frame" + str(i) + '.tif'
        os.rename(tiffile, newName)
    
    