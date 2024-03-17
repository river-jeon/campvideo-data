# Facial feature vector extraction from video
'''
code: campvideo/models/image
https://github.com/atarr3/campvideo/blob/master/campvideo/image.py

faces of the politicians
https://github.com/atarr3/campvideo-data/tree/main/data/ids
'''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sys
sys.path.append('/home/j.soyeon/vit/camp/campvideo')
from campvideo.image import Keyframes
import numpy as np

# image frame extracted (test ver)
def find_face(image_dir, id_dir):
    kfs = Keyframes.fromdir(image_dir)#'./campvideo-data/data/extracted_frames/test')
    identity = np.load(id_dir)#'./campvideo-data/data/ids/2012_AZ_carmona.npy')
    print(kfs.facerec(identity))

