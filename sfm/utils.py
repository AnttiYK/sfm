import cv2 as cv
import os
import numpy as np

'''
Reads images from dir and returns them as array
'''
def readImages(dir):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv.imread(os.path.join(dir, i))
        img = cv.resize(img, (1200, 800))
        imgs.append(np.asarray(img))
    return np.asarray(imgs)





