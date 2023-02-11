import cv2 as cv
import os
import numpy as np

def readImages(dir):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv.imread(os.path.join(dir, i))
        img = cv.resize(img, (1200, 800))
        imgs.append(np.asarray(img))
    return np.asarray(imgs)

def undistort(imgs, mtx, dist):
    for i in imgs:
       i = cv.undistort(i, mtx, dist)
    return imgs




