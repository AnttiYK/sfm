import cv2
import os
import numpy as np

'''
Reads images from dir and returns them as array
'''
def readImages(dir, Color):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv2.imread(os.path.join(dir, i))
        if(Color == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(np.asarray(img))
    return np.asarray(imgs)





