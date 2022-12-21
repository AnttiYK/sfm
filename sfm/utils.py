import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt
import random

def readImages(dir):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv.imread(os.path.join(dir, i))
        img = cv.resize(img, (800,600))
        imgs.append(np.asarray(img))
    return np.asarray(imgs)

def showImage(img):
    plt.imshow(img), plt.show()

def showFeatures(kps, imgs):
    r = random.randint(0, len(imgs)-1)
    img = imgs[r]
    kp = kps[r][0]
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
    plt.imshow(img2), plt.show()

def showMatches(img1, img2, kp1, kp2, matches):
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags = cv.DrawMatchesFlags_DEFAULT)
    plt.imshow(img3), plt.show()