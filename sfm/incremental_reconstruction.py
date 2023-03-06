import cv2 as cv
import numpy as np

def initialize(init, transformations):
    ret = []
    ret.append(init[0])
    ret.append(init[1])
    length = len(transformations[0]) - 1
     



        
    
    
    

def reconstruction(images, features, transformations, init, calibration, matches):
    i1 = init[0]
    i2 = init[1]
    img1 = images[i1]
    img2 = images[i2]
    H = transformations[i1][i2]['H']
    src = np.float32([features[i1]['kp'][m.queryIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    dst = np.float32([features[i2]['kp'][m.trainIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    src = src[:4]
    dst = dst[:4]
    M1 = cv.getPerspectiveTransform(src, dst)
    M2 = cv.getPerspectiveTransform(dst, src)
    r1 = cv.warpPerspective(img1, M1, (img1.shape[0], img1.shape[1]))
    r2 = cv.warpPerspective(img2, M2, (img2.shape[0], img2.shape[1]))
    matcher = cv.StereoMatcher()
    disparity = matcher.match(r1, r2)
    
    