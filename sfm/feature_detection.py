import cv2
import matplotlib.pyplot as plt
import random        
import numpy as np

def akaze(img):
    detector = cv2.AKAZE_create(nOctaves=8)
    kp = detector.detect(img, None)
    kp, des = detector.compute(img, kp)
    return kp, des
    
    

    
    
def showFeatures(kps, imgs):
    r = random.randint(0, len(imgs)-1)
    img = imgs[r]
    kp = kps[r]['kp']
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.imshow(img2), plt.show()

