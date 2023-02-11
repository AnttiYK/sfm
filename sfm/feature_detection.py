import cv2 as cv
import matplotlib.pyplot as plt
import random

def akaze(imgs):
    detector = cv.AKAZE_create()
    features = []
    for i in imgs:
        kp = detector.detect(i, None)
        kp, des = detector.compute(i, kp)
        features.append([kp,des])
    return features

def showFeatures(kps, imgs):
    r = random.randint(0, len(imgs)-1)
    img = imgs[r]
    kp = kps[r][0]
    img2 = cv.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
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

