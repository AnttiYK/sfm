import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
'''
Brute force matching for 
'''
def bfMatch(features):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = []
    for i in range(len(features)):
        matches.append([])
        for j in range(len(features)):
            sub_matches = matcher.match(features[i]['des'], features[j]['des'])
            sub_matches = sorted(sub_matches, key = lambda x:x.distance)
            matches[i].append(sub_matches)
    return matches

def perspective(images, features, matches):
    transformation = []
    init = (0, 0)
    init_val = 0
    for i in range(len(features)):
        img = images[i]
        h, w = len(img), len(img[0])
        pts = np.float32([ [0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        transformation.append([])
        for j in range(len(features)):
            src = np.float32([features[i]['kp'][m.queryIdx].pt for m in matches[i][j]]).reshape(-1, 1, 2)
            dst = np.float32([features[j]['kp'][m.trainIdx].pt for m in matches[i][j]]).reshape(-1, 1, 2)
            M, mask = cv.findHomography(src, dst, cv.RANSAC, 5.0)
            ## sets mask sum to 0 when image is compared to itself
            if (i == j):
                mask = [0]
            else:
                mask = mask.ravel().tolist()
            if (sum(mask)> init_val):
                init_val = sum(mask)
                init = (i, j)
            H = cv.perspectiveTransform(pts, M)
            transformation[i].append({'H': H, 'mask': mask})
    return init, transformation

def showMatches(images, transformations, features, matches):
    firstImageIndex = 3
    secondImageIndex = 2
    mask = transformations[firstImageIndex][secondImageIndex]['mask']
    draw_params2 = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mask, flags = 2)
    draw_params1 = dict(matchColor = (255, 0, 0), singlePointColor = None, flags = 2)
    img1 = cv.drawMatches(images[firstImageIndex], features[firstImageIndex]['kp'], images[secondImageIndex], features[secondImageIndex]['kp'], matches[firstImageIndex][secondImageIndex], None, **draw_params1)
    img2 = cv.drawMatches(images[firstImageIndex], features[firstImageIndex]['kp'], images[secondImageIndex], features[secondImageIndex]['kp'], matches[firstImageIndex][secondImageIndex], None, **draw_params2)
    fig = plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    plt.imshow(img1, 'gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img2, 'gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()