import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def bfMatch(features):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = []
    for i in range(len(features)):
        matches.append([])
        for j in range(len(features)):
            sub_matches = matcher.match(features[i][1], features[j][1])
            sub_matches = sorted(sub_matches, key = lambda x:x.distance)
            matches[i].append(sub_matches)
    return matches

def showMatches(images, features, matches):
    img = cv.drawMatches(images[0], features[0][0], images[1], features[1][0], matches[0][1][:20], None, flags = cv.DrawMatchesFlags_DEFAULT)
    plt.imshow(img)
    ax = plt.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()