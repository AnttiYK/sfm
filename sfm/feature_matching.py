import cv2 as cv
import matplotlib.pyplot as plt

def bfMatch(f1, f2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    d1 = f1[1]
    d2 = f2[1]
    matches = matcher.match(d1,d2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def showMatches(img1, img2, kp1, kp2, matches):
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags = cv.DrawMatchesFlags_DEFAULT)
    plt.imshow(img3), plt.show()