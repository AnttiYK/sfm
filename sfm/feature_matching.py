import cv2 as cv

def bfMatch(f1, f2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    d1 = f1[1]
    d2 = f2[1]
    matches = matcher.match(d1,d2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches