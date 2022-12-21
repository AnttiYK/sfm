import cv2 as cv

def orb(imgs):
    detector = cv.ORB_create(nfeatures=500)
    features = []
    for i in imgs:
        kp = detector.detect(i, None)
        kp, des = detector.compute(i, kp)
        features.append([kp,des])
    return features

