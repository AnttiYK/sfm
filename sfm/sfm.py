
from utils import readImages, showImage, showFeatures, showMatches, undistort
from visualization import visualize
from feature_detection import orb
from feature_matching import bfMatch
from camera_calibration import parameters

def main():  # pragma: no cover

    ## camera calibration
    dir = "images/calibration_images"
    calibration_images = readImages(dir)
    mtx, dist = parameters(calibration_images)

    ## read images
    dir = "images/boat_images"
    images = readImages(dir)
    c_images = undistort(images, mtx, dist)
    #showImage(images[0])

    ## visualize plots
    visualize(images, calibration_images, mtx, dist)

    ## feature detection
    ## orb[i] = [kp, des]
    #orb_f = orb(images)
    #showFeatures(orb_f, images)

    ## feature matching
    #matches = bfMatch(orb_f[0], orb_f[1])
    #showMatches(images[0], images[1], orb_f[0][0], orb_f[1][0], matches)
