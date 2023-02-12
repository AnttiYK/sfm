
from utils import readImages, undistort
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch,  showMatches
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
    ## returns array where akaze[i] = [kp, des] contains keypoint coordinates [i][0] and the descriptor values [i][1] for image i
    features = akaze(images)
    #showFeatures(features, images)
    
    ## feature matching
    ## returns array where matches[i][j] contain matches between image i and j sorted from best match to worst
    matches = bfMatch(features)
    showMatches(images, features, matches)
