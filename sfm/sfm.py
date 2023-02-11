
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
    ## akaze[i] = [kp, des]
    akaze_f = akaze(images)
    showFeatures(akaze_f, images)
    
    ## feature matching
    #matches = bfMatch(orb_f[0], orb_f[1])
    #showMatches(images[0], images[1], orb_f[0][0], orb_f[1][0], matches)
