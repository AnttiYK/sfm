
from utils import readImages, showImage, showFeatures, showMatches
from visualization import visualize
from feature_detection import orb
from feature_matching import bfMatch
from camera_calibration import parameters

def main():  # pragma: no cover

    

    ## camera calibration
    #dir = "images/calibration_images"
    #calibration_images = readImages(dir)
    #parameters(calibration_images)


    ## read images
    dir = "images/boat_images"
    images = readImages(dir)
    #showImage(images[0])

    ## visualize needed plots
    visualize(images)

    ## feature detection
    ## orb[i] = [kp, des]
    #orb_f = orb(images)
    #showFeatures(orb_f, images)

    ## feature matching
    #matches = bfMatch(orb_f[0], orb_f[1])
    #showMatches(images[0], images[1], orb_f[0][0], orb_f[1][0], matches)
