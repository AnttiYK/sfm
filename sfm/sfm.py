
from utils import readImages
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, perspective,  showMatches
from camera_calibration import parameters, undistort

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
    ## perspective transformation
    ## returns array transformation[i][j] = [tm, mask] where tm is transformation matrix between images i and j and mask stores inlier information
    transformations = perspective(images, features, matches)
    mask_count=[]
    for i in range(len(transformations)):
        mask_count.append([])
        for j in range(len(transformations[i])):
            mask_count[i].append(sum(transformations[i][j][1]))
    print(mask_count)
    showMatches(images, transformations, features, matches)