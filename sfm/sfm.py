from utils import readImages
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, perspective, perspective2, verified_matches, showMatches
from camera_calibration import parameters, undistort

 
    
def main():  # pragma: no cover
    ## camera calibration
    dir = "images/calibration_images"
    images = readImages(dir, Color = True)
    ## mtx = camera matrix, dist = distCoeffs
    K, dist = parameters(images)
    
    ## read images
    dir = "images/100CANON"
    #dir = "images/CAB"
    images = readImages(dir, Color = False)
    
    ## undistort images
    images = undistort(images, K, dist)

    ## visualize plots not
    ## this contains secondary visualizations that are not directly related to sfm pipeline
    ## sfm related visualizations are located in their respective py files
    #visualize(struct.images, struct.calibration_images, struct.calibration)

    ## feature detection
    ## returns array where akaze[i] = {kp, des} contains keypoint coordinates [i]['kp'] and the descriptor values [i]['des'] for image i
    keypoints = []
    src_kp, src_des = akaze(images[0])
    keypoints.append(src_kp.pt)
    
    for i in range(1, len(images)):
        dst_kp, dst_des = akaze(images[i])
        idx_pairs = bfMatch(src_des, dst_des)
        

    