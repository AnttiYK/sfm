from utils import readImages
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, perspective,  showMatches
from camera_calibration import parameters, undistort
from incremental_reconstruction import reconstruction

class structure:
    calibration_images = None
    images = None
    calibration = {"mtx" : None, 'dist' : None, 'rvecs':None, 'tvecs':None}
    features = None
    matches = None
    init = None
    transformations = None
    reconstruction = None
 
    
def main():  # pragma: no cover
    struct = structure()
    ## camera calibration
    dir = "images/calibration_images"
    struct.calibration_images = readImages(dir)
    ## mtx = camera matrix, dist = distCoeffs
    struct.calibration['mtx'], struct.calibration['dist'], struct.calibration['rvecs'], struct.calibration['tvecs'] = parameters(struct.calibration_images)
    
    ## read images
    dir = "images/boat_images"
    struct.images = readImages(dir)
    ## undistort images
    struct.images = undistort(struct.images, struct.calibration)

    ## visualize plots not
    ## this contains secondary visualizations that are not directly related to sfm pipeline
    ## sfm related visualizations are located in their respective py files
    visualize(struct.images, struct.calibration_images, struct.calibration)

    ## feature detection
    ## returns array where akaze[i] = {kp, des} contains keypoint coordinates [i]['kp'] and the descriptor values [i]['des'] for image i
    struct.features = akaze(struct.images)
    #showFeatures(struct.features, struct.images)
    
    ## feature matching
    ## returns array where matches[i][j] contain matches between image i and j sorted from best match to worst
    struct.matches = bfMatch(struct.features)
    
    ## perspective transformation
    ## returns array transformation[i][j] = {H, mask} where H is homography matrix between images i and j and mask stores inlier information
    ## also returns initialization images with most matches
    struct.init, struct.transformations = perspective(struct.images, struct.features, struct.matches)
    #showMatches(struct.images, struct.transformations, struct.features, struct.matches)
    
    ##incremental reconstruction
    struct.reconstruction = reconstruction(struct.images, struct.features, struct.transformations, struct.init, struct.calibration, struct.matches)