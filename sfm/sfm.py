from read_images import readImages
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, get_matches, showMatches, ransac_fundamental
from camera_calibration import parameters, undistort
from pose_estimation import essential_matrix, estimate_pose
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import numpy as np

    
def main():  # pragma: no cover
    # ## camera calibration
    # dir = "images/calibration_images"
    # calibration_images = readImages(dir, Color=True)
    # ## mtx = camera matrix, dist = distCoeffs
    # K, dist = parameters(calibration_images)
    
    # ## read images
    # dir = "images/100CANON"
    # images = readImages(dir, Color=False)
    # ## undistort images
    # images = undistort(images, K, dist)

    # ## visualize plots not
    # ## this contains secondary visualizations that are not directly related to sfm pipeline
    # ## sfm related visualizations are located in their respective py files
    # #visualize(struct.images, struct.calibration_images, struct.calibration)
    # kp = []
    # features = []

    # src_kp, src_features = akaze(images[0])
    # kp.append(src_kp)
    # #showFeatures(struct.features, struct.images)

    # for i in tqdm(range(1, len(images))):
    #     dst_kp, dst_features = akaze(images[1])
    #     kp.append(dst_kp)
    #     ## match_idx[i][0] = index for src, match[i][1] = index for src for pair i
    #     match_idx = bfMatch(src_features, dst_features)
    #     src_kp, dst_kp = get_matches(src_kp, dst_kp, match_idx)
    #     F = ransac_fundamental(src_kp, dst_kp)
    #     E = essential_matrix(F, K)
    #     pose = estimate_pose(E, src_kp, dst_kp, K)


   
    ## camera calibration
    dir = "images/calibration_images"
    calibration_images = readImages(dir, Color=True)
    ## mtx = camera matrix, dist = distCoeffs
    K, dist = parameters(calibration_images)
    
    ## read images
    dir = "images/100CANON"
    images = readImages(dir, Color=False)
    ## undistort images
    #images = undistort(images, K, dist)

    
    akaze = cv2.AKAZE_create()
    src_kp, src_desc = akaze.detectAndCompute(images[0], None)

    for i in tqdm(range(1, len(images))):
        dst_kp, dst_desc = akaze.detectAndCompute(images[i], None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
        nn_matches = matcher.knnMatch(src_desc, dst_desc, 2)

    
        good = []
        for m,n in nn_matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        
        src_pts = np.float32([ src_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ dst_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = images[i].shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        
        img2 = cv2.polylines(images[i],[np.int32(dst)],True,255,3, cv2.LINE_AA)

        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        img3 = cv2.drawMatches(images[i-1],src_kp,img2,dst_kp,good,None,**draw_params)
        plt.imshow(img3, 'gray'),plt.show()


        src_kp, src_desc =  dst_kp, dst_desc



   