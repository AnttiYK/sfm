from utils import readImages, pts2ply
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, align_matches, align_3D_matches, matches2D3D, plotNewCamera
from camera_calibration import parameters, undistort
from fundamental import ransacFundamental, epipolar, showEpipolar
from cameraPose import cameraPose, showCameraPose, disambiguatePose
from triangulate import triangulate, showTriangulate, imgC
from pnp import ransacPnP
import cv2
import numpy as np

import matplotlib.pyplot as plt
    
def main():  # pragma: no cover
    ## camera calibration
    dir = "images/boat/cal"
    calibration_images = readImages(dir, True)
    ## mtx = camera matrix, dist = distCoeffs
    K, dist = parameters(calibration_images)
    
    ## read images
    dir = "images/boat/img"
    images = readImages(dir, False)
    ## undistort images
    images = undistort(images, K, dist)

    ## visualize plots not
    ## this contains secondary visualizations that are not directly related to sfm pipeline
    ## sfm related visualizations are located in their respective py files
    #visualize(struct.images, struct.calibration_images, struct.calibration)
    
    ## feature detection
    kp1, des1 = akaze(images[0])
    kp2, des2 = akaze(images[1])
    #showFeatures(struct.features, struct.images)
    
    ## feature matching
    ## returns array where matches[i][j] contain matches between image i and j sorted from best match to worst
    matches = bfMatch(des1, des2)
    ## align matches
    points1, points2, des= align_matches(kp1, kp2, matches, des2)
    
    ## Fundamental matrix
    F, F_mask = ransacFundamental(points1, points2)
    
    #F, F_mask = cv2.findFundamentalMat(points1, points2)

    ## Epipolar lines
    #e1 = epipolar(points1[F_mask], 1 ,F)
    #e2 = epipolar(points2[F_mask], 2, F)
    #showEpipolar(images[0], images[1], e1, e2, points1[F_mask], points2[F_mask])
    
    ## Camera poses 
    E = K.T.dot(F.dot(K))

    _, R,t,_ = cv2.recoverPose(E, points1[F_mask], points2[F_mask], K)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    #points = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points = triangulate(points1,points2, K, P1, P2)
    


    #R1, R2, t = cameraPose(E)
    #showCameraPose(R1, R2, t)
    ## Triangulate points 
    #points3d = triangulate(points1[F_mask], points2[F_mask], K, R2, t)
    #showTriangulate(R1, R2, t, points1, points2, K, F_mask)
    
    ## Disambiguate camera pose
    #R, t, count = disambiguatePose(points1[F_mask], points2[F_mask], R1, R2, t, K)
    
    #points3D = triangulate(points1[F_mask], points2[F_mask], K, R, t)
    points3D = points[:3]
    des_total = des
    points_total = []
    for i in range(len(points3D[0])):
        points_total.append([points3D[0][i], points3D[1][i], points3D[2][i]])

    pts2ply(points, "results.ply")
    for i in range(2, len(images)):
        '''
        kp3, des3 = akaze(images[i])
        matches = bfMatch(des2, des3)
        points1, points2, i1, i2 = align_matches(kp2, des2, kp3, des3, matches)
        E = K.T.dot(F.dot(K))
        F, F_mask = ransacFundamental(points1, points2)
        _, R,t,_ = cv2.recoverPose(E, points1[F_mask], points2[F_mask], K)
        P1 =P2
        P2 = np.hstack((R, t))
        #points = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points =  triangulate(points1,points2,K, P1, P2)
        points = points[:3]
        points3D = np.concatenate((points3D, points), axis = 1)
        des2  = des3
        kp2 = kp3
        '''
        print(des_total.shape)
        print(len(points_total))
        kp3, des3 = akaze(images[i])
        matches = bfMatch(des_total, des3)
    
        world_points, points1, des = align_3D_matches(points_total, kp3, matches, des_total)
        #R, t = ransacPnP(world_points, img_points, K)

        ret, R, t = cv2.solvePnP(world_points, points1, K, dist)
        points2, _ = cv2.projectPoints(world_points, R, t, K, dist)
    
        R,_ = cv2.Rodrigues(R)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.T
        points = triangulate(points1,points2, K, P1, P2)




        des_total = np.concatenate((des_total,des))
        points3D = np.concatenate((points3D, points[:3]), axis = 1)
        points_total = []
        for i in range(len(points3D[0])):
            points_total.append([points3D[0][i], points3D[1][i], points3D[2][i]])
        

    
    pts2ply(points3D, "acc.ply")