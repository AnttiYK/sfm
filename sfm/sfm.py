from utils import readImages, pts2ply
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, align_matches, matches2D3D, plotNewCamera
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
    points1, points2, i1, i2 = align_matches(kp1, des1, kp2, des2, matches)
    
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
    points = triangulate(points1,points2, K, R, t)
    


    #R1, R2, t = cameraPose(E)
    #showCameraPose(R1, R2, t)
    ## Triangulate points 
    #points3d = triangulate(points1[F_mask], points2[F_mask], K, R2, t)
    #showTriangulate(R1, R2, t, points1, points2, K, F_mask)
    
    ## Disambiguate camera pose
    #R, t, count = disambiguatePose(points1[F_mask], points2[F_mask], R1, R2, t, K)
    
    #points3D = triangulate(points1[F_mask], points2[F_mask], K, R, t)
    points3D = points
    pts2ply(points, "results.ply")
    for i in range(2, len(images)):
        kp3, des3 = akaze(images[i])
        matches = bfMatch(des2, des3)
        points1, points2, i1, i2 = align_matches(kp2, des2, kp3, des3, matches)
        E = K.T.dot(F.dot(K))
        F, F_mask = ransacFundamental(points1, points2)
        _, R,t,_ = cv2.recoverPose(E, points1[F_mask], points2[F_mask], K)
        P1 = P2
        P2 = np.hstack((R, t))
        points = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        
        points3D = np.concatenate((points3D, points[:3]/points[3]), axis = 1)
        #points3D[0] = np.append(points3D[0],points[0]/points[3])
        #points3D[1] = np.append(points3D[1],points[1]/points[3])
        #points3D[2] = np.append(points3D[2],points[2]/points[3])
        ## PnP and new camera ragistration
        #kp3, des3 = akaze(images[i])
        #img2D, pts3D = matches2D3D(des1,i1,des2, i2, des3, kp3, F_mask, points3D)

        #ret, Rvec, tnew, PnP_mask = cv2.solvePnPRansac(pts3D, img2D, K, dist)
       
        #retval, rvec, tvec = cv2.solvePnP(pts3D, img2D, K, dist, flags=cv2.SOLVEPNP_EPNP)
        #print(tvec)
        #tnew = tvec[:, 0]
        #print(tnew)
        #print(rvec)
        #[Rnew, j] = cv2.Rodrigues(rvec)
    
        #ii = img2D[0]
        #ii = np.hstack((ii, 1))
        
        
    
        #l =  np.matmul(np.linalg.inv(Rnew), ii.transpose() - tnew)
        #print(l)
        #print(i)
        #Rnew,tnew =ransacPnP(pts3D,img3D,K)
        #tnew = tnew[:, 0]

        ## Re-triangulate points
        #kpNew, descNew = kp3, des3
        #if i == 2:
         #   kpOld,descOld = kp2,des2
         

        

        #accPts = []
        #for (ROld, tOld, kpOld, descOld) in [(np.eye(3),np.zeros((3,1)), kp1,des1),(R,t,kp2,des2)]:  
         #   matches = bfMatch(descOld, des3)

          #  imgOldPts, imgNewPts, _, _ = align_matches(kpOld, descOld, kpNew, descNew, matches)
           # F, mask = ransacFundamental(imgOldPts, imgNewPts)
           # mask = mask.flatten().astype(bool)
           # imgOldPts=imgOldPts[mask]
           # imgNewPts=imgNewPts[mask]
        
    

         #   newPts = triangulate(imgOldPts,imgNewPts, K, Rnew,tnew[:,np.newaxis])
    
            #Adding newly triangulated points to the collection
          #  accPts.append(newPts)
        #kpOld = kpNew
        #descOld = descNew
    
    pts2ply(points3D, "acc.ply")