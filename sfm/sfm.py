from utils import readImages, pts2ply
from visualization import visualize
from feature_detection import akaze, showFeatures
from feature_matching import bfMatch, align_matches, matches2D3D, plotNewCamera
from camera_calibration import parameters, undistort
from fundamental import ransacFundamental, epipolar, showEpipolar
from cameraPose import cameraPose, showCameraPose, disambiguatePose
from triangulate import triangulate, showTriangulate
from pnp import ransacPnP
import cv2
import numpy as np

import numpy as np

    
def main():  # pragma: no cover
    ## camera calibration
    dir = "images/calibration_images"
    calibration_images = readImages(dir, True)
    ## mtx = camera matrix, dist = distCoeffs
    K, dist = parameters(calibration_images)
    
    ## read images
    dir = "images/100CANON"
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
    
    ## Epipolar lines
    e1 = epipolar(points1[F_mask], 1 ,F)
    e2 = epipolar(points2[F_mask], 2, F)
    #showEpipolar(images[0], images[1], e1, e2, points1[F_mask], points2[F_mask])
    
    ## Camera poses 
    E = K.T.dot(F.dot(K))
    R1, R2, t = cameraPose(E)
    #showCameraPose(R1, R2, t)
    ## Triangulate points 
    points3d = triangulate(points1[F_mask], points2[F_mask], K, R2, t)
    #showTriangulate(R1, R2, t, points1, points2, K, F_mask)
    
    ## Disambiguate camera pose
    R, t, count = disambiguatePose(points1[F_mask], points2[F_mask], R1, R2, t, K)
    
    points3D = triangulate(points1[F_mask], points2[F_mask], K, R, t)
    pts2ply(points3D, "results.ply")
    for i in range(2, len(images)):
        ## PnP and new camera ragistration
        kp3, des3 = akaze(images[i])
        img3D, pts3D = matches2D3D(des1,i1,des2, i2, des3, kp3, F_mask, points3D)
        print(i)
        ret, Rvec, tnew, PnP_mask = cv2.solvePnPRansac(pts3D, img3D, K, dist)
        Rnew = cv2.Rodrigues(Rvec)
        tnew = tnew[:, 0]
    
        #Rnew,tnew =ransacPnP(pts3D,img3D,K)
        #tnew = tnew[:, 0]

        ## Re-triangulate points
        kpNew, descNew = kp3, des3
        if i == 2:
            kpOld,descOld = kp2,des2
         

        

        accPts = []
        for (ROld, tOld, kpOld, descOld) in [(np.eye(3),np.zeros((3,1)), kp1,des1),(R,t,kp2,des2)]:  
            matches = bfMatch(descOld, des3)

            imgOldPts, imgNewPts, _, _ = align_matches(kpOld, descOld, kpNew, descNew, matches)
            F, mask = ransacFundamental(imgOldPts, imgNewPts)
            mask = mask.flatten().astype(bool)
            imgOldPts=imgOldPts[mask]
            imgNewPts=imgNewPts[mask]
        
    

            newPts = triangulate(imgOldPts,imgNewPts, K, Rnew,tnew[:,np.newaxis])
    
            #Adding newly triangulated points to the collection
            accPts.append(newPts)
        kpOld = kpNew
        descOld = descNew
        
    pts2ply(np.concatenate(accPts, axis=0), "acc.ply")