import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def parameters(imgs):

    #board size
    CHECKERBOARD = (7,10)

    #termination
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
    # array for 3D points
    objpoints = []
    # array for 2D points
    imgpoints = [] 
 
    # world coordinates for 3D points
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    for i in imgs:
        gray = cv.cvtColor(i,cv.COLOR_BGR2GRAY)
        
        #find corners
        ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)

        if ret == True:
            objpoints.append(objp)
            # refining pixel coordinates for given 2d points.
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
         
            imgpoints.append(corners2)
 
            # Draw and display the corners
            img = cv.drawChessboardCorners(i, CHECKERBOARD, corners2, ret) 
        cv.imshow('img',img)
        cv.waitKey(0)
 
    cv.destroyAllWindows()
 
    h,w = img.shape[:2]
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 
    print("Camera matrix : \n")
    print(mtx)
    print("dist : \n")
    print(dist)
    print("rvecs : \n")
    print(rvecs)
    print("tvecs : \n")
    print(tvecs)
    