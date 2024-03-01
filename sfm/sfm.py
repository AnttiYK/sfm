from utils import readImages, pts2ply, get_projection_error
from visualization import visualize
from feature_detection_matching import feature_detect_match
from feature_matching import bfMatch, align_matches, align_3D_matches, matches2D3D, plotNewCamera
from camera_calibration import parameters, undistort
from fundamental import ransacFundamental, epipolar, showEpipolar, get_f_matrix, get_e_matrix
from cameraPose import cameraPose, showCameraPose, disambiguatePose, get_camera_pose
from triangulate import triangulate, showTriangulate, imgC, get_triangulate, get_nonlinear_triangulate
from pnp import ransacPnP, get_PnPRansac
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
    
def main():  # pragma: no cover
    
    ## camera calibration
    dir = "images/building/cal"
    print("Calibrating camera with dir:{}".format(dir))
    calibration_images = readImages(dir, True)
    ## mtx = camera matrix, dist = distCoeffs
    K, dist = parameters(calibration_images)
   
    ## read images
    dir = "images/building/img"
    print("Starting sfm with dir:{}".format(dir))
    images = readImages(dir, False)
    
    print("Feature detection and matching")
    ## feature detection and matching
    matches = feature_detect_match(images)
            


    

    print("################# Initializing with first two images #################")
    #f_matrix of first two images    
    print("Calculating fundamental matrix")   
    f_matrix, points_src, points_dst = get_f_matrix(matches[0][1][0], matches[0][1][1])

    #Essential matrix for first two images
    print("Calculating essential matrix")
    e_matrix = get_e_matrix(K,f_matrix)

    #camera pose 
    print("Calculating camera pose")
    R, C = get_camera_pose(e_matrix)

    R_ = np.identity(3)
    C_ = np.zeros((3,1))
    I = np.identity(3)
    pts3D=[]
    print("Linear triangulation")
    for i in range(len(C)):
        x = get_triangulate(K, C_, R_, C[i], R[i], points_src, points_dst)
        x = x/x[:,3].reshape(-1,1)
        pts3D.append(x)

    R_best, C_best, points = disambiguatePose(R, C, pts3D)
    points = points/points[:,3].reshape(-1,1)

    # triangulate points
    print("Nonlinear triangulation")
    points_ = get_nonlinear_triangulate(K, C_, R_, R_best, C_best, points_src, points_dst, points)
    points_ = points_ / points_[:,3].reshape(-1,1)

    mean_error1 = get_projection_error(points, points_src, points_dst, R_, C_, R_best, C_best, K )
    mean_error2 = get_projection_error(points_, points_src, points_dst, R_, C_, R_best, C_best, K )
    print('Mean errors: Linear triangulation : ', mean_error1, 'Nonlinear triangulation : ', mean_error2)
   
    print("################# Initialization done #################")
    print("Registering first two cameras")

    cameras = np.zeros((len(matches), 3))
    camera_idx = np.zeros((len(matches), 1), dtype=int)
    cameras_ = np.zeros((len(matches), 1), dtype=int)
    idx = np.zeros((len(matches), len(matches)))

    C_arr = []
    R_arr = []
    C_arr.append(np.zeros(3))
    R_arr.append(np.identity(3))
    C_arr.append(C_best)
    R_arr.append(R_best)
    print("Registering remaining cameras")

    for i in range(2, len(images)):
        print("Image number ", i +1, "out of ", len(images))

        points_i = np.hstack((matches[:i,i]))
        R_init, C_init = get_PnPRansac(K,points_i, points_, 1000, 5)



    #showFeatures(struct.features, struct.images)
    '''
    ## feature matching
    ## returns array where matches[i][j] contain matches between image i and j sorted from best match to worst
    matches = bfMatch(des1, des2)
    ## align matches
    points1, points2, des= align_matches(kp1, kp2, matches, des2)
    
    ## Fundamental matrix
    F, F_mask = ransacFundamental(points1, points2)
  

    ## Epipolar lines
    #e1 = epipolar(points1[F_mask], 1 ,F)
    #e2 = epipolar(points2[F_mask], 2, F)
    #showEpipolar(images[0], images[1], e1, e2, points1[F_mask], points2[F_mask])
    
    ## Camera poses 
    E = np.matmul(np.matmul(np.transpose(K), F), K)

    _, R,t,_ = cv2.recoverPose(E, points1[F_mask], points2[F_mask], K)

    P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])
    P2 = np.hstack((R, t))

    points = triangulate(points1,points2, K, P1, P2)
    

    points3D = points[:3]
    des_total = des
    points_total = np.zeros((3,len(points3D[0])))

    for i in range(len(points3D[0])):
        points_total[0][i] = points3D[0][i]
        points_total[1][i] = points3D[1][i]
        points_total[2][i] = points3D[2][i]
        
    pts2ply(points, "results.ply")
    for i in range(2, len(images)):
        #kp1, des1 = kp2, des2
        kp2, des2 = akaze(images[i])
        matches = bfMatch(des1, des2)
        points1, points2, des= align_matches(kp1, kp2, matches, des2)
        F, F_mask = ransacFundamental(points1, points2)
        E = np.matmul(np.matmul(np.transpose(K), F), K)
        
        _, R,t,_ = cv2.recoverPose(E, points1[F_mask], points2[F_mask], K)
        #P1 = P2
        #R_ = P1[:3,:3]
        #R = np.matmul(R,R_)
        #t_ = P1[:3,3]
        #t = t_ + np.matmul(t_, t.ravel())
        P2 = np.hstack((R,t))
        #P2 = np.zeros((3,4))
        #P2[:3,:3] = R
        #P2[:3, 3] = t.ravel()
        
           
        
        points = triangulate(points1,points2, K, P1, P2)
        #points = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
        points3D = points[:3]
        #points3D = points[:3]/points[3]
        points_total = np.concatenate((points_total, points3D), axis = 1)
    
        

    
    pts2ply(points_total, "acc.ply")
    '''