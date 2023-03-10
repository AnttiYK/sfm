import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt


def get_order(transformations, init):
    order = []
    order.append(init[0])
    order.append(init[1])
    i = init[1]
    j = 0
    max_val = 0
    max_index = 0
    while(len(order) < len(transformations[0])):
        temp_max_val = sum(transformations[i][j]['mask'])
        temp_max_index = j
        if((temp_max_val >= max_val) and (temp_max_index not in order)):
            max_val = temp_max_val
            max_index = temp_max_index
        if(j >= len(transformations[0])-1):
            order.append(max_index)
            i = max_index
            j = 0
            max_val = 0
        else:
            j = j + 1
    return order
        
def triangulate(pts1, pts2, p_mat1, p_mat2) -> tuple:
        points = cv.triangulatePoints(pts1, pts2, p_mat1.T, p_mat2.T)
        print(points)
        print(points[3])
        return p_mat1.T, p_mat2.T, (points / points[3])  
    
def error(obj_p, img_p, transform, calibration, mode) -> tuple:
    rot_m = transform[:3, :3]
    tran_v = transform[:3, 3]
    rot_v, _ = cv.Rodrigues(rot_m)
    if mode == 1:
        obj_p = cv.convertPointsFromHomogeneous(obj_p.T)
    img_p_c, _ = cv.projectPoints(obj_p, rot_v, tran_v, calibration, None)
    img_p_c = np.float32(img_p_c[:,0,:])
    error = cv.norm(img_p_c, np.float32(img_p.T) if mode == 1 else np.float32(img_p), cv.NORM_L2)
    error = 1
    return error / len(img_p_c), obj_p

def PnP(points, features_dst, mtx, dist, rvec, mode):
    if mode == 1:
        points = points[:, 0, :]
        features_dst = features_dst.T
        rvec = rvec.T
    _, rvec_calculated, tvec, inlier = cv.solvePnPRansac(points, features_dst, mtx, dist, cv.SOLVEPNP_ITERATIVE)
    rotation_matrix = cv.Rodrigues(rvec_calculated)
    if inlier is not None:
        features_dst = features_dst[inlier[:,0]]
        points = points[inlier[:,0]]
        rvec = rvec[inlier[:, 0]]
    return rotation_matrix, tvec, features_dst, points, rvec
    

def initialize(features, matches, calibration, order):
    transform_matrix_src = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    transform_matrix_dst = np.empty((3,4))
    i1 = order[0]
    i2 = order[1]
    matches_src = np.float32([features[i1]['kp'][m.queryIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    matches_dst = np.float32([features[i2]['kp'][m.trainIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    E_matrix, E_mask = cv.findEssentialMat(matches_src, matches_dst, calibration['mtx'], method=cv.RANSAC, prob=0.999, threshold=0.4, mask=None)
    f0 = matches_src[E_mask.ravel()==1]
    f1 = matches_dst[E_mask.ravel()==1]
    _, rotation_matrix, translation_matrix, _ = cv.recoverPose(E_matrix, f0, f1, calibration['mtx'])
    transform_matrix_dst[:3, :3] = np.matmul(rotation_matrix, transform_matrix_src[:3, :3])
    transform_matrix_dst[:3, 3] = transform_matrix_dst[:3, 3] + np.matmul(transform_matrix_src[:3, :3], translation_matrix.ravel())
    pose_src = np.matmul(calibration['mtx'], transform_matrix_src)
    pose_dst = np.empty((3,4))
    pose_dst = np.matmul(calibration['mtx'], transform_matrix_dst)
    features_src, features_dst, points  = triangulate(pose_src, pose_dst, matches_src,  matches_dst);
    points = cv.convertPointsFromHomogeneous(points.T)
    pose_array = calibration['mtx'].ravel()
    _, _, features_src, points, _ = PnP(points, features_dst, calibration['mtx'], calibration['dist'],features_src, mode = 1)
    pose_array = np.hstack((np.hstack((pose_array, pose_src.ravel())), pose_dst.ravel()))
    return points, pose_dst, pose_src, pose_array, features_src, features_dst

def common_points(features_src, features_cur, features_dst):
    points_src = []
    points_dst = []
    for i in range(features_src.shape[0]):
        a = np.where(features_cur == features_src[i,:])
        if a[0].size != 0:
            points_src.append(i)
            points_dst.append(a[0][0])
    mask_src = np.ma.array(features_cur, mask = False)
    mask_src.mask[points_dst] = True
    mask_src = mask_src.compressed()
    mask_src = mask_src.reshape(int(mask_src.shape[0] / 2), 2)
    mask_dst = np.ma.array(features_dst, mask=False)
    mask_dst.mask[points_dst]=True
    mask_dst = mask_dst.compressed()
    mask_dst = mask_dst.reshape(int(mask_dst.shape[0] / 2), 2)
    return np.array(points_src), np.array(points_dst), mask_src, mask_dst
    
def incremental_reconstruction(images, points, pose_dst, pose_src, pose_array, features_src, features_dst, order, matches, features, calibration):
    length = len(order)-1
    total_points = np.zeros((1,3))
    total_colors = np.zeros((1,3))
    for i in range(1, length):
        i1 = order[i]
        i2 = order[i+1]
        features_cur = np.float32([features[i1]['kp'][m.queryIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
        features_2 = np.float32([features[i2]['kp'][m.trainIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
        if i != 1:
            features_src, features_dst, points  = triangulate(pose_src, pose_dst, features_src,  features_dst);
            features_src = features_src.T
            points = cv.convertPointsFromHomogeneous(points.T)
            points = points[:,0.:]
        common_points_src, common_points_dst, common_mask_src, common_mask_dst = common_points(features_src, features_cur, features_2)
        common_points_2 = features_2[common_points_dst]
        common_points_cur = features_cur[common_points_dst]
        rotation_matrix, translation_matrix, common_mask_dst, points, common_points_cur = PnP(points[common_points_src], common_points_2, calibration['mtx'], calibration['dist'], common_points_cur, mode = 0)
        transform_matrix_1 = np.hstack((rotation_matrix, translation_matrix))
        
        pose_2 = np.matmul(calibration['mtx'], transform_matrix_1)
        
        common_mask_src, common_mask_dst = triangulate(pose_dst, pose_2), common_mask_src, common_mask_dst
        points = cv.convertPointsFromHomogeneous(points.T)
       
        pose_array = np.hstack((pose_array, pose_2.ravel()))
        
        #Implement bundle adjustment
        total_points = np.vstack((total_points, points[:,0,:]))
        points_left = np.array(common_mask_dst, dtype=np.int32)
        img = images[i]
        colorvec = np.array([img[l[1], l[0]] for l in points_left.T])
        total_colors = np.vstack((total_colors, colorvec))
        
        transform_matrix_0 = np.copy(transform_matrix_1)
        pose_src = np.copy(pose_dst)
        features_src = np.copy(features_cur)
        features_dst= np.copy(features_2)      
        pose_dst = np.copy(pose_2)
        
        
def reconstruction(images, features, transformations, init, calibration, matches):
    order = get_order(transformations, init)
    points, pose_dst, pose_src, pose_array, features_src, features_dst = initialize(features, matches, calibration, order[:2])
    incremental_reconstruction(images, points, pose_dst, pose_src, pose_array, features_src, features_dst, order, matches, features, calibration)    
    
    