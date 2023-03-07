import cv2 as cv
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

def triangulate(pts1, pts2, p_mat1, p_mat2) -> tuple:
        points = cv.triangulatePoints(pts1, pts2, p_mat1.T, p_mat2.T)
        return p_mat1.T, p_mat2.T, (points / points[3])  
    
def error(obj_p, img_p, transform, calibration, mode) -> tuple:
    rot_m = transform[:3, :3]
    tran_v = transform[:3, 3]
    rot_v, _ = cv.Rodrigues(rot_m)
    if mode == 1:
        obj_p = cv.convertPointsFromHomogeneous(obj_p.T)
    img_p_c, _ = cv.projectPoints(obj_p, rot_v, tran_v, calibration['mtx'], None)
    img_p_c = np.float32(img_p_c[:,0,:])
    #error = cv.norm(img_p_c, np.float32(img_p.T) if mode == 1 else np.float32(img_p), cv.NORM_L2)
    error = 1
    return error / len(img_p_c), obj_p
    
def PnP(obj_p, img_p, calibration, dist, rot_v, mode):
    if mode == 1:
        obj_p = obj_p[:, 0, :]
        img_p = img_p.T
        rot_v = rot_v.T
    _, rot_v_c, tran_v, inliers = cv.solvePnPRansac(obj_p, img_p, calibration['mtx'], dist , cv.SOLVEPNP_ITERATIVE)
    rot_m, _ = cv.Rodrigues(rot_v_c)
    if inliers != None:
        img_p = img_p[inliers[:, 0]]
        obj_p = obj_p[inliers[:, 0]]
        rot_v = rot_v[inliers[:, 0]]
    return rot_m, tran_v, img_p, obj_p, rot_v

def common_points(pts1, pts2, pts3):
    c_points1 = []
    c_points2 = []
    for i in range(pts1.shape[0]):
        a = np.where(pts2 == pts1[i,:])
        if a[0].size != 0:
            c_points1.append(i)
            c_points2.append(a[0][0])
    mask1 = np.ma.array(pts2, mask=False)
    mask1.mask[c_points2] = True
    mask1 = mask1.compressed()
    mask1 = mask1.reshape(int(mask1.shape[0]/2), 2)
    
    mask2 = np.ma.array(pts3, mask=False)
    mask2.mask[c_points2] = True
    mask2 = mask2.compressed()
    mask2 = mask2.reshape(int(mask2.shape[0]/2), 2)
    return np.array(c_points1), np.array(c_points2), mask1, mask2

def optimal_error(points):
    m = points[0:12].reshape((3,4))
    k = points[12:21].reshape((3,3))
    r = int(len(points[21:])*0.4)
    p = points[21:21 + r].reshape((2, int(r/2))).T
    points = points[21 + r:].reshape((int(len(points[21+r:])/3),3))
    rot = m[:3, :3]
    tran = m[:3, 3]
    rvec = cv.Rodrigues(rot)
    ip = cv.projectPoints(points, rvec, tran, k, None)
    ip = ip[:, 0, :]
    e = [(p[idx]-ip[idx]**2 for idx in range(len(p)))]
    optimal_error = np.array(e).ravel()/len(p)
    return optimal_error

def bundle_adjustment(points, opt, t_mat, calibration, error):
    variables = np.hstack((t_mat.ravel(), calibration['mtx'].ravel()))
    print(variables)
    variables = np.hstack((variables, opt.ravel()))
    variables = np.hstack((variables, points.ravel()))
    values = least_squares(optimal_error, variables, gtol= error, ftol = error, xtol = error).x
    k = values[12:21].reshape((3,3))
    r=int(len(values[21:])*0.4)
    return values[21 + r:].reshape((int(len(values[21+r:])/3),3)), values[21:21+r].reshape((2, int(r/2))).T, values[0:12].reshape((3,4))

def pose_estimation(matches, features, order, calibration, images):
    poses = calibration['mtx'].ravel()
    ## initialization
    t_mat0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    t_mat1 = np.empty((3,4))
    p0 = np.matmul(calibration['mtx'], t_mat0)
    p1 = np.empty((3,4))
    t_points = np.zeros((1,3))
    t_colors = np.zeros((1,3))
    i1 = order[0]
    i2 = order[1]
    src = np.float32([features[i1]['kp'][m.queryIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    dst = np.float32([features[i2]['kp'][m.trainIdx].pt for m in matches[i1][i2]]).reshape(-1, 1, 2)
    E, e_mask = cv.findEssentialMat(src, dst, calibration['mtx'], method=cv.RANSAC, prob=0.999, threshold=0.4, mask=None)
    f0 = src[e_mask.ravel()==1]
    f1 = dst[e_mask.ravel()==1]
    _, rot, tran, p_mask = cv.recoverPose(E, f0, f1, calibration['mtx'])
    f0 = src[e_mask.ravel()>0]
    f1 = dst[e_mask.ravel()>0]
    t_mat1[:3, :3] = np.matmul(rot, t_mat0[:3, :3])
    t_mat1[:3, 3] = t_mat0[:3, 3] + np.matmul(t_mat0[:3, :3], tran.ravel())
    p1 = np.matmul(calibration['mtx'], t_mat1)
    f0, f1, points = triangulate(p0, p1, f0, f1)
    err, points = error(points, f1, t_mat1, calibration, 1)
    _, _, f1, points, _ = PnP(points, f1, calibration, np.zeros((5, 1), dtype=np.float32), f0, 1)
    poses = np.hstack((np.hstack((poses, p0.ravel())), p1.ravel()))
    
    ## incremental reconstruction

    for i in range(1, len(order)-1):
        i1 = order[i]
        i2 = order[i+1]
        src = np.float32([features[i1]['kp'][m.queryIdx].pt for m in matches[i1][i2]])
        dst = np.float32([features[i2]['kp'][m.trainIdx].pt for m in matches[i1][i2]])    
        if i != 1:
            f0, f1, points = triangulate(p0, p1, f0, f1)
            f1 = f1.T
            points = cv.convertPointsFromHomogeneous(points.T)
            points = points[:, 0, :]
        
        c_points0, c_points1, c_mask0, c_mask1 = common_points(f1, src, dst)
        c_points2 = dst[c_points1]
        c_pointsC = src[c_points1]
        
        rot, tran, c_points2, points, c_pointsC = PnP(points[c_points0], c_points2, calibration, np.zeros((5,1), dtype=np.float32), c_pointsC, 0)
        t_mat1 = np.hstack((rot, tran))
        p2 = np.matmul(calibration['mtx'], t_mat1)
        
        err, points = error(points, c_points2, t_mat1, calibration, 0)
        
        c_mask0, c_mask1, points = triangulate(p1, p2, c_mask0, c_mask1)
        err, points = error(points, c_mask1, t_mat1, calibration, 1)
        
        poses = np.hstack((poses, p2.ravel()))
        
        ##bundle adjustment
        #points, c_mask1, t_mat1 = bundle_adjustment(points, c_mask1, t_mat1, calibration, 0.5)
        #p2 = np.matmul(calibration['mtx'], t_mat1)
        #err, points = error(points, c_mask1, t_mat1, calibration, 0)
        #t_points = np.vstack((t_points, points))
        #pl = np.array(c_mask1, dtype =np.int32)
        #img = images[i2]
        #cvec = np.array([img[l[1], l[0]]]for l in pl.T)
        #t_colors = np.vstack((t_colors, cvec))

        t_points = 

        t_mat0 = np.copy(t_mat1)
        p0 = np.copy(p0)
        plt.scatter(i, err)
        plt.pause(0.05)
        
        img0 = np.copy(images[i1])
        img1 = np.copy(images[i2])
        f0 = np.copy(src)
        f1 = np.copy(dst)
        p1 = np.copy(p2)
        
        cv.imshow(images[0].split('\\')[-2], img1)
                
        
    return poses

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

def reconstruction(images, features, transformations, init, calibration, matches):
    order = get_order(transformations, init)
    poses = pose_estimation(matches, features, order, calibration, images)
    
    