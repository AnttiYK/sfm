import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
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
        

    
def reconstruction(struct):
    order = get_order(struct.transformations, struct.init)
    obj_points = []
    img_points_list = []
    K = struct.calibration['mtx']
    dist = struct.calibration['dist']
    R_list = []
    t_list = []
    length = struct.length
    
    for i in range(length):
        i1 = order[i]
        if (i == length-1):
            i -= 2
        i2 = order[i+1]
        src = cv.KeyPoint_convert(struct.verified_matches[i1][i2])
        dst = cv.KeyPoint_convert(struct.verified_matches[i2][i1])
        l1 = len(src)
        l2 = len(dst)
        l = min(l1, l2)
        src = src[:l]
        dst = dst[:l]
        E, _ = cv.findEssentialMat(src, dst, K)
        _, R, t, _ = cv.recoverPose(E, src, dst, K)
        R_list.append(R)
        t_list.append(t)
    
    P_list = [K @ np.hstack((R, t)) for R, t in zip(R_list, t_list)]
    points3d = []
    points2d = []
    
    for i in range(length-1):
        i1 = order[i]
        i2 = order[i+1]
        matches = struct.matches[i1][i2]
        src = cv.KeyPoint_convert(struct.features[i1]['kp'])
        dst = cv.KeyPoint_convert(struct.features[i2]['kp'])
        l1 = len(src)
        l2 = len(dst)
        l = min(l1, l2)
        src = src[:l]
        dst = dst[:l]
        points_homogenous = cv.triangulatePoints(P_list[i1], P_list[i2], src.T, dst.T)
        points = cv.convertPointsFromHomogeneous(points_homogenous.T).reshape(-1,3)
        points3d.append(points)
        points2d.append(src)
        
    ## Bundle adjust
    #object_points = np.concatenate(points3d)
    #image_points = np.concatenate(points2d)
    #ret, K, dist, rvec, tvec = cv.calibrateCamera([object_points], [image_points], struct.images[0].shape[1::-1], K, dist, flags=cv.CALIB_USE_INTRINSIC_GUESS)
    #retval, rvec, tvec = cv.solvePnP(object_points, image_points, K, dist, rvec[0], tvec[0], flags=cv.SOLVEPNP_ITERATIVE)
    
    # Plot 3D points and camera poses
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    for p in points3d:
        ax.scatter(p[:,0], p[:,1], p[:,2], marker='o', s=0.2, color='black')
    # Plot camera positions
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        camera_pos = -R.T @ t
        ax.scatter(camera_pos[0], camera_pos[1], camera_pos[2], marker='s', s=50, color=f'C{i}')
        print(camera_pos)
        # Plot camera axes
        axes_len = 0.1
        #x_axis = R.T @ np.array([axes_len, 0, 0]) + camera_pos.reshape(3,1)
        #y_axis = R.T @ np.array([0, axes_len, 0]) + camera_pos.reshape(3,1)
        #z_axis = R.T @ np.array([0, 0, axes_len]) + camera_pos.reshape(3,1)
        #ax.plot([camera_pos[0], x_axis[0,0]], [camera_pos[1], x_axis[1,0]], [camera_pos[2], x_axis[2,0]], color='red')
        #ax.plot([camera_pos[0], y_axis[0,0]], [camera_pos[1], y_axis[1,0]], [camera_pos[2], y_axis[2,0]], color='green')
        #ax.plot([camera_pos[0], z_axis[0,0]], [camera_pos[1], z_axis[1,0]], [camera_pos[2], z_axis[2,0]], color='blue')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    