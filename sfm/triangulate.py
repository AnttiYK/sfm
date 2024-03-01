import numpy as np
import cv2
import matplotlib.pyplot as plt
from cameraPose import PlotCamera
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as mpatches
from tqdm import tqdm
import scipy.optimize as optimize

def get_projection_matrix(R, C, K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def get_loss(x, pts_src, pts_dst, P_src, P_dst):
    p1_src, p2_src, p3_src = P_src
    p1_src, p2_src, p3_src = p1_src.reshape(1,-1), p2_src.reshape(1,-1),p3_src.reshape(1,-1)

    p1_dst, p2_dst, p3_dst = P_dst
    p1_dst, p2_dst, p3_dst = p1_dst.reshape(1,-1), p2_dst.reshape(1,-1), p3_dst.reshape(1,-1)

    u_src,v_src = pts_src[0], pts_src[1]
    u_src_ = np.divide(p1_src.dot(x), p3_src.dot(x))
    v_src_ = np.divide(p2_src.dot(x), p3_src.dot(x))
    E_src = np.square(v_src-v_src_) + np.square(u_src-u_src_)

    u_dst,v_dst = pts_dst[0], pts_dst[1]
    u_dst_ = np.divide(p1_dst.dot(x), p3_dst.dot(x))
    v_dst_ = np.divide(p2_dst.dot(x), p3_dst.dot(x))
    E_dst = np.square(v_dst-v_dst_) + np.square(u_dst-u_dst_)

    err = E_src + E_dst
    return err.squeeze()




def get_nonlinear_triangulate(K, C_src, R_src, R_dst, C_dst, points_src, points_dst, points):
    P_src = get_projection_matrix(R_src, C_src, K)
    P_dst = get_projection_matrix(R_dst, C_dst, K)
    points_ = []
    for i in tqdm(range(len(points))):
        opt = optimize.least_squares(fun=get_loss, x0 = points[i], method="trf", args=[points_src[i], points_dst[i], P_src, P_dst])
        X = opt.x
        points_.append(X)
    return np.array(points_)




def get_triangulate(K, C_src, R_src, C_dst, R_dst, points_src, points_dst):
    I = np.identity(3)
    C_src = np.reshape(C_src, (3,1))
    C_dst = np.reshape(C_dst, (3,1))

    P_src = np.dot(K, np.dot(R_src, np.hstack((I, -C_src))))
    P_dst = np.dot(K, np.dot(R_dst, np.hstack((I, -C_dst))))

    p1_src_t = P_src[0,:].reshape(1,4)
    p2_src_t = P_src[1,:].reshape(1,4)
    p3_src_t = P_src[2,:].reshape(1,4)

    p1_dst_t = P_dst[0,:].reshape(1,4)
    p2_dst_t = P_dst[1,:].reshape(1,4)
    p3_dst_t = P_dst[2,:].reshape(1,4)

    points = []

    for i in range(points_src.shape[0]):
        x_src = points_src[i, 0]
        y_src = points_src[i, 1]
        x_dst = points_dst[i, 0]
        y_dst = points_dst[i, 1]

        A = []
        A.append((y_src*p3_src_t)-p2_src_t)
        A.append(p1_src_t-(x_src*p3_src_t))
        A.append((y_dst*p3_dst_t)-p2_dst_t)
        A.append(p1_dst_t-(x_dst-p3_dst_t))

        A = np.array(A).reshape(4,4)

        _,_,vt = np.linalg.svd(A)
        v = vt.T
        v = v[:,-1]
        points.append(v)
    return np.array(points)

def triangulate(x1, x2, K, P1, P2):

    R_ = np.eye(3)
    t_ = np.zeros((3,1))
    
    x1hom = cv2.convertPointsToHomogeneous(x1)[:,0,:]
    x2hom = cv2.convertPointsToHomogeneous(x2)[:,0,:]
    
    x1norm = (np.linalg.inv(K).dot(x1hom.T)).T
    x2norm = (np.linalg.inv(K).dot(x2hom.T)).T
    
    x1norm = cv2.convertPointsFromHomogeneous(x1norm)[:,0,:]
    x2norm = cv2.convertPointsFromHomogeneous(x2norm)[:,0,:]

    points3D = np.zeros((3,1))

    for i in range(len(x1)):
        A = np.zeros((4, 4))

        A[0, :] = x1norm[i][0]*P1[2, :] - P1[0, :]
        A[1, :] = x1norm[i][1]*P1[2, :] - P1[1, :]
        A[2, :] = x2norm[i][0]*P2[2, :] - P2[0, :]
        A[3, :] = x2norm[i][1]*P2[2, :] - P2[1, :]
    
        U, S, Vt = np.linalg.svd(A)
        X_homog = Vt[-1, :]
        #X_homog /= X_homog[:3]
        X0 = X_homog[0]
        X1 = X_homog[1]
        X2 = X_homog[2]
        #print(X)
        X_ = [[X0], [X1], [X2]]
        points3D = np.concatenate((points3D, X_), axis = 1)
    
    points3D = cv2.triangulatePoints(P1, P2, x1norm.T, x2norm.T)
    
    return points3D

def showTriangulate(R1, R2, t, x1, x2, K, mask):
    sets = [None, None, None, None]
    sets[0] = (R1, t, triangulate(x1[mask], x2[mask], K, R1, t))
    sets[1] = (R1, -t, triangulate(x1[mask], x2[mask], K, R1, -t))
    sets[2] = (R2, t, triangulate(x1[mask], x2[mask], K, R2, t))
    sets[3] = (R2, -t, triangulate(x1[mask], x2[mask], K, R2, -t))
    i = 1
    fig, axs = plt.subplots(2,2)
    
    for s in sets: 
        ax = fig.add_subplot(2,2,i, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


        PlotCamera(np.eye(3,3),np.zeros((3,)),ax,scale=5,depth=5)
        PlotCamera(s[0],s[1][:,0],ax,scale=5,depth=5)

        pts3d = s[-1]
        ax.scatter3D(pts3d[:,0],pts3d[:,1],pts3d[:,2])

        ax.set_xlim(left=-50,right=50)
        ax.set_ylim(bottom=-50,top=50)
        ax.set_zlim(bottom=-20,top=20)
        i = i+1
    plt.show()

def imgC():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid()
    ## point P
    plt.plot(7,6,5,marker = 'o', color = 'blue')
    ## world coordinate origin
    plt.plot(6, -5, 5, marker = 'x', color = 'red')
    
    ## line from world origin to P
    plt.plot((6,8), (-5,10), (5,5), color = 'red', linestyle = 'dashed')
    ## camera coordinate origin
    plt.plot(15, 5.5, 5, marker = 'x', color = 'green')
    ## line from camera coordinate origin to P
    plt.plot((15,0), (6,5.5), (5,5), color = 'green', linestyle = 'dashed')
    ## optical axis
   
    plt.plot(10, 5.5, 5, marker = 'o', color = 'black')
    plt.plot(7, 0, 5, marker = 'o', color = 'black')
    ## image plane
    rec1 = plt.Rectangle((3,2), 5, 5, fc=(0,0,1,0.2))
    ax.add_patch(rec1)
    art3d.patch_2d_to_3d(rec1, z=10, zdir='x')
    rec2 = plt.Rectangle((3,2), 5, 5, fc=(0,0,1,0.2))
    ax.add_patch(rec2)
    art3d.patch_2d_to_3d(rec2, z=0, zdir='y')

    ax.set_xlim(0,15)
    ax.set_ylim(-5,10)
    ax.set_zlim(0,15)
    plt.show()