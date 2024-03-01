import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

def get_camera_pose(E):
    U, S, V_T = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R = []
    C = []
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    R.append(np.dot(U, np.dot(W.T, V_T)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    for i in range(4):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]

    return R, C


def triangulate(x1, x2, K, R, t):

    x1hom = cv2.convertPointsToHomogeneous(x1)[:,0,:]
    x2hom = cv2.convertPointsToHomogeneous(x2)[:,0,:]
    
    x1norm = (np.linalg.inv(K).dot(x1hom.T)).T
    x2norm = (np.linalg.inv(K).dot(x2hom.T)).T
    
    x1norm = cv2.convertPointsFromHomogeneous(x1norm)[:,0,:]
    x2norm = cv2.convertPointsFromHomogeneous(x2norm)[:,0,:]
    
    points4d = cv2.triangulatePoints(np.eye(3,4), np.hstack((R, t)), x1norm.T, x2norm.T)
    points3D = cv2.convertPointsFromHomogeneous(points4d.T)[:,0,:]
    
    return points3D

def get_pos_depth(points, r, c):
    n = 0
    for p in points:
        p = p.reshape(-1,1)
        if r.dot(p-c) > 0 and p[2]>0:
            n+=1
    return n

def disambiguatePose(r, c, points):
    best_idx = 0
    max_pos_depth = 0
    for i in range(len(r)):
        R, C = r[i], c[i].reshape(-1, 1)
        r3 = R[2,:].reshape(1,-1)
        points_ = points[i]
        points_ = points_ / points_[:,3].reshape(-1,1)
        points_=points_[:,0:3]
        n_pos_depth = get_pos_depth(points_, r3, C)
        if n_pos_depth > max_pos_depth:
            best_idx = i
            max_pos_depth = n_pos_depth
    R, C, P = r[best_idx], c[best_idx], points[best_idx]

    return R,C,P

    
def cameraPose(E):
    U, _, V = np.linalg.svd(E)
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    
    t = U[:, -1]
    R1 = U.dot(W.dot(V))
    R2 = U.dot(W.T.dot(V))
    
    if np.linalg.det(R1) < 0:
        R1 = R1 * -1
    if np.linalg.det(R2) < 0:
        R2 = R2 * -1
    t = t[:,np.newaxis]
    return R1, R2, t

def PlotCamera(R,t,ax,scale=.5,depth=.5,faceColor='grey'):
    C = -t #camera center (in world coordinate system)

    #Generating camera coordinate axes
    axes = np.zeros((3,6))
    axes[0,1], axes[1,3],axes[2,5] = 1,1,1
    
    #Transforming to world coordinate system 
    axes = R.T.dot(axes)+C[:,np.newaxis]

    #Plotting axes
    ax.plot3D(xs=axes[0,:2],ys=axes[1,:2],zs=axes[2,:2],c='r')
    ax.plot3D(xs=axes[0,2:4],ys=axes[1,2:4],zs=axes[2,2:4],c='g')
    ax.plot3D(xs=axes[0,4:],ys=axes[1,4:],zs=axes[2,4:],c='b')

    #generating 5 corners of camera polygon 
    pt1 = np.array([[0,0,0]]).T #camera centre
    pt2 = np.array([[scale,-scale,depth]]).T #upper right 
    pt3 = np.array([[scale,scale,depth]]).T #lower right 
    pt4 = np.array([[-scale,-scale,depth]]).T #upper left
    pt5 = np.array([[-scale,scale,depth]]).T #lower left
    pts = np.concatenate((pt1,pt2,pt3,pt4,pt5),axis=-1)
    
    #Transforming to world-coordinate system
    pts = R.T.dot(pts)+C[:,np.newaxis]
    ax.scatter3D(xs=pts[0,:],ys=pts[1,:],zs=pts[2,:],c='k')
    
    #Generating a list of vertices to be connected in polygon
    verts = [[pts[:,0],pts[:,1],pts[:,2]], [pts[:,0],pts[:,2],pts[:,-1]],
            [pts[:,0],pts[:,-1],pts[:,-2]],[pts[:,0],pts[:,-2],pts[:,1]]]
    
    #Generating a polygon now..
    ax.add_collection3d(Poly3DCollection(verts, facecolors=faceColor,
                                         linewidths=1, edgecolors='k', alpha=.25))

def showCameraPose(R1, R2, t):
    for R_ in [R1,R2]: 
        for t_ in [t,-t]:

            fig = plt.figure(figsize=(9,6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')

            PlotCamera(np.eye(3,3),np.zeros((3,)),ax)
            PlotCamera(R_,t_,ax)
    plt.show()