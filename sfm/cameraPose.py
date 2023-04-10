import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import cv2

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

def disambiguatePose(x1, x2, R1, R2, t, K):
    sets = [None, None, None, None]
    sets[0] = (R1, t, triangulate(x1, x2, K, R1, t))
    sets[1] = (R1, -t, triangulate(x1, x2, K, R1, -t))
    sets[2] = (R2, t, triangulate(x1, x2, K, R2, t))
    sets[3] = (R2, -t, triangulate(x1, x2, K, R2, -t))
    count = -1
    bestR = None
    bestT = None
    for R_, t_, p_ in sets:
        c1 = p_[:,-1] > 0
        c2 = (R_.dot(p_.T)+t_).T
        c2 = c2[:,-1] > 0
        count_ = np.sum(c1 & c2)
        if count_ > count:
            count = count_
            bestR, bestT = R_, t_
    return bestR, bestT, count
    
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