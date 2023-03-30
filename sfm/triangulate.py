import numpy as np
import cv2
import matplotlib.pyplot as plt
from cameraPose import PlotCamera

def triangulate(x1, x2, K, R, t, R_, t_):
    if R_ == None:
        R_ = np.eye(3)
        t_ = np.zeros((3,1))
    
    x1hom = cv2.convertPointsToHomogeneous(x1)[:,0,:]
    x2hom = cv2.convertPointsToHomogeneous(x2)[:,0,:]
    
    x1norm = (np.linalg.inv(K).dot(x1hom.T)).T
    x2norm = (np.linalg.inv(K).dot(x2hom.T)).T
    
    x1norm = cv2.convertPointsFromHomogeneous(x1norm)[:,0,:]
    x2norm = cv2.convertPointsFromHomogeneous(x2norm)[:,0,:]
    
    points4d = cv2.triangulatePoints(np.eye(3,4), np.hstack((R, t)), x1norm.T, x2norm.T)
    points3D = cv2.convertPointsFromHomogeneous(points4d.T)[:,0,:]
    
    return points3D

def showTriangulate(R1, R2, t, x1, x2, K, mask):
    sets = [None, None, None, None]
    sets[0] = (R1, t, triangulate(x1[mask], x2[mask], K, R1, t))
    sets[1] = (R1, -t, triangulate(x1[mask], x2[mask], K, R1, -t))
    sets[2] = (R2, t, triangulate(x1[mask], x2[mask], K, R2, t))
    sets[3] = (R2, -t, triangulate(x1[mask], x2[mask], K, R2, -t))
    for s in sets: 
        fig = plt.figure(figsize=(9,6))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        PlotCamera(np.eye(3,3),np.zeros((3,)),ax,scale=5,depth=5)
        PlotCamera(s[0],s[1][:,0],ax,scale=5,depth=5)

        pts3d = s[-1]
        ax.scatter3D(pts3d[:,0],pts3d[:,1],pts3d[:,2])

        ax.set_xlim(left=-50,right=50)
        ax.set_ylim(bottom=-50,top=50)
        ax.set_zlim(bottom=-20,top=20)
    plt.show()