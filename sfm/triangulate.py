import numpy as np
import cv2
import matplotlib.pyplot as plt
from cameraPose import PlotCamera
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as mpatches

def triangulate(x1, x2, K, R, t):

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