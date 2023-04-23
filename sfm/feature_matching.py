import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from cameraPose import PlotCamera

def plotNewCamera(R, t, Rnew, tnew):
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    PlotCamera(np.eye(3,3),np.zeros((3,)),ax)
    PlotCamera(R,t,ax)
    PlotCamera(Rnew,tnew,ax,faceColor='red')
    plt.show()
    
'''
Brute force matching for 
'''
def matches2D3D(des1,i1,des2, i2, des3, kp3, mask, points3D):
    des1_ = des1[i1][mask]
    des2_ = des2[i2][mask]
    matches = bfMatch(des3, np.concatenate((des1_, des2_), axis = 0))
    
    i3 = np.array([m.queryIdx for m in matches])
    kp3_ = (np.array(kp3))[i3]
    p3 = np.array([kp.pt for kp in kp3_])
    
    ## Filter out already triangulated matches
    ip = np.array([m.trainIdx for m in matches])
    ip[ip >= points3D.shape[0]] = ip[ip >= points3D.shape[0]] - points3D.shape[0]
    points3D_ = points3D[ip]
    
    return p3, points3D_

def bfMatch(f1, f2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches = matcher.match(f1,f2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches

def align_matches(kp1, kp2, matches, des):
    i1 = np.array([m.queryIdx for m in matches])
    i2 = np.array([m.trainIdx for m in matches])

    ## Filter out non matched keypoints
    kp1_ = (np.array(kp1))[i1]
    kp2_ = (np.array(kp2))[i2]
    des_ = (np.array(des))[i2]
    ## Coordinates for keypoints
    c1 = np.array([k.pt for k in kp1_])
    c2 = np.array([k.pt for k in kp2_])
    
    return c1, c2, des_

def align_3D_matches(world_points, img_points, matches, des):
    i1 = np.array([m.queryIdx for m in matches])
    i2 = np.array([m.trainIdx for m in matches])
    ## Filter out non matched keypoints
    world_points = (np.array(world_points))[i1]

    des_ = (np.array(des))[i1]
    img_points = (np.array(img_points))[i2]
    
    ## Coordinates for keypoints
    img_points = np.array([k.pt for k in img_points])
  
    
    return world_points, img_points, des_

def showMatches(images, transformations, features, matches):
    firstImageIndex = 3
    secondImageIndex = 2
    mask = transformations[firstImageIndex][secondImageIndex]['mask']
    draw_params2 = dict(matchColor = (0, 255, 0), singlePointColor = None, matchesMask = mask, flags = 2)
    draw_params1 = dict(matchColor = (255, 0, 0), singlePointColor = None, flags = 2)
    img1 = cv.drawMatches(images[firstImageIndex], features[firstImageIndex]['kp'], images[secondImageIndex], features[secondImageIndex]['kp'], matches[firstImageIndex][secondImageIndex], None, **draw_params1)
    img2 = cv.drawMatches(images[firstImageIndex], features[firstImageIndex]['kp'], images[secondImageIndex], features[secondImageIndex]['kp'], matches[firstImageIndex][secondImageIndex], None, **draw_params2)
    fig = plt.figure()
    ax=fig.add_subplot(1, 1, 1)
    plt.imshow(img1, 'gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(img2, 'gray')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()