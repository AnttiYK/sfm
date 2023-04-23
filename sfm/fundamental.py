import numpy as np
import cv2
import matplotlib.pyplot as plt

def epipolar(x, i, F):
    if x.shape[1] == 2:
        x = cv2.convertPointsToHomogeneous(x)[:,0,:]
    if i == 1:
        e = F.dot(x.T)
    elif i == 2:
        e = F.T.dot(x.T)
    return e.T

## Eight point algorithm to solve fundamental matrix
def fundamentalMatrix(x1, x2):
    
    
    A = np.zeros((x1.shape[0], 9))
    
    x1_ = x1.repeat(3, axis= 1)
    x2_ = np.tile(x2, (1, 3))
    
    A = np.multiply(x1_, x2_)
    
    _, _, V = np.linalg.svd(A)
    F = V[-1,:].reshape((3,3), order = 'F')
    
    U, S, V = np.linalg.svd(F)
    F = U.dot(np.diag(S).dot(V))
    
    F = F/F[-1, -1]
    
    return F

## Calculates the sampson error
def error(F, x1, x2):
    n = np.sum(x1.dot(F) * x2, axis=-1)
    F_ = np.dot(F, x1.T)
    Ft_ = np.dot(F.T, x2.T)
    Fs_ = np.sum(x2 * F_.T, axis= 1)
    
    return np.abs(Fs_) / np.sqrt(F_[0]**2 + F_[1]**2 + Ft_[0]**2 + Ft_[1]**2)
    

def ransacFundamental(x1, x2):
    p = 0.99
    t = 0.1
    iters = 2000
    inliers = 0
    F = None
    mask = None
    ## Convert to homogenous
    if x1.shape[1] == 2:
        x1 = cv2.convertPointsToHomogeneous(x1)[:,0,:]
        x2 = cv2.convertPointsToHomogeneous(x2)[:,0,:]
    for i in range(iters):
        mask_ = np.random.randint(low=0, high = x1.shape[0], size=(8,))
        x1_ = x1[mask_]
        x2_ = x2[mask_]
        
        F_ = fundamentalMatrix(x1_, x2_)
        e = error(F_, x1, x2)
        
        mask_ = e < t
        inliers_ = np.sum(mask_)
        
        if inliers < inliers_:
            inliers = inliers_
            F = F_
            mask = mask_
            
    F = fundamentalMatrix(x1[mask], x2[mask])
    return F, mask

def drawlines(img1,img2,lines,pts1,pts2,drawOnly=None,linesize=3,circlesize=10):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape

    img1_, img2_ = np.copy(img1), np.copy(img2)

    drawOnly = lines.shape[0] if (drawOnly is None) else drawOnly

    i = 0 
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        
        img1_ = cv2.line(img1_, (x0,y0), (x1,y1), color,linesize)
        img1_ = cv2.circle(img1_,tuple(pt1.astype(int)),circlesize,color,-1)
        img2_ = cv2.circle(img2_,tuple(pt2.astype(int)),circlesize,color,-1)

        i += 1 
        
        if i > drawOnly: 
            break 

    return img1_,img2_

def showEpipolar(i1, i2, e1, e2, x1, x2):
    tup = drawlines(i1,i2,e2,x2,x1,drawOnly=10, linesize=10,circlesize=30)
    epilines2 = np.concatenate(tup[::-1],axis=1) #reversing the order of left and right images

    plt.figure(figsize=(9,4))
    plt.imshow(epilines2)

    tup = drawlines(i1,i2,e1,x1,x2,drawOnly=10,linesize=10,circlesize=30)
    epilines1 = np.concatenate(tup,axis=1) 

    plt.figure(figsize=(9,4))
    plt.imshow(epilines1)
    plt.show()
        