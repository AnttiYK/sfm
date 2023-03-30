import numpy as np
import cv2

def ComputeReprojections(X,R,t,K): 
    """
    X: (n,3) 3D triangulated points in world coordinate system
    R: (3,3) Rotation Matrix to convert from world to camera coordinate system
    t: (3,1) Translation vector (from camera's origin to world's origin)
    K: (3,3) Camera calibration matrix
    
    out: (n,2) Projected points into image plane"""
    outh = K.dot(R.dot(X.T) + t )
    out = cv2.convertPointsFromHomogeneous(outh.T)[:,0,:]
    return out 

def LinearPnP(X, x, K, isNormalized=False): 
    if X.shape[1]==3:
        X = np.hstack((X, np.ones((X.shape[0],1))))
    if x.shape[1]==2: 
        x = np.hstack((x, np.ones((x.shape[0],1))))

    if isNormalized==False: 
        x = np.linalg.inv(K).dot(x.T).T

    A = np.zeros((X.shape[0]*3,12))

    for i in range(X.shape[0]): 
        A[i*3,:] = np.concatenate((np.zeros((4,)), -X[i,:], x[i,1]*X[i,:]))
        A[i*3+1,:] = np.concatenate((X[i,:], np.zeros((4,)), -x[i,0]*X[i,:]))
        A[i*3+2,:] = np.concatenate((-x[i,1]*X[i,:], x[i,0]*X[i,:], np.zeros((4,))))    
    
    u,s,v = np.linalg.svd(A)
    P = v[-1,:].reshape((4,3),order='F').T
    R, t = P[:,:3], P[:,-1]
    
    u,s,v = np.linalg.svd(R)
    R = u.dot(v)
    t = t/s[0]

    if np.linalg.det(u.dot(v)) < 0:
        R = R*-1
        t = t*-1
    
    return R, t

def ransacPnP(x1, x2, K):
    threshold = 100.0
    iters = 10000

    bestR,bestt,bestmask,bestInlierCount = None,None,None,0

    for i in range(iters): 

        #Randomly selecting 6 points for linear pnp
        mask = np.random.randint(low=0,high=x1.shape[0],size=(6,))
        x1_ = x1[mask]
        x2_ = x2[mask]

        #Estimating pose and evaluating (reprojection error)
        Riter,titer = LinearPnP(x1_,x2_,K,isNormalized=False)

        xreproj = ComputeReprojections(x1_, Riter, titer[:,np.newaxis], K)        
        errs = np.sqrt(np.sum((x2_-xreproj)**2,axis=-1))

        mask = errs < threshold
        numInliers = np.sum(mask)

        #updating best parameters if appropriate
        if numInliers > bestInlierCount: 
            bestInlierCount = numInliers
            bestR,bestt,bestmask = Riter,titer, mask
            bestX1 = x1_
            bestX2 = x2_

    #Final least squares fit on best mask
    R,t = LinearPnP(bestX1,bestX2,K,isNormalized=False)
    return R,t,