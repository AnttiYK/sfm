import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_e_matrix(K, F):
    E = K.T.dot(F).dot(K)
    u,s,v = np.linalg.svd(E)
    s = [1,1,0]
    E_ = np.dot(u,np.dot(np.diag(s),v))
    return E_

def estimate_f_matrix(points_src, points_dst):
    N = len(points_src)
    m_src = np.average(points_src, axis=0)
    m_dst = np.average(points_dst, axis=0)
    m_src_mean = points_src-m_src.reshape(1,2)
    m_dst_mean = points_dst-m_dst.reshape(1,2)
    sum_src = np.sum((m_src_mean)**2, axis=None)
    sum_dst = np.sum((m_dst_mean)**2, axis=None)
    s_src = (sum_src/(2*N))**0.5
    s_src_inv = 1/s_src
    s_dst = (sum_dst/(2*N))**0.5
    s_dst_inv = 1/s_dst
    x = m_src_mean*s_src_inv
    y = m_dst_mean*s_dst_inv
    Y = np.ones((N,9))
    Y = np.ones((N,9))	
    Y[:,0:2] = x*y[:,0].reshape(N,1)
    Y[:,2] = y[:,0]
    Y[:,3:5] = x*y[:,1].reshape(N,1)
    Y[:,5] = y[:,1]
    Y[:,6:8] = x
    _,_,vt = np.linalg.svd(Y,full_matrices=True)
    F = vt[8,:].reshape(3,3)
    U, S, Vt = np.linalg.svd(F, full_matrices=True)
    S[2] = 0
    Smat = np.diag(S)
    F = np.dot(U, np.dot( Smat, Vt))
    T_src = np.zeros((3,3))
    T_src[0,0] = s_src_inv
    T_src[1,1] = s_src_inv
    T_src[2,2] = 1
    T_src[0,2] = -s_src_inv*m_src[0]
    T_src[1,2] = -s_src_inv*m_src[1]

    T_dst = np.zeros((3,3))
    T_dst[0,0] = s_dst_inv
    T_dst[1,1] = s_dst_inv
    T_dst[2,2] = 1
    T_dst[0,2] = -s_dst_inv*m_dst[0]
    T_dst[1,2] = -s_dst_inv*m_dst[1]
    F = np.dot( np.transpose( T_dst ), np.dot( F, T_src))
    return F

def get_f_matrix(points_src, points_dst):
    N = 1500
    S = points_dst.shape[0]
    r = np.random.randint(S,size=(N,8))
    m_src = np.ones((3,S))
    m_src[0:2,:]=points_src.T
    m_dst = np.ones((3,S))
    m_dst[0:2,:]=points_dst.T
    count = np.zeros(N)
    cost = np.zeros(S)
    t=1e-2

    for i in tqdm(range(N)):
        #cost_ = np.zeros(8)
        F = estimate_f_matrix(points_src[r[i,:],:], points_dst[r[i,:],:])
        for j in range(S):
            cost[j] = np.dot(np.dot(m_dst[:,j].T,F),m_src[:,j])
        inliers = np.absolute(cost)<t
        count[i]=np.sum(inliers+np.zeros(S), axis=None)

    index = np.argsort(-count)
    best = index[0]
    best_F = estimate_f_matrix(points_src[r[best,:],:], points_dst[r[best,:],:])
    for i in range(S):
        cost[i]=np.dot(np.dot(m_dst[:,i].T, best_F), m_src[:,i])
    confidence = np.absolute(cost)
    index = np.argsort(confidence)
    inliers_src = points_src[index]
    inliers_dst = points_dst[index]
    inliers_src = inliers_src[:100,:]
    inliers_dst = inliers_dst[:100,:] 

    return best_F, inliers_src, inliers_dst     

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
        