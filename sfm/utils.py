import os
import cv2
import numpy as np
def last_4chars(x):
    return(int(x[-8:-4]))

def readImages(dir, color):
    imgs = []
    for i in sorted(os.listdir(dir), key = last_4chars):
        img =  cv2.imread(os.path.join(dir, i))
        if(color == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(np.asarray(img))
    return np.asarray(imgs)

def pts2ply(pts,filename='out.ply'): 
    if(pts.any() > 5):
        return
    else:
        f = open(filename,'w')
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(len(pts[0])))
    
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
    
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
    
        f.write('end_header\n')
        for i in range(len(pts[0])): 
            f.write('{} {} {} 255 255 255\n'.format(pts[0][i],pts[1][i],pts[2][i]))
        f.close()


def ProjectionMatrix(R,C,K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P


def ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K ):
    
    P1 = ProjectionMatrix(R1,C1,K) 
    P2 = ProjectionMatrix(R2,C2,K)

    # X = homo(X.reshape(1,-1)).reshape(-1,1) # make X it a column of homogenous vector
    
    p1_1T, p1_2T, p1_3T = P1 # rows of P1
    p1_1T, p1_2T, p1_3T = p1_1T.reshape(1,-1), p1_2T.reshape(1,-1),p1_3T.reshape(1,-1)

    p2_1T, p2_2T, p2_3T = P2 # rows of P2
    p2_1T, p2_2T, p2_3T = p2_1T.reshape(1,-1), p2_2T.reshape(1,-1), p2_3T.reshape(1,-1)

    ## reprojection error for reference camera points - j = 1
    u1,v1 = pt1[0], pt1[1]
    u1_proj = np.divide(p1_1T.dot(X) , p1_3T.dot(X))
    v1_proj =  np.divide(p1_2T.dot(X) , p1_3T.dot(X))
    E1= np.square(v1 - v1_proj) + np.square(u1 - u1_proj)
    
    ## reprojection error for second camera points - j = 2    
    u2,v2 = pt2[0], pt2[1]
    u2_proj = np.divide(p2_1T.dot(X) , p2_3T.dot(X))
    v2_proj =  np.divide(p2_2T.dot(X) , p2_3T.dot(X))
    
    E2= np.square(v2 - v2_proj) + np.square(u2 - u2_proj)
    
    return E1, E2

def get_projection_error(x3D, pts1, pts2, R1, C1, R2, C2, K ):    
    Error = []
    for pt1, pt2, X in zip(pts1, pts2, x3D):
        e1,e2 = ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K )
        Error.append(e1+e2)
        
    return np.mean(Error)