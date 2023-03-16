import numpy as np
import cv2

def essential_matrix(F, K):
    E = np.matmul(K.T, F, K)
    return E

def estimate_pose(E, pts1, pts2, K):
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    U, _, Vt = np.linalg.svd(E)
    R1 = U.dot(W).dot(Vt)
    t1 = U[:, 2]
    R2 = U.dot(W.T).dot(Vt)
    t2 = -U[:, 2]
    pts1 = np.array([kp.pt for kp in pts1], dtype=np.float32)
    pts2 = np.array([kp.pt for kp in pts2], dtype=np.float32)
    
    P1 = K.dot(np.hstack((np.eye(3), np.zeros((3, 1)))))
    P2_1 = K.dot(np.hstack((R1, t1.reshape((3, 1)))))
    P2_2 = K.dot(np.hstack((R1, t2.reshape((3, 1)))))
    P2_3 = K.dot(np.hstack((R2, t1.reshape((3, 1)))))
    P2_4 = K.dot(np.hstack((R2, t2.reshape((3, 1)))))

    # Extract possible 3D point correspondences from essential matrix
    x1_hom = np.vstack((np.transpose(pts1), np.ones((1, len(pts1)))))
    x2_hom = np.vstack((np.transpose(pts2), np.ones((1, len(pts2)))))
    pts3d_1 = cv2.triangulatePoints(P1, P2_1, pts1, pts2)
    pts3d_2 = cv2.triangulatePoints(P1, P2_2, x1_hom, x2_hom)
    pts3d_3 = cv2.triangulatePoints(P1, P2_3, x1_hom, x2_hom)
    pts3d_4 = cv2.triangulatePoints(P1, P2_4, x1_hom, x2_hom)
    return W