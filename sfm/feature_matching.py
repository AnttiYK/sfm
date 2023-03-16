import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
'''
Brute force matching for 
'''
def bfMatch(f1, f2):
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck = True)
    matches_index = np.empty((len(f1), 2), dtype=np.int32)
    for i in range(len(f1)):
        best_distance = int(10000000)
        p1 = f1[i]
        idx = 0
        for j in range(len(f2)):
            p2 = f2[j]
            dist = np.linalg.norm(p1-p2)
            if(dist < best_distance):
                best_distance = dist
                idx = j
        matches_index[i][0] = i
        matches_index[i][1] = idx
    return matches_index

def get_matches(kp1, kp2, idx):
    k1 = []
    k2 = []
    for i in range(len(idx)):
        i1 = idx[i][0]
        i2 = idx[i][1]
        k1.append(kp1[i1])
        k2.append(kp2[i2])
    return k1, k2

def normalize_points(pts):
    mean = np.mean(pts, axis=0)
    std = np.std(pts, axis=0)
    scale = np.sqrt(2) / np.mean(std)
    T = np.array([[scale, 0, -scale*mean[0]],
              [0, scale, -scale*mean[1]],
              [0, 0, 1]])
    return T.dot(pts.T).T




def fundamental_matrix(p1, p2):
    # Normalize the points to improve numerical stability
    T1 = normalize_points(p1)
    T2 = normalize_points(p2)
    
    # Construct the matrix A
    A = np.zeros((len(p1), 9))
    for i in range(len(p1)):
        x1, y1, _ = p1[i]
        x2, y2, _ = p2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Solve for the nullspace of A using SVD
    U, S, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape((3, 3))
    
    # Enforce the rank-2 constraint on F
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(U, np.dot(np.diag(S), Vt))
    F = np.dot(F, T1.T)
    # Denormalize the fundamental matrix
    F = np.dot(T2.T, F.T)
    
    return F

def ransac_fundamental(x1,x2):
    iters = 1000
    t = 1
    x1 = np.transpose(x1)
    x2 = np.transpose(x2)
    sz = 8
    num_inliers = 0
    for i in range(iters):
        rand_idx = np.random.randint(sz, size=(1,8))
        points1 = x1[rand_idx][0]

        points2 = x2[rand_idx][0]

        p1 = np.empty((sz, 2), dtype=np.float32)
        p2 = np.empty((sz, 2), dtype=np.float32)
        for i in range(sz):
            p1[i] = points1[i].pt
            p2[i] = points2[i].pt
        ##convert to homogenous
        pts1 = np.ones((sz, 3))
        pts2 = np.ones((sz, 3))
        for i in range(sz):
            pts1[i, :2] = p1[i]
            pts2[i, :2] = p2[i]

        F = fundamental_matrix(pts1, pts2)

        # Compute the epipolar lines in both views
        lines1 = np.dot(F, pts2.T).T
        lines2 = np.dot(F.T, pts1.T).T
        
        # Compute the distance between each point and its epipolar line
        dist1 = np.abs(np.sum(pts1 * lines1, axis=1))
        dist2 = np.abs(np.sum(pts2 * lines2, axis=1))
        dist = dist1 + dist2
        
        # Count the number of inliers
        inliers = np.where(dist < t)[0]
        num_inliers_cur = len(inliers)
        
        # Update the best estimate if we have more inliers
        if num_inliers_cur > num_inliers:
            num_inliers = num_inliers_cur
            best_inliers = inliers
            best_F = fundamental_matrix(pts1[inliers], pts2[inliers])
    
    return F


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