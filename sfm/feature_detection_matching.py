import cv2
import matplotlib.pyplot as plt
import random        
import numpy as np
from tqdm import tqdm

def feature_detect_match(imgs):
    N = len(imgs)
    detector = cv2.AKAZE_create(nOctaves=8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    features = np.empty((N,N), dtype=object)
    m = np.empty(N)
    for i in tqdm(range(0, N-1)):
        #extract features from images
        img_src = imgs[i]
        kp_src = detector.detect(img_src, None)
        kp_src, des_src = detector.compute(img_src, kp_src)
        des_dst=[]
        for j in range(i+1, N):
            #extract features from destination image
            img_dst = imgs[j]
            kp_dst = detector.detect(img_dst, None)
            kp_dst, des_dst = detector.compute(img_dst, kp_dst)
            # get matches     
            matches = matcher.match(des_src, des_dst)
            matches = sorted(matches, key = lambda x:x.distance)
            #sort features with matches
            index_src = np.array([m.queryIdx for m in matches])
            index_dst = np.array([m.trainIdx for m in matches])
            ## Filter out non matched keypoints
            kp_src_ = (np.array(kp_src))[index_src]
            kp_dst_ = (np.array(kp_dst))[index_dst]
            des_dst_ = (np.array(des_dst))[index_dst]
            ## Coordinates for keypoints
            cord_src = np.array([k.pt for k in kp_src_])
            cord_dst = np.array([k.pt for k in kp_dst_])
            features[i,j]=[cord_src,cord_dst]
           

    return features

    

    
    
def showFeatures(kps, imgs):
    r = random.randint(0, len(imgs)-1)
    img = imgs[r]
    kp = kps[r]['kp']
    img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0), flags=0)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.imshow(img2), plt.show()

