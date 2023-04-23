import os
import cv2
import numpy as np

def readImages(dir, color):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv2.imread(os.path.join(dir, i))
        if(color == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(np.asarray(img))
    return np.asarray(imgs)

def pts2ply(pts,filename='out.ply'): 
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