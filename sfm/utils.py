import cv2
import os
import numpy as np

'''
Reads images from dir and returns them as array
'''
def readImages(dir, Color):
    imgs = []
    for i in sorted(os.listdir(dir)):
        img =  cv2.imread(os.path.join(dir, i))
        if(Color == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgs.append(np.asarray(img))
    return np.asarray(imgs)

def pts2ply(pts,filename='out.ply'): 
    f = open(filename,'w')
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write('element vertex {}\n'.format(pts.shape[0]))
    
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    
    f.write('property uchar red\n')
    f.write('property uchar green\n')
    f.write('property uchar blue\n')
    
    f.write('end_header\n')
    
    for pt in pts: 
        f.write('{} {} {} 255 255 255\n'.format(pt[0],pt[1],pt[2]))
    f.close()




