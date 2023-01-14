import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.patches as mpatches
import cv2 as cv
import numpy as np

'''
Visualizations used in thesis
uncomment lines at visualize function at bottom to display plots
'''

## shows point in 2d and 3d space 
def dspace():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid()
    plt.plot(5,5,5,marker = 'o')
 
    bx = fig.add_subplot(1, 2, 2)
    bx.set_xlabel('x')
    bx.set_ylabel('y')
    plt.grid()
    plt.plot(5,5, marker = 'o')
    plt.show()

## shows point in 3d wolrd coordinates
def worldC():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid()
    ## point P
    plt.plot(5,5,5,marker = 'o', color = 'blue')
    ## world coordinate origin
    plt.plot(0, 0, 0, marker = 'x', color = 'red')
    plt.plot((0,3), (0,0), (0,0), color = 'red')
    plt.plot((0,0),(0,-3), (0,0), color = 'red')
    plt.plot((0,0), (0,0), (0,3), color = 'red')
    ax.text(1,0,0, 'origin')
    ax.text(0,5,6, 'P(Xw, Yw, Zw)')
    ax.set_xlim(-10,10)
    ax.set_ylim(-10,10)
    ax.set_zlim(0,10)
    plt.show()

## shows point in 3d world and camera coordinates
def camC():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    plt.grid()
    ## point P
    plt.plot(5,5,5,marker = 'o', color = 'blue')
    ## world coordinate origin
    plt.plot(0, 0, 0, marker = 'x', color = 'red')
    plt.plot((0,3), (0,0), (0,0), color = 'red')
    plt.plot((0,0),(0,-3), (0,0), color = 'red')
    plt.plot((0,0), (0,0), (0,3), color = 'red')
    ## camera coordinate origin
    plt.plot(15, 6, 3, marker = 'x', color = 'green')
    plt.plot((15,18), (6,6), (3,3), color = 'green')
    plt.plot((15,15),(6,3), (3,3), color = 'green')
    plt.plot((15,15), (6,6), (3,6), color = 'green')
    ax.text(1,0,0, 'origin')
    ax.text(5,5,6, '(Xc, Yc, Zc) P (Xw, Yw, Zw)')
    ax.text(16,6,3, 'camera')
    ax.set_xlim(-10,20)
    ax.set_ylim(-10,10)
    ax.set_zlim(0,10)
    plt.show()

## shows point in image coordinates
def imgC():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.grid()
    ## point P
    plt.plot(5,5,5,marker = 'o', color = 'blue')
    ## world coordinate origin
    plt.plot(0, 0, 0, marker = 'x', color = 'red')
    plt.plot((0,3), (0,0), (0,0), color = 'red')
    plt.plot((0,0),(0,-3), (0,0), color = 'red')
    plt.plot((0,0), (0,0), (0,3), color = 'red')
    ## line from world origin to P
    plt.plot((0,5), (0,5), (0,5), color = 'red', linestyle = 'dashed')
    ## camera coordinate origin
    plt.plot(15, 6, 3, marker = 'x', color = 'green')
    plt.plot((15,18), (6,6), (3,3), color = 'green')
    plt.plot((15,15),(6,3), (3,3), color = 'green')
    plt.plot((15,15), (6,6), (3,6), color = 'green')
    ## line from camera coordinate origin to P
    plt.plot((15,5), (6,5), (3,5), color = 'green', linestyle = 'dashed')
    ## optical axis
    plt.plot((15,-10), (6,6), (3,3), color = 'black')
    plt.plot((5,5), (6,5), (3,3), color = 'black')
    plt.plot((5,5), (5,5), (3,5), color = 'black')
    plt.plot((10,10), (6,5.5), (3,3), color = 'black')
    plt.plot((10,10), (5.5, 5.5), (3, 4), color = 'black')
    plt.plot(10, 5.5, 4, marker = 'o', color = 'black')
    ## image plane
    rec = plt.Rectangle((3,2), 5, 5, fc=(0,0,1,0.2))
    ax.add_patch(rec)
    art3d.patch_2d_to_3d(rec, z=10, zdir='x')
    ax.text(1,0,0, 'origin')
    ax.text(5,5,6, 'P')
    ax.text(10, 5.5, 5, 'P\'')
    ax.text(16,6,3, 'camera')
    ax.text(-10, 6, 3, 'optical axis')
    ax.set_xlim(-10,20)
    ax.set_ylim(-10,10)
    ax.set_zlim(0,10)
    plt.show()

## shows grayscale image determined in visualize
def image(img):
    fig, ax = plt.subplots()
    plt.axis([0,1200, 0, 800])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.gca().invert_yaxis()
    plt.imshow(gray, cmap='gray')
    ax.xaxis.tick_top()
    plt.show()

## shows colormap representation of part of image
def subImage(img):
    fig, ax = plt.subplots()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax.matshow(gray[0:5, 0:5], cmap = 'Greys')
    for i in range(5):
        for j in range(5):
            c = gray[j,i]
            ax.text(i, j, str(c))
    plt.show()

## visualization of binocular disparity
def binocularD():
    fig, ax = plt.subplots()
    ##hide grid and frames
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ## left camera center line
    plt.plot((1,1), (1,5), color = 'black')
    ## right camera center line
    plt.plot((5,5), (1,5), color = 'black')
    ##T
    plt.plot((1,5), (1,1), color = 'black', linestyle = 'dashed')
    ax.text(3, 1.2, 'T')
    ##left camera
    plt.plot(1,2, marker = 'x', color = 'black' )
    ax.text(0.8, 2.1, 'Cl')
    ##right camera
    plt.plot(5, 2, marker = 'x', color = 'black')
    ax.text(5.2, 2.1, 'Cr')
    ##left image
    plt.plot((0,2), (2.5,2.5), color = 'blue')
    ##right image
    plt.plot((4,6), (2.5, 2.5), color = 'blue')
    ##f
    plt.plot((0, 5), (2,2), color = 'black')
    plt.plot((0.25, 0.25), (2, 2.5), color = 'black', linestyle = 'dashed')
    ax.text(0, 2.25, 'f')
    ##P
    plt.plot(3, 5, marker = 'x', color = 'black')
    ax.text(3, 5.25, 'P')
    ##P to Cl
    plt.plot((1,3), (2,5), color = 'blue', linestyle = 'dashed')
    ##P to Cr
    plt.plot((5,3), (2,5), color = 'blue', linestyle = 'dashed')
    ##X left
    plt.plot(1.3, 2.5, marker = 'o', color = 'blue')
    ax.text(1.2, 2.6, 'Xl')
    ##X right
    plt.plot(4.7, 2.5, marker = 'o', color = 'blue')
    ax.text(4.6, 2.6, 'Xr')
    ##Z
    plt.plot((3,3), (2, 5), color = 'black', linestyle = 'dashed')
    ax.text(3.2, 3.5, 'Z')
    plt.show()

## visualization of motion parallax
def motionParallax():
    fig, ax = plt.subplots()
    ##hide grid and frames
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ##left camera
    plt.plot(1, 1, marker = 'x', color = 'black')
    ax.text(0.8, 1, 'Cl')
    ##right camera
    plt.plot(5, 1, marker = 'x', color = 'black')
    ax.text(5.2, 1, 'Cr')
    ##point A
    plt.plot(1, 5, marker = 'o', color = 'blue')
    ax.text(1, 5.2, 'A1', color = 'blue')
    plt.plot(3, 5, marker = 'o', color = 'blue')
    ax.text(3, 5.2, 'A2', color = 'blue')
    plt.plot((1,3), (5,5), color = 'blue', linestyle = 'dashed')
    ##point B
    plt.plot(1, 4, marker = 'o', color = 'red')
    ax.text(1, 4.2, 'B1', color = 'red')
    plt.plot(3, 4, marker = 'o', color = 'red')
    ax.text(3, 4.1, 'B2', color = 'red')
    plt.plot((1, 3), (4, 4), color = 'red', linestyle = 'dashed')
    ##Z
    plt.plot((0.2, 0.2), (1,5), color = 'black', linestyle = 'dashed')
    ax.text(0.3, 3, 'Z', color = 'black')
    plt.plot((0.2, 5.5), (1, 1), color = 'black', linestyle = 'dotted')
    plt.plot((0.3, 0.3), (4, 5), color = 'black', linestyle = 'dashed')
    ax.text(0.4, 4.5, 'dZ', color = 'black')
    ##left camera to A
    plt.plot((1, 1), (0.5, 5), color = 'black')
    plt.plot((1,3), (1, 5), color = 'blue')
    ##left camera to B2
    plt.plot((1, 3), (1, 4), color = 'red')
    ##right camera to A2
    plt.plot((5, 3), (1, 5), color = 'blue')
    ##right camera to B2
    plt.plot((5, 3), (1, 4), color = 'red')
    ##a arc
    aarc = mpatches.Arc((1, 1), 1, 1, theta1 = 62, theta2 = 90)
    ax.add_patch(aarc)
    ax.text(1.1, 1.6, "$\u03B1$")
    ##theta arcs
    tarc1 = mpatches.Arc((1.5, 2), 1, 1, theta1 = 40, theta2 = 65)
    ax.add_patch(tarc1)
    ax.text(1.85, 2.5, '$\\theta$')
    tarc2 = mpatches.Arc((4.5, 2), 1, 1, theta1 = 118, theta2 = 140)
    ax.add_patch(tarc2)
    ##dA arc
    daarc = mpatches.Arc((3, 5), 1, 1, theta1 = 242, theta2 = 295)
    ax.add_patch(daarc)
    ax.text(2.9, 4.3, 'd(A)')
    ##dbarc
    dbarc = mpatches.Arc((3, 4), 1, 1, theta1 = 238, theta2 = 302)
    ax.add_patch(dbarc)
    ax.text(2.9, 3.3, 'd(B)')
    plt.show()

## visualization of sphere in 2 and 3 dimensions
def sphere23D():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,2, projection='3d')
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, linewidth=0.0)
    bx = fig.add_subplot(1, 2, 1)
    c = plt.Circle((0.5, 0.5), 0.5)
    bx.add_patch(c)
    plt.show()

## visualization of shape from silhouette
def shapeFromSilhouette():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ##P1
    plt.plot(1, 5, 5, marker = 'o', color = 'black')
    ax.text(1,5,5.2, 'P1')
    ##image plane for P1
    p1Plane = plt.Rectangle((3 , 3), 4, 4, fc=(0, 0, 1, 0.2))
    ax.add_patch(p1Plane)
    art3d.patch_2d_to_3d(p1Plane, z= 3, zdir='x')
    ##shape on P1 image plane
    p1Shape = plt.Rectangle((4, 4), 2, 2, fc = (0, 0, 1, 1))
    ax.add_patch(p1Shape)
    art3d.patch_2d_to_3d(p1Shape, z = 3, zdir='x')
    ##P1 cone
    plt.plot((1, 10), (5,1), (5, 10), color = 'black') 
    plt.plot((1, 10), (5, 9), (5, 10), color = 'black')
    plt.plot((1, 10), (5, 1), (5, 1), color = 'black')
    plt.plot((1, 10), (5, 9), (5, 1), color = 'black')
    ##P1 projection
    p1Projection = plt.Rectangle((2.7, 2.7), 4.5, 5, fc = (1, 0, 0, 0.2))
    ax.add_patch(p1Projection)
    art3d.patch_2d_to_3d(p1Projection, z = 6, zdir= 'x')
    ##P2
    plt.plot(8, 14, 5, marker = 'o', color = 'black')
    ax.text(8, 14, 5.2, 'P2')
    ##image plane for P2
    p2Plane = plt.Rectangle((6 , 3), 4, 4, fc=(0, 0, 1, 0.2))
    ax.add_patch(p2Plane)
    art3d.patch_2d_to_3d(p2Plane, z= 12, zdir='y')
    ##shape on P2 image plane
    p2Shape = plt.Rectangle((7, 4), 2, 2, fc = (0, 0, 1, 1))
    ax.add_patch(p2Shape)
    art3d.patch_2d_to_3d(p2Shape, z = 12, zdir='y')
    ##P2 cone
    plt.plot((8, 3.7), (14, 4), (5, 9.2), color = 'black')
    plt.plot((8, 3.7), (14, 4), (5, 1.2), color = 'black')
    plt.plot((8, 12.3), (14, 4), (5, 9.2), color = 'black')
    plt.plot((8, 12.3), (14, 4), (5, 1.2), color = 'black')
    ##P2 projection
    p2Projection = plt.Rectangle((6, 2.7), 5, 5.2, fc = (1, 0, 0, 0.2))
    ax.add_patch(p2Projection)
    art3d.patch_2d_to_3d(p2Projection, z = 7.2, zdir= 'y')
    ##limits
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_zlim(0, 10)
    plt.show()

## displays silhouette of object using foreground extraction
def silhouette(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    fgbg = cv.bgsegm.createBackgroundSubtractorMOG()
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    plt.imshow(gray, cmap = 'Greys')
    ax = fig.add_subplot(1,2,2)
    estimatedThreshold, thresholdImage=cv.threshold(gray,90,255,cv.THRESH_BINARY) 
    plt.imshow(thresholdImage, cmap = 'Greys')
    plt.show()

## Linear perspective visualization
def linearPerspective():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    
    ##cube
    axes = [5, 5, 5]
    data = np.ones(axes)
    ax.voxels(data)
    ##lines
    plt.plot((5,5), (5,1000), (5,5), color = 'black')
    ax.text(5, 5, 5.2, '(5, 5, 5)', color = 'black')
    ax.text(-3, 5, 5, '(0, 5, 5)', color = 'black')
    plt.rcParams.update({'font.size': 6})
    ax.text(5, 1000, 5.2, '(5, 1000, 5)', color = 'black')
    ax.text(-18, 1000, 5.2, '(0, 1000, 5)', color = 'black')
    plt.plot((0, 0), (5, 1000), (5, 5), color = 'black')
    ##limits
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(0, 20)
    ##hide grid and frames
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()


'''
uncomment to show plot
dspace = point in 2 and 3 dimensional space
worldC = point in world coordinates
camC = point in camera coordinates
imgC = point in image coordinates
image = greyscale img
subimage = colormap representation of img with color values
binocularD = binocular disparity visualization
motionParallax = motion parallax visualization
sphere23D = 2 and 3D representations of sphere
shapeFromSilhouette = shape from silhouette visualization
silhouette = displays silhouette of object
'''
def visualize(images):
    img = images[0]
    #dspace()
    #worldC()
    #camC()
    #imgC()
    #image(img)
    #subImage(img)
    #binocularD()
    #motionParallax()
    #sphere23D()
    #shapeFromSilhouette()
    #silhouette(img)
    linearPerspective()