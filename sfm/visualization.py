import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import cv2 as cv

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

def image(img):
    fig, ax = plt.subplots()
    plt.axis([0,1200, 0, 800])
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    plt.gca().invert_yaxis()
    plt.imshow(gray, cmap='gray')
    ax.xaxis.tick_top()
    plt.show()

def subImage(img):
    fig, ax = plt.subplots()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ax.matshow(gray[0:5, 0:5], cmap = 'Greys')
    for i in range(5):
        for j in range(5):
            c = gray[j,i]
            ax.text(i, j, str(c))
    plt.show()

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
    

## uncomment functions to show plots
def visualize(images):
    img = images[0]
    #dspace()
    #worldC()
    #camC()
    #imgC()
    #image(img)
    #subImage(img)
    binocularD()