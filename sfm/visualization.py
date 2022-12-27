import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

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

## uncomment functions to show plots
def visualize():
    #dspace()
    #worldC()
    #camC()
    imgC()