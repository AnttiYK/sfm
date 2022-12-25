import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dspace():
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")
    plt.grid()
    plt.plot(5,5,5,marker = 'o')
 
    bx = fig.add_subplot(1, 2, 2)
    bx.set_xlabel('x')
    bx.set_ylabel('y')
    plt.grid()
    plt.plot(5,5, marker = 'o')
    plt.show()