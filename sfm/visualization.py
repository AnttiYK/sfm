import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def dspace():
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    #ax.axis('off')
    plt.show()
