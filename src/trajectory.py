import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Trajectory():
    def __init__(self, file_name):
        """[summary]

        Args:
            file_name ([type]): [description]
        """
        pose = pd.read_csv(file_name, sep=' ', names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'])
        self.pose = np.asarray(pose).reshape(-1, 3, 4)
        self.trajectory = self._trajectory(self.pose)
        self.name = file_name.split('/')[2].split('.')[0].split('_')[2]
        
        if(self.name == 'gt'): self.is_gt = True
        else: self.is_gt = False
        
    def _trajectory(self, x):
        origin = np.matrix([[0],
                            [0],
                            [0],
                            [1]])
        points = []
        for i in range(x.shape[0]):
            mat = np.asmatrix(x[i])
            transform = mat*origin
            points.append(np.asarray(transform))
        traj = np.asarray(points).reshape(-1,3)
        return traj
        
def plotXYZ(*traj):
    n_files = len(traj)
    
    plt.figure(figsize=(10,15))
    plt.subplot(3,1,1)
    for i in range(n_files):
        if (traj[i].is_gt): plt.plot(traj[i].trajectory[:,0], label=traj[i].name, ls='--')
        else: plt.plot(traj[i].trajectory[:,0], label=traj[i].name)
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3,1,2)
    for i in range(n_files):
        if (traj[i].is_gt): plt.plot(traj[i].trajectory[:,1], label=traj[i].name, ls='--')
        else: plt.plot(traj[i].trajectory[:,1], label=traj[i].name)
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3,1,3)
    for i in range(n_files):
        if (traj[i].is_gt): plt.plot(traj[i].trajectory[:,2], label=traj[i].name, ls='--')
        else: plt.plot(traj[i].trajectory[:,2], label=traj[i].name)
    plt.ylabel('z')
    plt.xlabel('index')
    plt.legend()

def plot2D(*traj):
    n_files = len(traj)
    
    plt.figure(figsize=(10,10))
    for i in range(n_files):
        if (traj[i].is_gt): plt.plot(traj[i].trajectory[:,0], traj[i].trajectory[:,2], label=traj[i].name, ls='--')
        else: plt.plot(traj[i].trajectory[:,0], traj[i].trajectory[:,2], label=traj[i].name)
    plt.xlabel("x")
    plt.ylabel("z")
    plt.legend()
    
def plot3D(*traj):
    n_files = len(traj)
    
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_files):
        ax.scatter(traj[i].trajectory[:,0], traj[i].trajectory[:,2], -traj[i].trajectory[:,1], 
                   label=traj[i].name)
    ax.legend()
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('-y')
    
def show():
    plt.show()