import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import rosbag

class Trajectory():
    def __init__(self, file_name):
        """Get trajectory and pose data from a file

        Args:
            file_name (string): data's file name 
        """
        print("Reading {}".format(file_name))
        if (file_name.endswith('.txt') or file_name.endswith('.csv')):
            pose = pd.read_csv(file_name, sep=' ', names=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12'])
            name = file_name.split('/')[2].split('.')[0].split('_')[2]
            time = None
            length = pose.shape[0]
        elif(file_name.endswith('.bag')):
            with rosbag.Bag(file_name) as bag:
                pose, name, time_dur, length = self._gen_pose(bag)
        else:
            print("unsupported type of data file")
            return
        
        self.pose = np.asarray(pose).reshape(-1, 3, 4) 
        self.trajectory = self._trajectory(self.pose)
        
        self.time = np.array(time_dur)
        self.name = name
        self.length = length
        
        self.is_gt = False
        if(self.name == 'gt' or self.name == 'ground_truth'): self.is_gt = True
        
        print("{} with length {}".format(self.name, self.length))
        
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
    
    def _quaternion2rotation(self, orientation):
        q1 = orientation[0]
        q2 = orientation[1]
        q3 = orientation[2]
        q0 = orientation[3]

        R = np.array([[2*(q0**2 + q1**2)-1, 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2)],
                    [2*(q1*q2 + q0*q3), 2*(q0**2 + q2**2)-1, 2*(q2*q3 - q0*q1)],
                    [2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 2*(q0**2 + q3**2)-1]])
        return R

    def _gen_pose(self, bag):
        pose = []
        time = []
        for topic, msg, _ in bag.read_messages():
            poses = msg.poses
        for msg in poses:
            t = np.array([msg.pose.position.x, 
                          msg.pose.position.y,
                          msg.pose.position.z]).reshape(3, 1)

            R = self._quaternion2rotation([msg.pose.orientation.x, 
                                           msg.pose.orientation.y, 
                                           msg.pose.orientation.z, 
                                           msg.pose.orientation.w])
            pose.append(np.hstack([R, t]).reshape(3, 4))
            time.append((msg.header.stamp-poses[0].header.stamp).to_nsec())
        return pose, topic, time, bag.get_message_count()
        
def plotXYZ(gt, trajs):   
    plt.figure(figsize=(10,15))
    plt.subplot(3,1,1)
    if (gt): plt.plot(gt.time, gt.trajectory[:,0], label=gt.name, ls='--')
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:,0], label=traj.name)
    plt.ylabel('x')
    plt.legend()

    plt.subplot(3,1,2)
    if (gt): plt.plot(gt.time, gt.trajectory[:,1], label=gt.name, ls='--')
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:,1], label=traj.name)
    plt.ylabel('y')
    plt.legend()

    plt.subplot(3,1,3)
    if (gt): plt.plot(gt.time, gt.trajectory[:,2], label=gt.name, ls='--')
    for traj in trajs:
        plt.plot(traj.time, traj.trajectory[:,2], label=traj.name)
    plt.ylabel('z')
    plt.xlabel('time[nano_sec]')
    plt.legend()

def plot2D(option, gt, trajs):
    plt.figure(figsize=(10,10))
    if (option == 'xy'):
        if (gt): plt.plot(gt.trajectory[:,0], gt.trajectory[:,1], label=gt.name, ls='--')
        for traj in trajs:
            plt.plot(traj.trajectory[:,0], traj.trajectory[:,1], label=traj.name)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    if (option == 'xz'):
        if (gt): plt.plot(gt.trajectory[:,0], gt.trajectory[:,2], label=gt.name, ls='--')
        for traj in trajs:
            plt.plot(traj.trajectory[:,0], traj.trajectory[:,2], label=traj.name)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.legend()
    
def plot3D(gt, trajs):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    if(gt): ax.scatter(gt.trajectory[:,0], gt.trajectory[:,1], gt.trajectory[:,2], label=gt.name)
    for traj in trajs:
        ax.scatter(traj.trajectory[:,0], traj.trajectory[:,1], traj.trajectory[:,2], label=traj.name)
    ax.legend()
    ax.set_zlim3d(-40, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')