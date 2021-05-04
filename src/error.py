import numpy as np
import matplotlib.pyplot as plt

from trajectory import Trajectory

class Error():
    def __init__(self, reference=None, estimate=None, delta = 1):
        """Calculate Error(APE, RPE)
        APE 

        Args:
            reference (Trajectory): reference trajectory or ground truth trajectory. Defaults to None.
            estimate  (Trajectory): estimated trajectory for evaluation. Defaults to None.
            delta  (int, optional): local accuracy of the trajectory over a fixed time interval delta(for RPE). Defaults to 1
        """
        self.ape_trans, self.ape_rot = self.APE(reference, estimate)
        self.ape_tans_stat = self._statistics(self.ape_trans)
        self.ape_rot_stat = self._statistics(self.ape_rot)
        
        self.rpe_trans, self.rpe_rot = self.RPE(reference, estimate, delta)
        self.rpe_tans_stat = self._statistics(self.rpe_trans)
        self.rpe_rot_stat = self._statistics(self.rpe_rot)
                
    def _rigid_transformation(self, GT, TEST):
        target_mean = GT.mean(0)
        estimate_mean = TEST.mean(0)
        
        target =  GT - target_mean
        estimate =  TEST - estimate_mean

        W = np.dot(target.T, estimate)

        U,_,V = np.linalg.svd(W,full_matrices=True,compute_uv=True)
        
        R = np.dot(U, V)
        t = target_mean - np.dot(R, estimate_mean)
        T = np.vstack([np.hstack([R, t.reshape(3,1)]), np.array([0,0,0,1])])
        return T
    
    def _statistics(self, error):
        std = np.std(error)
        mean = np.mean(error)
        median = np.median(error)
        minimum = np.min(error)
        maximum = np.max(error)
        rmse = np.sqrt((np.asarray(error)**2).mean())
        
        return {"mean"   : mean, 
                "std"    : std, 
                "median" : median, 
                "min"    : minimum, 
                "max"    : maximum, 
                "rmse"   : rmse}
        
        
    def APE(self, GT, TEST):
        target_mean = GT.trajectory.mean(0)
        estimate_mean = TEST.trajectory.mean(0)
        
        target =  GT.trajectory - target_mean
        estimate =  TEST.trajectory - estimate_mean

        W = np.dot(target.T, estimate)

        U,_,V = np.linalg.svd(W,full_matrices=True,compute_uv=True)
        
        R = np.dot(U, V)
        t = target_mean - np.dot(R, estimate_mean)
        T = np.vstack([np.hstack([R, t.reshape(3,1)]), np.array([0,0,0,1])])
        
        ape_trans = []
        ape_rot = []
        for i in range(GT.pose.shape[0]):
            Q = np.vstack([GT.pose[i], np.array([0,0,0,1])])
            P = np.vstack([TEST.pose[i], np.array([0,0,0,1])])
            E = np.dot(np.linalg.inv(Q),np.dot(T,P))
            
            ape_trans.append(np.linalg.norm(E[:3,3]))
            ape_rot.append(np.arccos((np.trace(E[:3,:3])-1)/2))
        return ape_trans, ape_rot

    def RPE(self, GT, TEST, delta):
        rpe_trans = []
        rpe_rot = []
        for i in range(GT.pose.shape[0]-delta):
            Q = np.vstack([GT.pose[i], np.array([0,0,0,1])])
            Q_delta = np.vstack([GT.pose[i+delta], np.array([0,0,0,1])])
            Q = np.dot(np.linalg.inv(Q), Q_delta)
            P = np.vstack([TEST.pose[i], np.array([0,0,0,1])])
            P_delta = np.vstack([TEST.pose[i+delta], np.array([0,0,0,1])])
            P = np.dot(np.linalg.inv(P), P_delta)
            
            E = np.dot(np.linalg.inv(Q),P)
            
            rpe_trans.append(np.linalg.norm(E[:3,3]))
            rpe_rot.append(np.arccos((np.trace(E[:3,:3])-1)/2))
        return rpe_trans, rpe_rot
    
def plotAPE(*errors):
    n_files = len(errors)
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for i in range(n_files):
        plt.plot(errors[i].ape_trans, label='APE_translation')
        for key, value in errors[i].ape_tans_stat.items():
            plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.xlabel('index')
    plt.ylabel('ape[m]')

    plt.subplot(2,1,2)
    for i in range(n_files):
        plt.plot(errors[i].ape_rot, label='APE_rotation')
        for key, value in errors[i].ape_tans_stat.items():
            plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.xlabel('index')
    plt.ylabel('ape[rad]')
    
def plotRPE(*errors):
    n_files = len(errors)
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for i in range(n_files):
        plt.plot(errors[i].rpe_trans, label='APE_translation')
    plt.xlabel('index')
    plt.ylabel('ape[m]')

    plt.subplot(2,1,2)
    for i in range(n_files):
        plt.plot(errors[i].rpe_rot, label='APE_rotation')
    plt.xlabel('index')
    plt.ylabel('ape[rad]')
    
def show():
    plt.show()