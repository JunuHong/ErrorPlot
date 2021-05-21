import numpy as np
import matplotlib.pyplot as plt
import copy
from rospy.rostime import genpy

from src.trajectory import Trajectory

class Error():
    def __init__(self, reference=None, estimate=None, delta = 1):
        """Calculate Error(APE, RPE)
        APE 

        Args:
            reference (Trajectory): reference trajectory or ground truth trajectory. Defaults to None.
            estimate  (Trajectory): estimated trajectory for evaluation. Defaults to None.
            delta  (int, optional): local accuracy of the trajectory over a fixed time interval delta(for RPE). Defaults to 1
        """
        self.name = estimate.name
        self.is_short = False
        
        self.reference, self.estimate = self._post_process(copy.deepcopy(reference), copy.deepcopy(estimate))
        self.time = [time.to_nsec() for time in self.estimate.time]
        
        self.ape_trans, self.ape_rot = self.APE(self.reference, self.estimate)
        self.ape_tans_stat = self._statistics(self.ape_trans)
        self.ape_rot_stat = self._statistics(self.ape_rot)
        
        self.rpe_trans, self.rpe_rot = self.RPE(self.reference, self.estimate, delta)
        self.rpe_tans_stat = self._statistics(self.rpe_trans)
        self.rpe_rot_stat = self._statistics(self.rpe_rot)
    
    def _post_process(self, GT, TEST): #TODO
        if (GT.length == TEST.length): return GT, TEST
        m = int(np.around(float(GT.length)/float(TEST.length)))
        if (m > 1):
            self.is_short = True
        
        index = []
        for i in range(GT.length):
            for j in range(TEST.length):
                if ((GT.time[i]-TEST.time[j])>genpy.Duration(-0.01) and (GT.time[i]-TEST.time[j])<genpy.Duration(0.01)):
                    index.append([i,j])
                    break
        index = np.array(index)
        
        GT.trajectory = GT.trajectory[index[:,0]]
        GT.pose = GT.pose[index[:,0]]
        GT.time = GT.time[index[:,0]] 
        GT.length = GT.trajectory.shape[0]
        
        TEST.trajectory = TEST.trajectory[index[:,1]]
        TEST.pose = TEST.pose[index[:,1]]
        TEST.time = TEST.time[index[:,1]]
        TEST.length = TEST.trajectory.shape[0]
        
        return GT, TEST
    
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
        plt.plot(errors[i].time, errors[i].ape_trans, label=errors[i].name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[m]')

    plt.subplot(2,1,2)
    for i in range(n_files):
        plt.plot(errors[i].time, errors[i].ape_rot, label=errors[i].name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[rad]')
    
def plotRPE(*errors):
    n_files = len(errors)
    
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for i in range(n_files):
        plt.plot(errors[i].time[1:], errors[i].rpe_trans, label=errors[i].name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[m]')

    plt.subplot(2,1,2)
    for i in range(n_files):
        plt.plot(errors[i].time[1:], errors[i].rpe_rot, label=errors[i].name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[rad]')