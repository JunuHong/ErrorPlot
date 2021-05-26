from operator import index
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
        print("Calculating {}'s Error with respect to Ground Truth Data".format(self.name))
        self.is_short = False
        
        self.reference, self.estimate = self._post_process(copy.deepcopy(reference), copy.deepcopy(estimate))
        self.time = self.estimate.time
        
        self.ape_trans, self.ape_rot = self.APE(self.reference, self.estimate)
        self.ape_tans_stat = self._statistics(self.ape_trans)
        self.ape_rot_stat = self._statistics(self.ape_rot)
        
        self.rpe_trans, self.rpe_rot = self.RPE(self.reference, self.estimate, delta)
        self.rpe_tans_stat = self._statistics(self.rpe_trans)
        self.rpe_rot_stat = self._statistics(self.rpe_rot)
    
    def _post_process(self, GT, TEST): 
        if (GT.length == TEST.length): return GT, TEST
        m = int(np.around(float(GT.length)/float(TEST.length)))
        if (m > 1):
            self.is_short = True
        
        index = []
        for i in range(GT.length):
            for j in range(TEST.length):
                if ((GT.time[i]-TEST.time[j])>-10000000 and (GT.time[i]-TEST.time[j])<10000000):
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
        
        return [mean,std,median,minimum,maximum,rmse]
           
    def APE(self, GT, TEST):
        target_mean = GT.trajectory.mean(0)
        estimate_mean = TEST.trajectory.mean(0)
        
        target =  GT.trajectory - target_mean
        estimate =  TEST.trajectory - estimate_mean

        W = np.dot(target.T, estimate)

        U,_,V = np.linalg.svd(W,full_matrices=True,compute_uv=True)
        
        #TODO check for posible bug/ when calculating lio_sam
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
   
def plotAPE(errors):
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for error in errors:
        plt.plot(error.time, error.ape_trans, label=error.name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[m]')

    plt.subplot(2,1,2)
    for error in errors:
        plt.plot(error.time, error.ape_rot, label=error.name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[rad]')
    
def plotRPE(errors):
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    for error in errors:
        plt.plot(error.time[1:], error.rpe_trans, label=error.name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[m]')

    plt.subplot(2,1,2)
    for error in errors:
        plt.plot(error.time[1:], error.rpe_rot, label=error.name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[rad]')
    
def plotAPEStats(errors):
    import pandas as pd
    index = ['mean','std','median','minimum','maximum','rmse']
    trans_dic = {}
    rot_dic = {}
    for error in errors:
        trans_dic[error.name] = error.ape_tans_stat
        rot_dic[error.name] = error.ape_rot_stat
    trans_data = pd.DataFrame(trans_dic, index=index)
    rot_data = pd.DataFrame(rot_dic, index=index)
    fig = plt.figure(figsize=(10,10))
    
    ax = fig.add_subplot(2,1,1)
    ax.title.set_text('APE Translation')
    trans_data.plot.bar(ax=ax)
    ax = fig.add_subplot(2,1,2)
    ax.title.set_text('APE Rotation')
    rot_data.plot.bar(ax=ax)
    
def plotRPEStats(errors):
    import pandas as pd
    index = ['mean','std','median','minimum','maximum','rmse']
    trans_dic = {}
    rot_dic = {}
    for error in errors:
        trans_dic[error.name] = error.rpe_tans_stat
        rot_dic[error.name] = error.rpe_rot_stat
    trans_data = pd.DataFrame(trans_dic, index=index)
    rot_data = pd.DataFrame(rot_dic, index=index)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(2,1,1)
    ax.title.set_text('RPE Translation')
    trans_data.plot.bar(ax=ax)
    ax = fig.add_subplot(2,1,2)
    ax.title.set_text('RPE Rotation')
    rot_data.plot.bar(ax=ax)