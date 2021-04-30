import numpy as np
import matplotlib.pyplot as plt

from trajectory import Trajectory

class Error():
    def __init__(self, *traj):
        self.n_traj = len(traj)
        gt_exsist, gt_index = self._is_there_gt(traj)
        
        if(gt_exsist):
            self.ape_trans, self.ape_rot = self.ATE(traj[gt_index].pose, traj[gt_index].pose)
        self.rpe_trans, self.rpe_rot = self.RTE()
        
    def _is_there_gt(self, traj):
        for i in range(self.n_traj):
            if (traj[i].is_gt): return True, i
        return False, None
                
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
        
    def ATE(self, GT, TEST):
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

    def RPE(self, GT, TEST):
        rpe_trans = []
        rpe_rot = []
        delta = 1
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
    plt.xlabel('index')
    plt.ylabel('ape[m]')

    plt.subplot(2,1,2)
    for i in range(n_files):
        plt.plot(errors[i].ape_rot, label='APE_rotation')
    plt.xlabel('index')
    plt.ylabel('ape[rad]')
    
def show():
    plt.show()