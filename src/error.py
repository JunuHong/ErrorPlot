import numpy as np
import matplotlib.pyplot as plt
import copy

from src.trajectory import Trajectory
import src.quaternion as quat

class Error:
    def __init__(self, reference=None, estimate=None, delta=1):
        """Calculate Error(APE, RPE)
        APE 

        Args:
            reference (Trajectory): reference trajectory or ground truth trajectory. Defaults to None.
            estimate  (Trajectory): estimated trajectory for evaluation. Defaults to None.
            delta  (int, optional): local accuracy of the trajectory over a fixed time interval delta(for RPE). Defaults to 1
        """
        self.name = estimate.name
        print("Calculating {}'s Error with respect to Ground Truth Data".format(self.name))

        self.reference, self.estimate = self._post_process(copy.deepcopy(reference), copy.deepcopy(estimate))
        self.time = self.estimate.time

        self.ape_trans, self.ape_rot = self.APE(self.reference, self.estimate)
        self.ape_tans_stat = self._statistics(self.ape_trans)
        self.ape_rot_stat = self._statistics(self.ape_rot)

        self.rpe_trans, self.rpe_rot = self.RPE(self.reference, self.estimate, delta)
        self.rpe_tans_stat = self._statistics(self.rpe_trans)
        self.rpe_rot_stat = self._statistics(self.rpe_rot)

    def _post_process(self, gt, test):
        orientation, trajectory, dur = [], [], []
        index = []
        for i in range(gt.length):
            time = gt.time[i]
            for j in range(test.length - 1):
                if test.time[j] < time < test.time[j+1]:
                    alpha = (time-test.time[j])/(test.time[j+1]-test.time[j])
                    orientation.append(quat.SLERP(test.orientation[j], test.orientation[j+1], alpha))
                    trajectory.append((1 - alpha) * test.trajectory[j] + alpha * test.trajectory[j + 1])
                    dur.append(time)
                    index.append(i)
        index = np.array(index)

        gt.trajectory = gt.trajectory[index]
        gt.orientation = gt.orientation[index]
        gt.time = gt.time[index]
        gt.length = gt.trajectory.shape[0]

        test.trajectory = np.array(trajectory)
        test.orientation = np.array(orientation)
        test.time = np.array(dur)
        test.length = test.trajectory.shape[0]

        return gt, test

    def _statistics(self, error):
        std = np.std(error)
        mean = np.mean(error)
        median = np.median(error)
        minimum = np.min(error)
        maximum = np.max(error)
        rmse = np.sqrt((np.asarray(error) ** 2).mean())

        return [mean, std, median, minimum, maximum, rmse]

    def APE(self, gt, test):
        target_mean = gt.trajectory.mean(0)
        estimate_mean = test.trajectory.mean(0)

        target = gt.trajectory - target_mean
        estimate = test.trajectory - estimate_mean

        W = np.dot(target.T, estimate)

        U, _, V = np.linalg.svd(W, full_matrices=True, compute_uv=True)

        # TODO check for possible bug/ when calculating lio_sam
        R = np.dot(U, V)
        t = target_mean - np.dot(R, estimate_mean)
        T = np.vstack([np.hstack([R, t.reshape(3, 1)]), np.array([0, 0, 0, 1])])

        ape_trans, ape_rot = [], []

        for i in range(gt.length):
            Q = np.vstack([np.hstack([gt.orientation[i].rotation(), gt.trajectory[i].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            P = np.vstack([np.hstack([test.orientation[i].rotation(), test.trajectory[i].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            E = np.dot(np.linalg.inv(Q), np.dot(T, P))

            ape_trans.append(np.linalg.norm(E[:3, 3]))
            ape_rot.append(np.arccos((np.trace(E[:3, :3]) - 1) / 2))
        return ape_trans, ape_rot

    def RPE(self, gt, test, delta):
        rpe_trans = []
        rpe_rot = []
        for i in range(gt.length - delta):
            Q = np.vstack([np.hstack([gt.orientation[i].rotation(), gt.trajectory[i].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            Q_delta = np.vstack([np.hstack([gt.orientation[i + delta].rotation(), gt.trajectory[i + delta].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            Q = np.dot(np.linalg.inv(Q), Q_delta)
            P = np.vstack([np.hstack([test.orientation[i].rotation(), test.trajectory[i].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            P_delta = np.vstack([np.hstack([test.orientation[i + delta].rotation(), test.trajectory[i + delta].reshape(3, 1)]), np.array([0, 0, 0, 1])])
            P = np.dot(np.linalg.inv(P), P_delta)

            E = np.dot(np.linalg.inv(Q), P)

            rpe_trans.append(np.linalg.norm(E[:3, 3]))
            rpe_rot.append(np.arccos((np.trace(E[:3, :3]) - 1) / 2))
        return rpe_trans, rpe_rot


def plotAPE(errors):
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    for error in errors:
        plt.plot(error.time, error.ape_trans, label=error.name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[m]')

    plt.subplot(2, 1, 2)
    for error in errors:
        plt.plot(error.time, error.ape_rot, label=error.name)
        # for key, value in errors[i].ape_tans_stat.items():
        #     plt.axhline(y=value, color='r', linestyle='-', label=key)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('ape[rad]')


def plotRPE(errors):
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    for error in errors:
        plt.plot(error.time[1:], error.rpe_trans, label=error.name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[m]')

    plt.subplot(2, 1, 2)
    for error in errors:
        plt.plot(error.time[1:], error.rpe_rot, label=error.name)
    plt.legend()
    plt.xlabel('time[nano_sec]')
    plt.ylabel('rpe[rad]')


def plotAPEStats(errors):
    import pandas as pd
    index = ['mean', 'std', 'median', 'minimum', 'maximum', 'rmse']
    trans_dic = {}
    rot_dic = {}
    for error in errors:
        trans_dic[error.name] = error.ape_tans_stat
        rot_dic[error.name] = error.ape_rot_stat
    trans_data = pd.DataFrame(trans_dic, index=index)
    rot_data = pd.DataFrame(rot_dic, index=index)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(2, 1, 1)
    ax.title.set_text('APE Translation')
    trans_data.plot.bar(ax=ax)
    ax = fig.add_subplot(2, 1, 2)
    ax.title.set_text('APE Rotation')
    rot_data.plot.bar(ax=ax)


def plotRPEStats(errors):
    import pandas as pd
    index = ['mean', 'std', 'median', 'minimum', 'maximum', 'rmse']
    trans_dic = {}
    rot_dic = {}
    for error in errors:
        trans_dic[error.name] = error.rpe_tans_stat
        rot_dic[error.name] = error.rpe_rot_stat
    trans_data = pd.DataFrame(trans_dic, index=index)
    rot_data = pd.DataFrame(rot_dic, index=index)

    # TODO add value to bar
    fig = plt.figure(figsize=(6, 6))

    ax = fig.add_subplot(2, 1, 1)
    ax.title.set_text('RPE Translation')
    trans_data.plot.barh(ax=ax)

    ax = fig.add_subplot(2, 1, 2)
    ax.title.set_text('RPE Rotation')
    rot_data.plot.barh(ax=ax)
