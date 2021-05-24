import matplotlib.pyplot as plt
import argparse
import os
import sys

import src.trajectory as tj
import src.error as error

def traj_process(data_files):
    tj_list = []
    gt = None
    for file in data_files:
        trajectory = tj.Trajectory(file)
        if(not trajectory.is_gt):
            tj_list.append(trajectory)
        else: gt = trajectory
    return gt, tj_list

def error_process(gt, tj_list):
    error_list = []
    if(gt == None):
        print('Need ground truth for error calculation.')
        return
    for tj in tj_list:
        error_list.append(error.Error(gt, tj))
    return error_list

def plot(plot_arg, gt, tj_list, error_list):
    if (plot_arg == 'all'):
        tj.plotXYZ(gt, tj_list)
        tj.plot2D('xy', gt, tj_list)
        tj.plot3D(gt, tj_list)
    
        error.plotAPE(error_list)
        error.plotRPE(error_list)
    return plt.show()

def main(args):
    #TODO : argument parser for user input/ think what and how to get settings input/
    # parser = argparse.ArgumentParser(description='plot trajectory and various errors to be benchmarked.')
    # parser.add_argument('-plot', help='plot every available chart')
    # parser.add_argument('-file',required=True, help='input file path')
    # args = parser.parse_args()
    
    # print (args.file)
    data = ['./data/07/07.bag', './data/07/aloam_path.bag', './data/07/lego_loam_path.bag', './data/07/lio_sam_path.bag']
    plot_arg = 'all'
    gt, tj_list = traj_process(data)
    error_list = error_process(gt, tj_list)
    
    plot(plot_arg, gt, tj_list, error_list)
        
if __name__ == '__main__':
    main(sys.argv[1:])