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

def plot_traj(gt, tj_list):
    tj.plotXYZ(gt, tj_list)
    tj.plot2D('xy', gt, tj_list)
    tj.plot3D(gt, tj_list)
    return plt.show()

def plot_error(plot_arg, gt, tj_list, error_list):
    if (plot_arg == 'all'):
        tj.plotXYZ(gt, tj_list)
        tj.plot2D('xy', gt, tj_list)
        tj.plot3D(gt, tj_list)
    
        error.plotAPE(error_list)
        error.plotAPEStats(error_list)
        error.plotRPE(error_list)
        error.plotRPEStats(error_list)
        
    if (plot_arg == 'error'):    
        error.plotAPE(error_list)
        error.plotAPEStats(error_list)
        error.plotRPE(error_list)
        error.plotRPEStats(error_list)
        
    if (plot_arg == 'stat'):
        tj.plotXYZ(gt, tj_list)
        tj.plot2D('xy', gt, tj_list)
        tj.plot3D(gt, tj_list)
    
        error.plotAPE(error_list)
        error.plotAPEStats(error_list)
        error.plotRPE(error_list)
        error.plotRPEStats(error_list)
    return plt.show()

def plot(plot_arg, file_list):
    gt, tj_list = traj_process(file_list)
    if plot_arg == 'traj' : return plot_traj(gt, tj_list)
    else:
        error_list = error_process(gt, tj_list)
        return plot_error(plot_arg, gt, tj_list, error_list)
    
def main(arg):
    #TODO : argument parser for user input/ think what and how to get settings input/ ==> what to plot and how to get the input.
    
    parser = argparse.ArgumentParser(description='plot trajectory and various errors to be benchmarked.')
    parser.add_argument('-p','--plot', choices=['all', 'traj', 'error', 'stat'], default='all', help='plot chart')
    parser.add_argument('--plot_mode', choices=['xy', 'xz', '3D', 'xyz'], default='all', help='plot chart')
    parser.add_argument('-F', '--folder', help='input file directory')
    parser.add_argument('-f', '--file', action='append', nargs='+',  help='input files')
    args = parser.parse_args()
    if len(arg) == 0:
        parser.print_help()
        sys.exit(0)

    if args.folder:
        file_list = []
        workDIr = os.path.abspath(args.folder)
        for dirpath, _, filenames in os.walk(workDIr):
            for filename in filenames:
                file_list.append(dirpath+'/'+filename)
    if args.file:
        file_list = args.file[0]
    
    plot_arg = args.plot
    
    plot(plot_arg, file_list)
    
if __name__ == '__main__':
    main(sys.argv[1:])