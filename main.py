import matplotlib.pyplot as plt
import argparse
import os
import sys

import src.trajectory as tj
import src.error as error

def plot():
    return

def main(args):
    #TODO : argument parser for user input/ think what and how to get settings input/
    # parser = argparse.ArgumentParser(description='plot trajectory and various errors to be benchmarked.')
    # parser.add_argument('file', nargs=1, help='input file path')
    # parser.add_argument('-plot', nargs=1, help='what to plot', default='60', type=int)
    # args = parser.parse_args()
    
    # print args.file[0]
    data = ['./data/07/07.bag', './data/07/aloam_path.bag', 
            './data/07/lego_loam_path.bag', './data/07/lio_sam_path.bag']
    
    # tj_list = []
    # for file in data:    
    #     tj_list.append(tj.Trajectory(file))
     
    # error_list = []
    # for i in xrange(len(tj_list)):
    #     if(tj_list[i].is_gt):
    #         gt = tj_list.pop(i)
    # print tj_list
    gt = tj.Trajectory(data[0])
    aloam = tj.Trajectory(data[1])
    lego_loam = tj.Trajectory(data[2])
    lio_sam = tj.Trajectory(data[3])
    
    # tj.plotXYZ(gt, aloam, lego_loam, lio_sam)
    # tj.plot2D('xy', gt, aloam, lego_loam, lio_sam)
    # tj.plot3D(gt, aloam, lego_loam, lio_sam)
    
    error_aloam = error.Error(gt, aloam)
    error_lego = error.Error(gt, lego_loam)
    error_lio = error.Error(gt, lio_sam)
    
    error.plotAPE(error_aloam, error_lego, error_lio)
    error.plotRPE(error_aloam, error_lego, error_lio)
    plt.show()
        
if __name__ == '__main__':
    main(sys.argv[1:])