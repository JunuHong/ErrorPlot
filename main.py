import matplotlib.pyplot as plt
import argparse
import os
import sys

import src.trajectory as tj
import src.error as error

def plot():
    return

def main(args):
    # parser = argparse.ArgumentParser(description='plot trajectory and various errors to be benchmarked.')
    # parser.add_argument('file', nargs=1, help='input file path')
    # parser.add_argument('-plot', nargs=1, help='what to plot', default='60', type=int)
    # args = parser.parse_args()
    
    # print args.file[0]
    data = ['./data/07/07.bag', './data/07/lio_sam_path.bag', './data/07/aloam_path.bag', './data/07/lego_loam_path.bag']
    
    tj_list = []
    for file in data:    
        tj_list.append(tj.Trajectory(file))
    
    for traj in tj_list:
        print traj.name

if __name__ == '__main__':
    main(sys.argv[1:])