from src.trajectory import *
import matplotlib.pyplot as plt

def main():
    GT = './data/KITTI_00_gt.txt'
    ORB = './data/KITTI_00_ORB.txt'
    SPTAM = './data/KITTI_00_SPTAM.txt'

    GT = Trajectory(GT)
    ORB = Trajectory(ORB)
    SPTAM = Trajectory(SPTAM)
    
    plot2D(GT, ORB, SPTAM)
    plotXYZ(GT, ORB, SPTAM)
    plot3D(GT, ORB, SPTAM)
    show()

    return

if __name__ == '__main__':
    main()