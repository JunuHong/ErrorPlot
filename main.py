import src.trajectory as tj
import src.error as error
import matplotlib.pyplot as plt

def main():
    GT = './data/KITTI_00_gt.txt'
    ORB = './data/KITTI_00_ORB.txt'
    SPTAM = './data/KITTI_00_SPTAM.txt'

    GT = tj.Trajectory(GT)
    ORB = tj.Trajectory(ORB)
    SPTAM = tj.Trajectory(SPTAM)
    
    ORB_error = error.Error(GT, ORB)
    SPTAM_error = error.Error(GT, SPTAM)
    
    tj.plot2D(GT, ORB, SPTAM)
    tj.plotXYZ(GT, ORB, SPTAM)
    
    error.plotAPE(ORB_error, SPTAM_error)    
    
    tj.show()

    return

if __name__ == '__main__':
    main()