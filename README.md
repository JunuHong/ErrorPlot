# ErrorPlot
ErrorPloting module for SLAM algrithms

This module is for plotting recorded data from `PathRecorder`  
Basic trajectory plot and error(APE, RPE) plot available

## Command Line Interface
**Arguments**
* `-f/--file` - indivisual input file relative path
* `-F/--folder` - input file folder relative path

* `-p/--plot` - plotting option  

**Available plotting option**
* `all` - plot all trajectories and errors and error statistics **default**
* `traj` - plot trajectories
* `error` - plot errors
* `stat` - plot error statistics

## Example
using `-f/--file` to plot indivisual or selected files.
```bash
$ python main.py -f ./data/07/07.bag ./data/07/lio_sam_path.bag
```
using `-F/--folder` to plot every file inside folder
```bash
$ python main.py -p ./data/07
```
