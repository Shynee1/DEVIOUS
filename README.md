# DEVIOUS: Deep Event-based Visual Inertial Odometry Using Synchronization

DEVIOUS is a novel Visual-Inertial Odometry (VIO) framework designed to leverage the unique advantages of event cameras in conjunction with inertial measurements. Unlike traditional frame-based VIO pipelines, DEVIOUS operates on dense optical flow fields derived from asynchronous event streams, providing high-speed, low-latency, and robust odometry estimation even in challenging environments.

## Installation and Dataset Setup

**Environment Installation**

See `requirements.txt` for environment requirements.

**Download Datasets**

DEVIOUS was benchmarked using the [Multi-robot, Multi-Sensor, Multi-Environment Event Dataset (M3ED)](https://m3ed.io/). Sequences were passed through E-RAFT to generate dense optical flow and Air-IO to predict inertial odometry.

Download the converted flow files, the ground truth files, and the AirIO/AirIMU predictions here.

**Download Pre-trained Model & Results**

The first component of DEVIOUS is the Visual Odometry (VO) model. You can download our pre-trained VO models and results below:
|   Training Dataset | Model | Results |
| :--------: | :----------: | :--------:|
| M3ED | VO Model | VO Results |
| KITTI | VO Model | VO Results |

The models must be moved into the `checkpoints` folder before inference. 

## Run with Default Configuration

You can immediately test our method on the M3ED dataset using pre-defined settings. 

**DEVIOUS EKF**

To run the EKF component of DEVIOUS, download the ground truth files, Air-IO predictions, and DEVIOUS-VO predictions from the links above. Then, adjust the `m3ed_ekf.json` configuration file to add the correct paths for the files. 

Then, run the following command: `python main.py ekf -d <dataset>`

**DEVIOUS VO**

If you prefer to run the VO model yourself, download the E-RAFT flow files from the link above. 

Then, run the following command: `python main.py model encoding cache -d m3ed`
Once that is finished, run `python main.py model recurrent test -d m3ed`

