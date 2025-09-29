# DEVIOUS: Deep Event-based Visual Inertial Odometry Using Synchronization

DEVIOUS is a novel Visual-Inertial Odometry (VIO) framework designed to leverage the unique advantages of event cameras in conjunction with inertial measurements. Unlike traditional frame-based VIO pipelines, DEVIOUS operates on dense optical flow fields derived from asynchronous event streams, providing high-speed, low-latency, and robust odometry estimation even in challenging environments.

## Installation and Dataset Setup

**Environment Installation**

See `requirements.txt` for environment requirements.

**Download Datasets**

DEVIOUS was benchmarked using the [Multi-robot, Multi-Sensor, Multi-Environment Event Dataset (M3ED)](https://m3ed.io/). Sequences were passed through E-RAFT to generate dense optical flow and Air-IO to predict inertial odometry.

Download the converted flow files and the AirIO/AirIMU predictions here.

You will also need to download the `data` and `depth` h5 files from the M3ED website.

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

To run the EKF and save the results, run this command:
```
python main.py ekf -d <dataset>
```

All outputs will be saved to `saved/<dataset>_ekf`

**DEVIOUS VO**

If you prefer to run the VO model yourself, download the E-RAFT flow files from the link above. 

To encode all flow values and cache the results, run this command:
```
python main.py model encoding cache -d <dataset>
```

To run inference on the cached flows, run this command:
```
python main.py model recurrent test -d <dataset>
```

All outputs will be saved to `saved/<dataset>_recurrent`

## Citations

Portions of the DEVIOUS codebase were adapted from E-RAFT:
```
@InProceedings{Gehrig3dv2021,
  author = {Mathias Gehrig and Mario Millh\"ausler and Daniel Gehrig and Davide Scaramuzza},
  title = {E-RAFT: Dense Optical Flow from Event Cameras},
  booktitle = {International Conference on 3D Vision (3DV)},
  year = {2021}
}
```
and AirIO:
```
@misc{qiu2025airiolearninginertialodometry,
      title={AirIO: Learning Inertial Odometry with Enhanced IMU Feature Observability}, 
      author={Yuheng Qiu and Can Xu and Yutian Chen and Shibo Zhao and Junyi Geng and Sebastian Scherer},
      year={2025},
      eprint={2501.15659},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2501.15659}, 
}
```

## Credits

This research was completed by Jack Ford and Joseph Kahana through the University of Pennsylvania's GRASP Laboratory. 

We were supervised by Prof. Kostas Daniilidis, Matthew Leonard, and Ioannis Asmanis. 


