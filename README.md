# DEVIOUS: Deep Event-based Visual Inertial Odometry Using Synchronization

DEVIOUS is a novel Visual-Inertial Odometry (VIO) framework designed to leverage the unique advantages of event cameras in conjunction with inertial measurements. Unlike traditional frame-based VIO pipelines, DEVIOUS operates on dense optical flow fields derived from asynchronous event streams, providing high-speed, low-latency, and robust odometry estimation even in challenging environments.

![model architecture](model_architecture.png)

## Overview

DEVIOUS consists of two components:

### DEVIOUS VO (Visual Odometry)
A deep learning model that processes dense optical flow from event cameras to predict visual odometry.

- **Input**: E-RAFT optical flow fields (from event camera data)
- **Output**: Visual odometry predictions (pose estimates from vision alone)

### DEVIOUS EKF (Extended Kalman Filter)
A sensor fusion module that combines visual and inertial odometry sources using an Extended Kalman Filter to produce the final VIO estimate.

- **Input**: DEVIOUS-VO predictions + Air-IO inertial odometry predictions
- **Output**: Fused VIO estimate combining visual and inertial information

## Installation and Dataset Setup

**Environment Installation**

See `requirements.txt` for environment requirements and install dependencies:
```bash
pip install -r requirements.txt
```

**Download Datasets**

DEVIOUS was benchmarked using the [Multi-robot, Multi-Sensor, Multi-Environment Event Dataset (M3ED)](https://m3ed.io/).

To run DEVIOUS, on an M3ED sequence, download the following files:
1. The `data` and `depth` h5 files from the [M3ED website](https://m3ed.io/download/).
2. The pre-processed Air-IO/Air-IMU predictions from [Google Drive](https://drive.google.com/drive/folders/111kW5DH8ZRleuIpjKy7HMHVOY5X-1DlQ?usp=sharing).
3. The pre-trained DEVIOUS-VO model & results from [Google Drive](https://drive.google.com/drive/folders/1LMsrI-2LrBANhFeLj9GDwt21ZOlP6edw?usp=sharing)

The models must be moved into the `checkpoints/` folder before inference. 

> [!NOTE]
> If you plan to inference DEVIOUS-VO manually, you will need to run E-RAFT on the event data to generate dense optical flow. 
> Our implementation for this can be found [here](https://github.com/Shynee1/)

## Quick Run using Pre-trained VO Model

You can immediately test our method on the M3ED dataset using the pre-trained VO model.

**Prerequisites:** M3ED ground truth and data files, Air-IO predictions, downloaded DEVIOUS-VO results

1. Move **all** M3ED data and ground truth files into one folder.
2. Edit the `data-root` attribute of the `m3ed_ekf.json` file to reflect the absolute location of the folder with the M3ED data.
3. Move the `devious_output.pickle` file into a folder with the Air-IO and Air-IMU results.
4. Edit the `dataset_root` attribute of the `m3ed_ekf.json` file to reflect the absolute location of the folder with the pickle data.
5. Run EKF fusion:
```bash
python main.py ekf -d m3ed
```
6. Results will be saved to `saved/m3ed_ekf/`

## Full Run using DEVIOUS-VO

To test our full pipeline, you can generate VO predictions from scratch and fuse them with the EKF.

**Prerequisites:** M3ED ground truth and data files, Air-IO predictions, E-RAFT flow files, pre-trained VO model in `checkpoints/`

1. Move **all** M3ED data and ground truth files into one folder.
2. Edit the `data-root` attribute of both `m3ed_encoder.json` and `m3ed_recurrent.json` to reflect the absolute location of the folder with the M3ED data.
3. Encode the flows and cache their results:
```bash
python main.py model encoder cache -d m3ed
```
4. Run VO inference:
```bash
python main.py model recurrent test -d m3ed
```
5. Results will be saved to `saved/m3ed_recurrent/`
6. Move all .pickle files into one folder
7. Edit the `dataset_root` attribute of the `m3ed_ekf.json` file to reflect the absolute location of the folder with the pickle data.
8. Run EKF fusion:
```bash
python main.py ekf -d m3ed
```
9. Results will be saved to `saved/m3ed_ekf/`

## Adding New Datasets

To add a new dataset, follow the steps below:
1. Create a custom data loader similar to `loaders/m3ed_loader.py`
2. Create custom config files similar to those in `configs/`
3. Adjust `main.py` to add your dataset as a valid command
4. Run the steps above for training/testing

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

Research was supervised by Prof. Kostas Daniilidis, Matthew Leonard, and Ioannis Asmanis. 


