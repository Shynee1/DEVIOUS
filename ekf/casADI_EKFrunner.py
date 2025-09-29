import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import os
import argparse
import numpy as np
import pypose as pp
import tqdm

import torch
import torch.utils.data as Data

from scipy.spatial.transform import Rotation
from utils.helper_functions import interp_xyz

from .casADI_ekf import CasADIEKF
from matplotlib import pyplot as plt
from .ekfutil import plot_bias_subplots, visualize_velocity
import yaml
import numpy as np

class EKF_runner():
    def __init__(self,
                q=torch.ones(12, dtype=torch.float64) * 0.01,
                r=torch.ones(3,  dtype=torch.float64) * 0.01,
                p=torch.zeros(15, dtype=torch.float64)):
        self.state = torch.zeros(15, dtype=torch.float64)      # x₀
        self.P     = torch.eye(15, dtype=torch.float64) * p**2 # P₀
        self.P_list = [self.P]

        self.current_idx = 0
        self.state_lengeth = 9
        self.window = 1

        self.r = r
        self.q = q
        self.p = p

        self.est_history = []
        self.gravity = np.array([0., 0., 9.81007])
        self.data = None

        self.filter = CasADIEKF()
    
    def get_result(self):
        return torch.stack(self.est_history), torch.stack(self.P_list)

    def propogate_update(self, imu_data, observation, Q=None, R=None):
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1).cpu().numpy()
        dt_np = float(imu_data['dt'])
        z_np = observation.cpu().numpy()
        Q_np = Q.cpu().numpy()
        R_np = R.cpu().numpy()
        x_np, P_np = self.filter.filtering(self.state[...,None].cpu().numpy(),input[...,None],dt_np, z_np, 
                                        self.P.cpu().numpy(), Q_np,R_np)
        self.state = torch.from_numpy(np.array(x_np, dtype=np.float64)).squeeze()
        self.P     = torch.from_numpy(np.array(P_np, dtype=np.float64)).squeeze()
    def propogate_state(self, imu_data, Q=None):
        d_bias_gyro = self.state[9:12]
        d_bias_acc = self.state[12:15]
        input = torch.cat([imu_data["gyro"], imu_data["acc"], d_bias_gyro, d_bias_acc], dim=-1).cpu().numpy()
        dt_np = float(imu_data['dt'])
        Q_np = Q.cpu().numpy()

        x_np, P_np = self.filter.predict(self.state[...,None].cpu().numpy(), input[...,None], dt_np, 
                                        self.P.cpu().numpy(), Q_np)
        self.state = torch.from_numpy(np.array(x_np, dtype=np.float64)).squeeze()
        self.P     = torch.from_numpy(np.array(P_np, dtype=np.float64)).squeeze()

    def run(self, imu_data, observation = None, Q=None, R=None):
        if observation is not None:
            self.propogate_update(imu_data, observation, Q=Q, R=R)
        else:
            self.propogate_state(imu_data, Q=Q)
            
        self.est_history.append(self.state.clone())
        self.P_list.append(self.P.clone())
        self.current_idx+=1

def run_ekf(egomotion_outputs, airimu_dataset, data_name, inference_state_load, save_path):
    init = airimu_dataset.get_init_value()

    egomotion_timestamps = egomotion_outputs[:, 0]
    egomotion_translations = egomotion_outputs[:, 1:4]
    egomotion_rotations = egomotion_outputs[:, 4:7]
    egomotion_covariances = egomotion_outputs[:, 7:]
    egomotion_timestamp_prev = 0

    # dict_keys(['cov', 'net_vel', 'ts'])
    io_result = inference_state_load[data_name]

    ekf = EKF_runner()
    bias_weight = 1e-12
    imu_cov_scale = 100
    io_cov_scale = 100
    ego_cov_scale = 0.1

    # STEP 1 state initialization
    initial_state = torch.zeros(15, dtype=torch.float64)
    initial_state[:3] = init["rot"].Log()
    initial_state[3:6] = init["vel"]
    initial_state[6:9] = init["pos"]
    ekf.state = initial_state
    
    # STEP 2 covariance initialization
    # the uncertainty of the initial state
    io_index = 0
    egomotion_idx = 0
    gt_state = {"pos": [], "vel": [], "rot": []}

    t_range = tqdm.tqdm(airimu_dataset)
    for idx, data in enumerate(t_range):
        # add the measurement of the airimu
        imu_data = {"gyro": data["gyro"][0], "acc": data["acc"][0], "dt": data["dt"][0]}
        io_stamp = io_result["ts"][io_index]
        
        # STEP 2 add the learned covariance
        r = torch.ones(3, dtype=torch.float64)
        q = torch.ones(12, dtype=torch.float64) * bias_weight
        q[:3] = data["gyro_cov"][0]
        q[3:6] = data["acc_cov"][0] * imu_cov_scale

        observation = None

        # Check if airio measurement is available
        if (io_stamp - data["timestamp"]).abs() < 0.001:
            observation = io_result["net_vel"][io_index]
            r[:3] = io_result["cov"][0][io_index] * io_cov_scale
            io_index += 1
        
        # Check if egomotion measurement is available
        egomotion_timestamp = egomotion_timestamps[egomotion_idx] / 1e6
        if (data["timestamp"] - egomotion_timestamp).abs() < 0.002:
            translation_v = egomotion_translations[egomotion_idx]
            rotation_v = egomotion_rotations[egomotion_idx]

            rotation_m = Rotation.from_euler('xyz', rotation_v, degrees=True).as_matrix()

            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation_m
            transform[:3, 3] = translation_v

            T_imu_imu_new = np.array([[1,  0,  0, 0],
                                      [0,  0, -1, 0],
                                      [0, -1,  0, 0],
                                      [0,  0,  0, 1]])
            T_cam_imu_old = np.array([[-0.9999502,   0.00978231, -0.00197473,  0.09332941],
                                      [ 0.00976959,  0.9999321,   0.006352,   -0.05807844],
                                      [ 0.00203673,  0.0063324,  -0.99997788, -0.01409921],
                                      [ 0.,          0.,          0.,          1.        ]])

            T_cam_imu_new = T_imu_imu_new @ T_cam_imu_old
            new_transform = T_cam_imu_new @ transform @ np.linalg.inv(T_cam_imu_new)

            dt = egomotion_timestamp - egomotion_timestamp_prev
           
            if dt != 0:
                observation = new_transform[:3, 3] # Translation
                observation = torch.tensor(observation / dt) # Velocity

                egomotion_timestamp_prev = egomotion_timestamp
                r[:3] = torch.tensor(egomotion_covariances[egomotion_idx][:3]) * ego_cov_scale

                # Deal with our silly silly mistakess
                r[0] = egomotion_covariances[egomotion_idx][0]
                r[1] = egomotion_covariances[egomotion_idx][2]
                r[2] = egomotion_covariances[egomotion_idx][1]

            egomotion_idx += 1 if egomotion_idx < len(egomotion_timestamps) - 1 else 0

        Q = torch.eye(12, dtype=torch.float64) * q
        R = torch.eye(3, dtype=torch.float64) * r
        
        ekf.run(imu_data, observation=observation, Q=Q, R=R)
        gt_state["pos"].append(data["gt_pos"][0])
        gt_state["vel"].append(data["gt_vel"][0])
        gt_state["rot"].append(data["gt_rot"][0])
           
    gtpos = torch.stack(gt_state["pos"])
    gtrot = torch.stack(gt_state["rot"])
    gtvel = torch.stack(gt_state["vel"])
    ekf_result, ekf_cov = ekf.get_result()
    ekf_result = ekf_result.numpy()
    
    if data_name == "BlackBird":
        data_name = os.path.dirname(data_name).split('/')[1]
    else:
        data_name = data_name

    os.makedirs(save_path, exist_ok=True)
    
    np.save(os.path.join(save_path, f"{data_name}_ekf_result.npy"), ekf_result)
    ekf_poses = np.zeros((ekf_result.shape[0], 8))
    ekf_poses[:, 1:4] = ekf_result[:, 6:9]
    ekf_poses[:, 4:] = pp.so3(ekf_result[:, :3]).Exp().numpy()
    np.save(os.path.join(save_path, f"{data_name}_ekf_poses.npy"), ekf_poses)
    
    plt.figure()
    plt.plot(ekf_result[:, 6], ekf_result[:, 7], label="EKF")
    plt.plot(gtpos[:, 0], gtpos[:, 1], label="GT")
    
    plt.savefig(os.path.join(save_path, f"{data_name}_ekf_result.png"))
    
    # visualize the net velocity
    io_ts = io_result["ts"][:,0]
    net_vel = io_result["net_vel"]

    interp_net_vel = interp_xyz(airimu_dataset.data["time"], io_ts, net_vel)
    interp_net_vel = airimu_dataset.data["gt_orientation"] @ interp_net_vel
    interp_net_vel = interp_net_vel[:len(ekf_result)]

    plot_bias_subplots(ekf_result[:, 9:12], title="EKF Bias", save_path=os.path.join(save_path, f"{data_name}_bias.png"))
    visualize_velocity(f"EKF_vel_{data_name}", gtvel, ekf_result[:, 3:6], interp_net_vel, save_folder=save_path)