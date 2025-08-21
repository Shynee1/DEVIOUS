from abc import ABC

import numpy as np
import torch
import torch.utils.data as Data
import os
import h5py
import pypose as pp
import torch
from scipy.spatial.transform import Rotation
from utils.helper_functions import qinterp
import pickle

def imu_seq_collate(data):
    acc = torch.stack([d["acc"] for d in data])
    gyro = torch.stack([d["gyro"] for d in data])

    gt_pos = torch.stack([d["gt_pos"] for d in data])
    gt_rot = torch.stack([d["gt_rot"] for d in data])
    gt_vel = torch.stack([d["gt_vel"] for d in data])

    init_pos = torch.stack([d["init_pos"] for d in data])
    init_rot = torch.stack([d["init_rot"] for d in data])
    init_vel = torch.stack([d["init_vel"] for d in data])

    dt = torch.stack([d["dt"] for d in data])

    return {
        "dt": dt,
        "acc": acc,
        "gyro": gyro,
        "gt_pos": gt_pos,
        "gt_vel": gt_vel,
        "gt_rot": gt_rot,
        "init_pos": init_pos,
        "init_vel": init_vel,
        "init_rot": init_rot,
    }

class Sequence(ABC):
    # Dictionary to keep track of subclasses
    subclasses = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls


class SeqDataset(Data.Dataset):
    def __init__(
        self,
        root,
        dataname,
        device="cpu",
        name="ALTO",
        duration=200,
        step_size=200,
        mode="inference",
        drop_last=True,
        conf={},
    ):
        super().__init__()

        self.DataClass = Sequence.subclasses
        self.conf = conf
        self.seq = self.DataClass[name](root, dataname, **self.conf)
        self.data = self.seq.data
        self.seqlen = self.seq.get_length() - 1
        self.gravity = conf.gravity if "gravity" in conf.keys() else 9.81007
        self.interpolate = True

        if duration is None:
            self.duration = self.seqlen
        else:
            self.duration = duration

        if step_size is None:
            self.step_size = self.seqlen
        else:
            self.step_size = step_size

        self.data["acc_cov"] = 0.08 * torch.ones_like(self.data["acc"])
        self.data["gyro_cov"] = 0.006 * torch.ones_like(self.data["gyro"])

        start_frame = 0
        end_frame = self.seqlen

        self.index_map = [
            [i, i + self.duration]
            for i in range(0, end_frame - start_frame - self.duration, self.step_size)
        ]
        if (self.index_map[-1][-1] < end_frame) and (not drop_last):
            self.index_map.append([self.index_map[-1][-1], end_frame])

        self.index_map = np.array(self.index_map)

        loaded_param = f"loaded: {root}"
        if "calib" in self.conf:
            loaded_param += f", calib: {self.conf.calib}"
        loaded_param += f", interpolate: {self.interpolate}, gravity: {self.gravity}"
        print(loaded_param)

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }

    def get_init_value(self):
        return {
            "pos": self.data["gt_translation"][:1],
            "rot": self.data["gt_orientation"][:1],
            "vel": self.data["velocity"][:1],
        }

    def get_mask(self):
        return self.data["mask"]

    def get_gravity(self):
        return self.gravity


class SeqInfDataset(SeqDataset):
    def __init__(
        self,
        root,
        dataname,
        inference_state,
        device="cpu",
        name="ALTO",
        duration=200,
        step_size=200,
        drop_last=True,
        mode="inference",
        usecov=True,
        useraw=False,
        usetimecut=False,
        conf={},
    ):
        super().__init__(
            root, dataname, device, name, duration, step_size, mode, drop_last, conf
        )
        time_cut = 0
        if usetimecut:
            time_cut = self.seq.time_cut
        if "correction_acc" in inference_state.keys() and not useraw:
            self.data["acc"][:-1] += inference_state["correction_acc"][:, time_cut:].cpu()[0]
            self.data["gyro"][:-1] += inference_state["correction_gyro"][:, time_cut:].cpu()[0]

        if "gyro_bias" in inference_state.keys():
            print("adapted gyro bias: ", inference_state["gyro_bias"][time_cut:].cpu())
            self.data["gyro"][:-1] -= inference_state["gyro_bias"][time_cut:].cpu()
        if "acc_bias" in inference_state.keys():
            print("adapted acc bias: ", inference_state["acc_bias"][time_cut:].cpu())
            self.data["acc"][:-1] -= inference_state["acc_bias"][time_cut:].cpu()

        if "adapt_acc" in inference_state.keys():
            self.data["acc"][:-1] -= np.array(inference_state["adapt_acc"][time_cut:])
            self.data["gyro"][:-1] -= np.array(inference_state["adapt_gyro"][time_cut:])

        if "acc_cov" in inference_state.keys() and usecov:
            self.data["acc_cov"] = inference_state["acc_cov"][0][time_cut:]

        if "gyro_cov" in inference_state.keys() and usecov:
            self.data["gyro_cov"] = inference_state["gyro_cov"][0][time_cut:]

    def get_bias(self):
        return {"gyro_bias": self.data["g_b"][:-1], "acc_bias": self.data["a_b"][:-1]}
    
    
    def __getitem__(self, i):
        frame_id, end_frame_id = self.index_map[i]
        return {
            "acc_cov": self.data["acc_cov"][frame_id:end_frame_id] if "acc_cov" in self.data.keys() else None,
            "gyro_cov": self.data["gyro_cov"][frame_id:end_frame_id] if "gyro_cov" in self.data.keys() else None,
            "timestamp": self.data['time'][frame_id+1: end_frame_id+1],
            "dt": self.data["dt"][frame_id:end_frame_id],
            "acc": self.data["acc"][frame_id:end_frame_id],
            "gyro": self.data["gyro"][frame_id:end_frame_id],
            "rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "gt_pos": self.data["gt_translation"][frame_id + 1 : end_frame_id + 1],
            "gt_rot": self.data["gt_orientation"][frame_id + 1 : end_frame_id + 1],
            "gt_vel": self.data["velocity"][frame_id + 1 : end_frame_id + 1],
            "init_pos": self.data["gt_translation"][frame_id][None, ...],
            "init_rot": self.data["gt_orientation"][frame_id:end_frame_id],
            "init_vel": self.data["velocity"][frame_id][None, ...],
        }


    
class M3ED(Sequence):
    def __init__(
        self,
        data_root,
        data_name,
        coordinate=None,
        mode=None,
        rot_path=None,
        rot_type=None,
        gravity=9.81007, 
        remove_g=False,
        **kwargs
    ):
        print("Loading M3ED dataset...")
        super(M3ED, self).__init__()
        (
            self.data_root,
            self.data_name,
            self.data,
            self.ts,
            self.targets,
            self.orientations,
            self.gt_pos,
            self.gt_ori,
        ) = (data_root, data_name, dict(), None, None, None, None, None)
        
        self.g_vector = torch.tensor([0, 0, gravity], dtype=torch.double)

        gt_file_name = data_name.replace("data", "depth_gt")
        gt_file_path = os.path.join(data_root, gt_file_name)

        data_path = os.path.join(data_root, data_name)
        # load imu data
        self.load_imu(data_path)
        self.load_gt(gt_file_path)

        t_start = np.max([self.data["gt_time"][0], self.data["time"][0]])
        t_end = np.min([self.data["gt_time"][-1], self.data["time"][-1]])

        print(f"range of time values: {t_start} - {t_end}")

        # find the index of the start and end
        idx_start_imu = np.searchsorted(self.data["time"], t_start) 
        idx_start_gt = np.searchsorted(self.data["gt_time"], t_start)

        idx_end_imu = np.searchsorted(self.data["time"], t_end, "right") - 1
        idx_end_gt = np.searchsorted(self.data["gt_time"], t_end, "right") - 1

        # Ensure indices are within bounds
        idx_end_imu = min(idx_end_imu, len(self.data["time"]) - 1)
        idx_end_gt = min(idx_end_gt, len(self.data["gt_time"]) - 1)

        for k in ["gt_time", "pos", "quat"]:
            self.data[k] = self.data[k][idx_start_gt:idx_end_gt+1]

        for k in ["time", "acc", "gyro"]:
            self.data[k] = self.data[k][idx_start_imu:idx_end_imu+1]

        ## start interpolation
        self.data["gt_orientation"] = self.interp_rot(
            self.data["time"], self.data["gt_time"], self.data["quat"]
        )
        self.data["gt_translation"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], self.data["pos"]
        )

        # Compute velocity from position derivatives
        self.compute_velocity()

        self.data["time"] = torch.tensor(self.data["time"])
        self.data["gt_time"] = torch.tensor(self.data["gt_time"])
        self.data["dt"] = (self.data["time"][1:] - self.data["time"][:-1])[:, None]
        self.data["mask"] = torch.ones(self.data["time"].shape[0], dtype=torch.bool)

        # Convert IMU data to torch tensors
        self.data["gyro"] = torch.tensor(self.data["gyro"])
        self.data["acc"] = torch.tensor(self.data["acc"])

        # when evaluation: load airimu or integrated orientation:
        self.set_orientation(rot_path, data_name, rot_type)
        
        # transform to global/body frame:
        self.update_coordinate(coordinate, mode)
        
        # remove gravity term
        self.remove_gravity(remove_g)
        
    def get_length(self):
        return self.data["time"].shape[0]

    def load_imu(self, h5_file_path):
        # Load IMU data from the M3ED dataset
        with h5py.File(h5_file_path, 'r') as f:
            imu_ts = f['ovc/imu/ts'][:]
            imu_accel = f['ovc/imu/accel'][:]
            imu_omega = f['ovc/imu/omega'][:]
            self.T_imu_to_camera = f['ovc/imu/calib/T_to_prophesee_left'][:]

        # Convert timestamps to seconds (microseconds)
        self.data["time"] = imu_ts / 1e6

        # Populate time array with constant time intervals
        dt = self.data["time"][1] - self.data["time"][0]
        self.data["time"] = np.ones_like(self.data["time"]) * dt
        self.data["time"] = np.cumsum(self.data["time"])
        print(f'dt: {dt}')

        # Convert from RDF (Right-Down-Forward) to RBD (Right-Backward-Down) coordinate frame
        self.data["gyro"] = imu_omega  # angular velocity in rad/s
        self.data["acc"]  = imu_accel  # acceleration in m/s^2

        self.data["gyro"][:, 0] =  imu_omega[:, 0] # Right -> Right
        self.data["gyro"][:, 1] = -imu_omega[:, 2] # Down -> Backward
        self.data["gyro"][:, 2] =  imu_omega[:, 1] # Forward -> Down

        self.data["acc"][:, 0] =  imu_accel[:, 0] # Right -> Right
        self.data["acc"][:, 1] = -imu_accel[:, 2] # Down -> Backward
        self.data["acc"][:, 2] =  imu_accel[:, 1] # Forward -> Down

        # filter out nan values
        for i in range(len(self.data["gyro"])):
            if np.isnan(self.data["gyro"][i]).any():
                self.data["gyro"][i] = self.data["gyro"][i - 1] if i > 0 else self.data["gyro"][i + 1]
        for i in range(len(self.data["acc"])):
            if np.isnan(self.data["acc"][i]).any():
                self.data["acc"][i] = self.data["acc"][i - 1] if i > 0 else self.data["acc"][i + 1]

        # Check for NaN or inf values in IMU data
        if np.isnan(self.data["gyro"]).any() or np.isinf(self.data["gyro"]).any():
            print("WARNING: IMU gyroscope contains NaN or inf values")
        if np.isnan(self.data["acc"]).any() or np.isinf(self.data["acc"]).any():
            print("WARNING: IMU acceleration contains NaN or inf values")

    def load_gt(self, gt_file_path):

        print(f"================={gt_file_path}=================")
        
        with h5py.File(gt_file_path, 'r') as f:
            gt_ts = f['ts'][:] 
            poses = f['Cn_T_C0'][:]
            
        # Convert timestamps to seconds - microseconds for GT data
        self.data["gt_time"] = gt_ts / 1e6

        R_rdf_to_rbd = np.array([[1,  0,  0],
                                 [0,  0, -1],
                                 [0, -1,  0]])

        poses = np.linalg.inv(poses)
        
        self.data["pos"] = np.zeros((len(poses), 3))
        self.data["quat"] = np.zeros((len(poses), 4))
        for i in range(len(poses)):
            R_i = poses[i][:3, :3]
            t_i = poses[i][:3, 3]

            # Transform from RDF to RBD
            R_it = R_rdf_to_rbd @ R_i @ np.linalg.inv(R_rdf_to_rbd)
            t_it = R_rdf_to_rbd @ t_i

            q_it = Rotation.from_matrix(R_it).as_quat(scalar_first = True)

            self.data["quat"][i] = q_it
            self.data["pos"][i] = t_it

    def interp_rot(self, time, opt_time, quat):
        # Ensure IMU timestamps are within GT time bounds
        time = np.clip(time, opt_time[0], opt_time[-1])
        
        imu_dt = torch.tensor(time - opt_time[0])
        gt_dt = torch.tensor(opt_time - opt_time[0])
        quat = torch.tensor(quat)
        quat = qinterp(quat, gt_dt, imu_dt).double()
        self.data["rot_wxyz"] = quat
        rot_xyzw = torch.zeros_like(quat)
        rot_xyzw[:, 3] = quat[:, 0]
        rot_xyzw[:, :3] = quat[:, 1:]
        return pp.SO3(rot_xyzw)

    def interp_xyz(self, time, opt_time, xyz):
        intep_x = np.interp(time, xp=opt_time, fp=xyz[:, 0])
        intep_y = np.interp(time, xp=opt_time, fp=xyz[:, 1])
        intep_z = np.interp(time, xp=opt_time, fp=xyz[:, 2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def compute_velocity(self):
        """Compute velocity from position derivatives"""
        gt_times = self.data["gt_time"]
        gt_pos = self.data["pos"]
        
        # Compute velocity using finite differences
        v_start = ((gt_pos[1] - gt_pos[0]) / (gt_times[1] - gt_times[0])).reshape((1, 3))
        gt_vel_raw = (gt_pos[1:] - gt_pos[:-1]) / (gt_times[1:] - gt_times[:-1])[:, None]
        gt_vel_raw = np.concatenate((v_start, gt_vel_raw), axis=0)
        
        # Apply smoothing filter
        gt_vel_x = np.convolve(gt_vel_raw[:, 0], np.ones(5) / 5, mode='same')
        gt_vel_y = np.convolve(gt_vel_raw[:, 1], np.ones(5) / 5, mode='same')
        gt_vel_z = np.convolve(gt_vel_raw[:, 2], np.ones(5) / 5, mode='same')
        
        gt_vel = np.stack([gt_vel_x, gt_vel_y, gt_vel_z]).transpose()
        
        # Interpolate velocity to IMU timestamps
        self.data["velocity"] = self.interp_xyz(
            self.data["time"], self.data["gt_time"], gt_vel
        )

    def update_coordinate(self, coordinate, mode):
        """
        Updates the data (imu / velocity) based on the required mode.
        :param coordinate: The target coordinate system ('glob_coord' or 'body_coord').
        :param mode: The dataset mode, only rotating the velocity during training. x,x, y,, z
         y, z
        
        """
        if coordinate is None:
            print("No coordinate system provided. Skipping update.")
            return
        try:
            if coordinate == "glob_coord":
                self.data["gyro"] = self.data["gt_orientation"] @ self.data["gyro"]
                self.data["acc"] = self.data["gt_orientation"] @ self.data["acc"]
            elif coordinate == "body_coord":
                self.g_vector = self.data["gt_orientation"].Inv() @ self.g_vector
                if mode != "infevaluate" and mode != "inference":
                    self.data["velocity"] = self.data["gt_orientation"].Inv() @ self.data["velocity"]
            else:
                raise ValueError(f"Unsupported coordinate system: {coordinate}")
        except Exception as e:
            print("An error occurred while updating coordinates:", e)
            raise e

    def set_orientation(self, exp_path, data_name, rotation_type):
        """
        Sets the ground truth orientation based on the provided rotation.
        :param exp_path: Path to the pickle file containing orientation data.
        :param rotation_type: The type of rotation within the pickle file (AirIMU corrected orientation / raw imu Pre-integration).
        """
        if rotation_type is None or rotation_type == "None" or rotation_type.lower() == "gtrot":
            return
        try:
            with open(exp_path, 'rb') as file:
                loaded_data = pickle.load(file)

            state = loaded_data[data_name]

            if rotation_type.lower() == "airimu":
                self.data["gt_orientation"] = state['airimu_rot']
            elif rotation_type.lower() == "integration":
                self.data["gt_orientation"] = state['inte_rot']
            else:
                print(f"Unsupported rotation type: {rotation_type}")
                raise ValueError(f"Unsupported rotation type: {rotation_type}")
        except FileNotFoundError:
            print(f"The file {exp_path} was not found.")
            raise

    def remove_gravity(self, remove_g):
        if remove_g is True:
            print("gravity has been removed")
            self.data["acc"] -= self.g_vector