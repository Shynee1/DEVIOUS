import h5py
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import weakref
from scipy.spatial.transform import Rotation
import os

class M3EDSequence(Dataset):
    def __init__(self, h5_path: Path, sequence_length: int = 0, cache_path: Path = None):
  
        self.h5f = h5py.File(h5_path, 'r')
        self.timestamps = np.asarray(self.h5f['timestamps'], dtype='float64')

        self.gt_path = h5_path.as_posix().replace("eraft", "depth_gt")
        self.gt_h5 = h5py.File(self.gt_path, 'r') if Path(self.gt_path).exists() else None

        self._finalizer = weakref.finalize(self, self.close_callback, self.h5f)
    
    @staticmethod
    def close_callback(h5f):
        h5f.close()

    def resize_flow(self, flow):
        """
        Resize flow from 1280x720 to 1280x640 using center cropping.
        Input flow shape: (H, W, 2) = (720, 1280, 2)
        Output flow shape: (H, W, 2) = (640, 1280, 2)
        """
        original_height = flow.shape[0]  # 720
        target_height = 384
        
        # Calculate crop parameters for center cropping
        crop_top = (original_height - target_height) // 2  # (720 - 384) // 2 = 168
        crop_bottom = crop_top + target_height  # 168 + 384 = 552

        # Center crop the flow
        cropped_flow = flow[crop_top:crop_bottom, :, :]  # Keep all width, crop height
        
        return cropped_flow

    def get_image_width_height(self):
        return 384, 1280  # Height x Width after resizing
    
    def __len__(self):
        return len(self.timestamps)

    def get_data_sample(self, index):
        output = {
            'file_index': index,
            'timestamp': self.timestamps[index],
        }

        # Load flow from E-RAFT
        flow = self.h5f['flow_data'][index]
        flow = np.asarray(flow, dtype='float32')
        
        # Resize flow from 1280x720 to 1280x640
        flow = self.resize_flow(flow)
        
        valid_mask = np.where((flow[..., 0] != 0) | (flow[..., 1] != 0), 1, 0)
        
        output['flow'] = flow.transpose(2, 0, 1)
        output['valid_mask'] = valid_mask.astype(np.float32)

        # Load ground truth pose if available
        if self.gt_h5 is not None:
            output['gt_transform'] = self.load_ground_truth_transform(index)

        return output

    def load_ground_truth_transform(self, index):
        assert self.gt_h5 is not None, "No ground truth transform file provided!"
        return np.asarray(self.gt_h5['Cn_T_C0'][index], dtype='float32')

    def __getitem__(self, idx):
        return self.get_data_sample(idx)

class M3EDSequenceRecurrent(M3EDSequence):
    def __init__(self, h5_path: Path, sequence_length=5, cache_path:Path=None):
        super().__init__(h5_path, sequence_length=sequence_length, cache_path=cache_path)
        self.sequence_length = sequence_length
        self.cached_encodings = None
        
        if cache_path is not None and os.path.exists(cache_path):
            self.cached_encodings = np.load(cache_path)
            print(f"Cached encodings loaded from {cache_path}")
        else:
            print(f"WARNING: Could not load cached encoding from {cache_path}")

    def calculate_relative_transform(self, prev_transform, curr_transform):
        # Ensure input transforms are float32
        prev_transform = prev_transform.astype(np.float32)
        curr_transform = curr_transform.astype(np.float32)
        
        relative_transform = curr_transform @ np.linalg.inv(prev_transform)
        relative_translation = relative_transform[:3, 3]
        relative_rotation = Rotation.from_matrix(relative_transform[:3, :3]).as_euler('xyz', degrees=True)
        result = np.concatenate([relative_translation, relative_rotation])
        
        # Ensure the result is float32
        return result.astype(np.float32)
    
    def __len__(self):
        return max(0, len(self.timestamps) - self.sequence_length + 1)

    def __getitem__(self, idx):
        # Map idx to first index of the sequence
        valid_idx = idx + self.sequence_length - 1

        if valid_idx >= len(self.timestamps):
            raise IndexError(f"Index {idx} maps to {valid_idx} which is out of bounds. Dataset has {len(self.timestamps)} timestamps.")

        flows = []
        valid_masks = []
        frame_ids = []
        timestamps = []
        encodings = []

        prev_abs_gt = None
        curr_abs_gt = None

        for i in range(self.sequence_length - 1, -1, -1):
            sample_index = valid_idx - i
            
            sample = self.get_data_sample(sample_index)
            flows.append(sample['flow'])
            valid_masks.append(sample['valid_mask'])
            frame_ids.append(sample['file_index'])
            timestamps.append(sample['timestamp'])

            if self.cached_encodings is not None:
                encoding = self.cached_encodings[sample_index]
                encodings.append(encoding)

            if 'gt_transform' not in sample:
                raise ValueError(f"Ground truth transform not found for index {sample_index}.")
            
            if i == 1:
                prev_abs_gt = sample['gt_transform']

            if i == 0:
                curr_abs_gt = sample['gt_transform']

        gt_transform = self.calculate_relative_transform(prev_abs_gt, curr_abs_gt)

        output = {
            'flow_sequence': np.asarray(flows, dtype=np.float32),
            'valid_masks': np.asarray(valid_masks, dtype=np.float32),
            'frame_ids': np.asarray(frame_ids, dtype=np.int64),
            'timestamps': np.asarray(timestamps, dtype=np.float64),
            'gt_transform': gt_transform
        }

        if len(encodings) != 0:
            output['encodings'] = np.asarray(encodings)

        return output