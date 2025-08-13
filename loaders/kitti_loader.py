import pykitti
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from scipy.spatial.transform import Rotation
import os
import torch
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
from torchvision.transforms import CenterCrop
from torchvision.transforms.functional import to_tensor
from PIL import Image

class KITTISequence(Dataset):
    def __init__(self, sequence_path: Path, sequence_length: int = 0, cache_path: Path = None, lazy_cache_dir: Path | None = None):
        base_dir = sequence_path.parent.parent
        sequence = sequence_path.name
        self.data = pykitti.odometry(base_dir, sequence)
        self.timestamps = np.asarray([t.total_seconds() for t in self.data.timestamps])
        self.gt = self.data.poses

        if (len(self.gt) == 0):
            self.gt = None
        
        self.device = 'cuda' if torch.cuda.is_available()  else 'cpu'

        self.raft_target_h = 384
        self.raft_target_w = 1280

        # Directory for lazy per-flow caching (npy per pair)
        if lazy_cache_dir is None:
            self.lazy_cache_dir = Path(sequence_path) / 'raft_flows_lazy'
        else:
            self.lazy_cache_dir = Path(lazy_cache_dir)
        self.lazy_cache_dir.mkdir(parents=True, exist_ok=True)

        self.load_raft()

        self.num_pairs = len(self.timestamps) - 1

    # ---------------- RAFT Flow Generation ----------------
    def load_raft(self):
        if not hasattr(KITTISequence, '_raft_model'):
            KITTISequence._raft_model = None

        if KITTISequence._raft_model is None:
            model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            model = model.eval()
            KITTISequence._raft_model = model

        self.raft = KITTISequence._raft_model

    def prepare_raft_image(self, img):
        resized = img.resize((self.raft_target_w, self.raft_target_h), Image.BILINEAR)
        output = to_tensor(resized).unsqueeze(0)
        return output

    def get_flow(self, index: int):
        img1 = self.data.get_cam2(index)
        img2 = self.data.get_cam2(index + 1)

        t1 = self.prepare_raft_image(img1)
        t2 = self.prepare_raft_image(img2)

        with torch.no_grad():
            flow = self.raft(t1, t2)

        flow = flow[-1].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
        return flow

    def get_image_width_height(self):
        return self.raft_target_h, self.raft_target_w  # (H,W)

    def __len__(self):
        return len(self.timestamps) - 1

    def get_data_sample(self, index):
        output = {
            'file_index': index,
            'timestamp': self.timestamps[index],
        }

        flow = self.get_flow(index).astype(np.float32) # (H, W, 2)
        valid_mask = np.where((flow[..., 0] != 0) | (flow[..., 1] != 0), 1, 0).astype(np.float32)

        output['flow'] = flow.transpose(2, 0, 1)
        output['valid_mask'] = valid_mask

        if self.gt is not None:
            output['gt_transform'] = np.asarray(self.gt[index], dtype=np.float32)

        return output

    def __getitem__(self, idx):
        return self.get_data_sample(idx)

class KITTISequenceRecurrent(KITTISequence):
    def __init__(self, sequence_path: Path, sequence_length=5, cache_path: Path = None):
        super().__init__(sequence_path)

        self.sequence_length = sequence_length
        self.cached_encodings = None

        if cache_path is not None and os.path.exists(cache_path):
            self.cached_encodings = np.load(cache_path)
            print(f"Cached encodings loaded from {cache_path}")
        elif cache_path is not None:
            print(f"WARNING: Could not load cached encoding from {cache_path}")

    def calculate_relative_transform(self, prev_transform, curr_transform):
        prev_transform = prev_transform.astype(np.float32)
        curr_transform = curr_transform.astype(np.float32)

        relative_transform = curr_transform @ np.linalg.inv(prev_transform)
        relative_translation = relative_transform[:3, 3]
        relative_rotation = Rotation.from_matrix(relative_transform[:3, :3]).as_euler('xyz', degrees=True)

        result = np.concatenate([relative_translation, relative_rotation])
        return result.astype(np.float32)

    def __len__(self):
        return max(0, len(self.timestamps) - self.sequence_length + 1)

    def __getitem__(self, idx):
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
                encodings.append(self.cached_encodings[sample_index])

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