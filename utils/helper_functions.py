import os
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import numpy as np
import pickle 
import io

def get_save_path(config):
    save_root = config.get('save_dir', 'saved')
    name = config.get('name', 'run')
    
    # Ensure save directory exists
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    path = os.path.join(save_root, name)

    # Check if path already exists. if yes -> append a number
    if os.path.exists(path):
        i = 1
        while os.path.exists(path + "_" + str(i)):
            i += 1
        path = path + '_' + str(i)
    
    return path

def get_device(config):
    """Get the appropriate device based on config"""
    if config['cuda'] and torch.cuda.is_available():
        return torch.device('cuda:' + str(config['gpu']))
    else:
        return torch.device('cpu')
    
def get_num_gpus():
    return torch.cuda.device_count() if torch.cuda.is_available() else 0

def load_model(model, checkpoint_path, strict=True):
    """Load model weights from checkpoint.
    Handles checkpoints saved from both single-GPU and DataParallel models.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading model from {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location='cpu')

    # Try direct load first
    try:
        model.load_state_dict(state, strict=strict)
    except RuntimeError as e:
        # Attempt to strip 'module.' prefixes if present
        if any(k.startswith('module.') for k in state.keys()):
            stripped_state = {k.replace('module.', '', 1): v for k, v in state.items()}
            model.load_state_dict(stripped_state, strict=strict)
        else:
            raise e
        
    print("Model loaded successfully!")
    return model

def qinterp(qs, t, t_int):
    qs = R.from_quat(qs.numpy())
    slerp = Slerp(t, qs)
    interp_rot = slerp(t_int).as_quat()
    return torch.tensor(interp_rot)

def interp_xyz(time, opt_time, xyz):
    intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
    intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
    intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
    inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
    return torch.tensor(inte_xyz)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

def load_pickle_results(path):
    if path is not None:
        if os.path.isfile(path):
            with open(path, 'rb') as handle:
                state_load = CPU_Unpickler(handle).load()
            return state_load
        else:
            raise Exception(f"Unable to load the network result: {path}")