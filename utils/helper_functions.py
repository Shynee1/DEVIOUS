import os
import torch

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
    
def load_model(model, checkpoint_path, device=None):
    """Load model weights from checkpoint"""
    if os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        if device is not None:
            model.to(device)
        print("Model loaded successfully!")
        return model
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
