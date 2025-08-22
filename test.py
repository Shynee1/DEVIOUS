import torch
import torch.nn as nn
import tqdm
import numpy as np
import os

from utils.loss_functions import RMSELoss, DiagLnCovLoss
from utils.visualization import visualize_optical_flow
from utils.helper_functions import get_num_gpus, get_device

class Test(object):
    def __init__(self, model, config, data_loader, save_path):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        self.save_path = save_path

        use_cuda = config.get('cuda', True) and torch.cuda.is_available()
        requested_gpu = config.get('gpu', 0)
        num_gpus = get_num_gpus() if use_cuda else 0

        if use_cuda and num_gpus > 0:
            # Primary device
            self.device = torch.device(f'cuda:{requested_gpu if requested_gpu < num_gpus else 0}')
            model.to(self.device)
            if num_gpus > 1:
                model = nn.DataParallel(model)
        else:
            self.device = torch.device('cpu')

        self.model.to(self.device)

        self.loss_function = nn.MSELoss()

    def summary(self):
        print("==================================== TEST SUMMARY ====================================")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Test Set: {self.data_loader.__class__.__name__}")
        print(f"Dataset Length: {len(self.data_loader.dataset)}")
        print(f"Batch Size: {self.data_loader.batch_size}")
        print("======================================================================================")

    def _test(self):
        raise NotImplementedError

class TestEncoder(Test):
    def __init__(self, model, config, data_loader, save_path):
        super().__init__(model, config, data_loader, save_path)
        self.loss_function = RMSELoss()

    def _test(self):
        self.model.eval()

        with torch.no_grad():
            total_loss = 0.0
            t_range = tqdm.tqdm(self.data_loader)
            
            # Reset data loader
            t_range = tqdm.tqdm(self.data_loader)

            for batch_idx, batch in enumerate(t_range):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                flow = batch['flow']
                valid_mask = batch['valid_mask'].bool()

                # Forward pass
                outputs = self.model(flow)

                if self.save_path is not None:
                    flow_cpu = flow.squeeze().cpu().numpy()
                    outputs_cpu = outputs.squeeze().cpu().numpy()

                    os.makedirs(self.save_path, exist_ok=True)

                    batch_size, C, H, W = outputs_cpu.shape

                    for i in range(batch_size):
                        visualize_optical_flow(flow_cpu[i], os.path.join(self.save_path, f"input_{batch_idx+1}.png"), text=f"Input Flow Batch {batch_idx+1}")
                        visualize_optical_flow(outputs_cpu[i], os.path.join(self.save_path, f"output_{batch_idx+1}.png"), text=f"Reconstructed Flow Batch {batch_idx+1}")

                # Compute loss
                loss = self.loss_function(outputs, flow, valid_mask)
                total_loss += loss.item()

            avg_loss = total_loss / len(self.data_loader)

            print(f"Test Loss: {avg_loss}")

        return avg_loss

class TestEncoderCache(Test):
    def __init__(self, model, config, data_loader, save_path):
        super().__init__(model, config, data_loader, save_path)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.config = config
        
    def _test(self):
        self.model.eval()
        
        # Process each dataset separately to maintain proper file mapping
        cache_root = self.config.get('cache-root', 'cached_encodes')
        
        # Get dataset names from config
        test_data_names = self.config['test']['data_names']
        
        for dataset_idx, dataset_name in enumerate(test_data_names):
            print(f"Processing dataset: {dataset_name}")
            
            # Create dataset-specific loader
            from loaders.m3ed_loader import M3EDSequence
            data_path = os.path.join(self.config['data-root'], dataset_name)
            dataset = M3EDSequence(h5_path=data_path)
            
            dataset_loader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=self.data_loader.batch_size,
                shuffle=False,
                num_workers=0
            )
            
            all_encodings = []
            
            with torch.no_grad():
                t_range = tqdm.tqdm(dataset_loader)
                
                for batch_idx, batch in enumerate(t_range):
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                    flow = batch['flow']
                    
                    # Forward pass through encoder only
                    encoded, _ = self.model(flow)  # This should be just the encoder
                    
                    # Apply max pooling
                    pooled = self.pooling(encoded)
                    
                    # Move to CPU and convert to numpy
                    pooled_cpu = pooled.cpu().numpy()
                    
                    # Store encodings in order
                    for i in range(pooled_cpu.shape[0]):
                        encoding = pooled_cpu[i]
                        all_encodings.append(encoding)

                    t_range.set_description(f"Batch {batch_idx+1}/{len(dataset_loader)}")

            # Convert to numpy array
            encodings_array = np.stack(all_encodings)
            
            # Save to cache file
            cache_filename = dataset_name.replace("flow.h5", "encoded.npy")
            cache_path = os.path.join(cache_root, cache_filename)
            
            # Ensure cache directory exists
            os.makedirs(cache_root, exist_ok=True)
            
            np.save(cache_path, encodings_array)
            print(f"Saved {len(encodings_array)} encodings to {cache_path}")
            print(f"Encoding shape: {encodings_array.shape}")
        
        print("All datasets processed and cached!")
        return cache_root
    
class TestRecurrent(Test):
    def __init__(self, recurrent_model, encoding_model, config, data_loader, save_path, visualizer=None, save=False):
        super().__init__(recurrent_model, config, data_loader, save_path)
        self.loss_function = DiagLnCovLoss()
        self.encoding_model = encoding_model
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sequence_length = config['train']['sequence_length']
        self.visualizer = visualizer
        self.save = save

        device = get_device(config)
        self.encoding_model.to(device)

    def encode_flows(self, flow_sequence):
        batch_size, seq_len = flow_sequence.shape[:2]
        
        # Pre-allocate the output tensor
        encoded_flows = torch.zeros(
            (batch_size, seq_len, 1024, 3, 10),
            device=flow_sequence.device,
            dtype=flow_sequence.dtype
        )
        
        # Process each frame and store directly in pre-allocated tensor
        for i in range(seq_len):
            flow = flow_sequence[:, i, :, :, :]
            encoded, _ = self.encoding_model(flow)
            encoded_flows[:, i] = self.pooling(encoded)
        
        return encoded_flows

    def _test(self):
        self.model.eval()

        t_range = tqdm.tqdm(self.data_loader)
        total_loss = 0.0
        save_outputs = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(t_range):

                # Move batch to devices
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                flow = batch['flow_sequence']                       # (batch, seq_length, C, H, W)
                ground_truth = batch['gt_transform']                # (batch, 6)
                timestamps = batch['timestamps']                    # (batch, seq_length)

                if 'encodings' in batch:
                    encodings = batch['encodings']
                else:
                    encodings = self.encode_flows(flow)

                # Forward pass
                outputs, cov = self.model(encodings)               # (batch, 6)

                # Save and visualize results
                batch_size = outputs.shape[0]
                for i in range(batch_size):
                    timestamp = timestamps[i][-1].item()
                    output = np.concatenate([[timestamp], outputs[i].cpu().numpy(), cov[i].cpu().numpy()])
                    save_outputs.append(output)

                    if self.visualizer is not None:
                        self.visualizer.add_measurement(timestamp, outputs[i], ground_truth[i])

                if self.visualizer is not None:
                    if batch_idx % 10 == 0:
                        self.visualizer.update_plots(save_plots=True)
                        self.visualizer.update_trajectory_plots(save_plots=True)

                # Compute loss
                loss = self.loss_function(outputs, ground_truth, cov)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.data_loader)

        if self.visualizer is not None:
            self.visualizer.update_plots(save_plots=True)
            self.visualizer.update_trajectory_plots(save_plots=True)

        if self.save:
            parent = os.path.dirname(self.save_path)
            os.makedirs(parent, exist_ok=True)
            save_outputs = np.array(save_outputs)
            np.save(self.save_path, save_outputs)
            print(f"Saved test outputs to {self.save_path}")

        print(f"Test Loss: {avg_loss}")

        return avg_loss
