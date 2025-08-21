from loaders.m3ed_loader import M3EDSequence, M3EDSequenceRecurrent
from loaders.kitti_loader import KITTISequence, KITTISequenceRecurrent
from loaders.airio_loader import SeqInfDataset, imu_seq_collate
from model.encoder_pipeline import EncoderPipeline
from model.pose_estimator import PoseEstimator
from test import TestEncoder, TestRecurrent, TestEncoderCache
from train import TrainEncoder, TrainRecurrent
from utils.egomotion_visualizer import EgomotionVisualizer
from utils.helper_functions import get_save_path, load_model, get_num_gpus, load_pickle_results, get_device
from ekf.casADI_EKFrunner import run_ekf
import numpy as np
from pathlib import Path

from torch.utils.data import DataLoader
import argparse
import torch
import json
import os

def get_dataset_class(dataset_name, uses_sequences=True):
    """Get the dataset class based on the dataset name"""
    if dataset_name == 'm3ed':
        if uses_sequences:
            return M3EDSequenceRecurrent
        else:
            return M3EDSequence
    elif dataset_name == 'kitti':
        if uses_sequences:
            return KITTISequenceRecurrent
        else:
            return KITTISequence
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")

def create_data_loaders(config, dataset_name='m3ed', mode='train', use_sequences=True, uses_visualizer=False):
    """Create data loaders for training or testing"""
    data_root = config['data-root']
    num_gpus = get_num_gpus()
    num_workers = 4 * num_gpus if num_gpus > 0 else 0
    
    if mode == 'train':
        data_names = config['train']['data_names']
        batch_size = config['train']['batch_size']
        sequence_length = config['train']['sequence_length']
        use_cache = config['train'].get('use_cache', False)
        shuffle_train = True
    else:  # test
        data_names = config['test']['data_names']
        batch_size = config['test']['batch_size']
        sequence_length = config['test'].get('sequence_length', config['train']['sequence_length'])
        use_cache = config['test'].get('use_cache', False)
        shuffle_train = False

    # Override num_workers if specified
    if uses_visualizer:
        num_workers = 0

    datasets = []
    for name in data_names:

        data_path = Path(os.path.join(data_root, name))
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        dataset_class = get_dataset_class(dataset_name, use_sequences)
        
        if use_cache and 'cache-root' in config:
            cache_root = config['cache-root']
            cached_data_path = os.path.join(cache_root, name.replace("flow.h5", "encoded.npy"))
        
            dataset = dataset_class(data_path, 
                                    sequence_length=sequence_length,
                                    cache_path=cached_data_path)
        
        else:
            dataset = dataset_class(data_path, sequence_length=sequence_length)
        
        datasets.append(dataset)

    data = torch.utils.data.ConcatDataset(datasets)
    
    if mode == 'train':
        # Split into train and test sets
        train_set, test_set = torch.utils.data.random_split(
            data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))]
        )
        
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, test_loader
    else:
        # Return single test loader
        test_loader = DataLoader(
            dataset=data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return test_loader

def train_recurrent(config, dataset_name):
    """Train the recurrent pose estimation model"""
    print("Starting recurrent model training...")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(config, dataset_name=dataset_name, mode='train', use_sequences=True)

    save_path = get_save_path(config)
    
    # Load encoding model
    encoder_checkpoint = config['checkpoints']['encoder']
    encoding_model = EncoderPipeline()
    encoding_model = load_model(encoding_model, encoder_checkpoint)
    
    # Create pose estimation model
    batch_size = config['train']['batch_size']
    sequence_length = config['train']['sequence_length']
    model = PoseEstimator(batch_size=batch_size, sequence_length=sequence_length)
    model.train()
    
    # Train the model
    trainer = TrainRecurrent(
        model,
        encoding_model.encoder,
        config,
        train_loader,
        test_loader,
        save_path
    )
    
    trainer.summary()
    trainer._train()
    
    print("Recurrent model training complete.")

def test_recurrent(config, dataset_name):
    """Test the recurrent pose estimation model"""
    print("Starting recurrent model testing...")
    
    # Create test data loader with num_workers=0 for visualizer compatibility
    test_loader = create_data_loaders(config, dataset_name=dataset_name, mode='test', use_sequences=True, uses_visualizer=True)

    save_path = os.path.join(get_save_path(config), config['test']['data_names'][0])

    # Setup models
    encoding_model = EncoderPipeline()
    visualizer = EgomotionVisualizer(save_path)

    # Load encoding model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        encoding_model = load_model(encoding_model, encoder_checkpoint)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained encoder model for testing...")
    
    # Load recurrent model
    sequence_length = config['test'].get('sequence_length', config['train']['sequence_length'])
    model = PoseEstimator(batch_size=config['train']['batch_size'], sequence_length=sequence_length)
    try:
        recurrent_checkpoint = config['checkpoints']['recurrent']
        model = load_model(model, recurrent_checkpoint)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained recurrent model for testing...")
    
    # Test the model
    tester = TestRecurrent(model, encoding_model.encoder, config, test_loader, save_path, visualizer=visualizer, save=True)
    tester.summary()
    tester._test()
    
    print("Recurrent model testing complete.")

def train_encoder(config, dataset_name):
    """Train the encoder model"""
    print("Starting encoder training...")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(config, dataset_name=dataset_name, mode='train', use_sequences=False)

    save_path = get_save_path(config)
    
    # Create and train encoder model
    model = EncoderPipeline()
    model.train()
    
    trainer = TrainEncoder(model, config, train_loader, test_loader, save_path)
    trainer.summary()
    trainer._train()
    
    print("Encoder training complete.")

def cache_encoder(config, dataset_name):
    """Cache encoder outputs for all test data"""
    print("Starting encoder caching...")
    
    # Create test data loader (single dataset processing happens in TestEncoderCache)
    test_loader = create_data_loaders(config, dataset_name=dataset_name, mode='test', use_sequences=False)
    save_path = get_save_path(config)

    # Setup model
    model = EncoderPipeline()
    
    # Load trained model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        model = load_model(model, encoder_checkpoint)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained model for caching...")
    
    # Cache the encodings using only the encoder part
    tester = TestEncoderCache(model.encoder, config, test_loader, save_path)
    tester.summary()
    cache_path = tester._test()
    
    print(f"Encoder caching complete. Files saved to: {cache_path}")

def test_encoder(config, dataset_name):
    """Test the encoder model"""
    print("Starting encoder testing...")
    
    # Create test data loader
    test_loader = create_data_loaders(config, dataset_name=dataset_name, mode='test', use_sequences=False)
    
    # Setup model
    model = EncoderPipeline()
    save_path = get_save_path(config)
    
    # Load trained model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        model = load_model(model, encoder_checkpoint)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained model for testing...")
    
    # Test the model
    tester = TestEncoder(model, config, test_loader, save_path)
    tester.summary()
    tester._test()
    
    print("Encoder testing complete.")

def ekf(config, dataset_name):
    airio_path = config['ekf']['airio_path']
    airimu_path = config['ekf']['airimu_path']
    egomotion_path = config['ekf']['egomotion_path']

    device = get_device(config)

    inference_state_load = load_pickle_results(airio_path)
    airimu_ori_load = load_pickle_results(airimu_path)

    data_root = config['data-root']
    data_names = config['ekf']['data_names']
    
    for name in data_names:

        data_path = Path(os.path.join(data_root, name))
        if not data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        airimu_state = airimu_ori_load[name]

        egomotion_output = np.load(os.path.join(egomotion_path, f"{name}.npy"))

        airimu_dataset = SeqInfDataset(
            data_root,
            name,
            airimu_state,
            device=device,
            name=config['name'],
            duration=1,
            step_size=1
        )

        run_ekf(egomotion_output, airimu_dataset, name, inference_state_load)

def main():
    parser = argparse.ArgumentParser(description='Egomotion estimation training and testing')
    parser.add_argument('model_type', choices=['encoder', 'recurrent', 'ekf'],
                       help='Type of model to work with')
    parser.add_argument('action', choices=['train', 'test', 'cache'],
                       help='Action to perform')
    parser.add_argument('-d', '--dataset', choices=['m3ed', 'kitti'], default='m3ed',
                       help='Dataset type to use (default: m3ed)')
    
    args = parser.parse_args()
    
    # Determine config file
    config_path = f'configs/{args.dataset}_{args.model_type}.json'
    
    # Load configuration
    try:
        config = json.load(open(config_path))
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print(f"Available configs: configs/encoder.json, configs/recurrent.json")
        return
    
    # Execute command
    if args.model_type == 'encoder':
        if args.action == 'train':
            train_encoder(config, args.dataset)
        elif args.action == 'test':
            test_encoder(config, args.dataset)
        elif args.action == 'cache':
            cache_encoder(config, args.dataset)
    elif args.model_type == 'recurrent':
        if args.action == 'train':
            train_recurrent(config, args.dataset)
        elif args.action == 'test':
            test_recurrent(config, args.dataset)
        elif args.action == 'cache':
            print("Cache action is only available for encoder model type")
            print("Usage: python main.py encoder cache")
    elif args.model_type == 'ekf':
        run_ekf(config)

if __name__ == "__main__":
    main()