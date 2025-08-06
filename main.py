from loaders.m3ed_loader import M3EDSequence, M3EDSequenceRecurrent
from model.encoder_pipeline import EncoderPipeline
from model.pose_estimator import PoseEstimator
from test import TestEncoder, TestRecurrent, TestEncoderCache
from train import TrainEncoder, TrainRecurrent
from utils.egomotion_visualizer import EgomotionVisualizer
from utils.helper_functions import get_save_path, get_device, load_model

from torch.utils.data import DataLoader
import argparse
import torch
import json
import os

def create_data_loaders(config, mode='train', use_sequences=True, uses_visualizer=False):
    """Create data loaders for training or testing"""
    data_root = config['data-root']
    
    if mode == 'train':
        data_names = config['train']['data_names']
        batch_size = config['train']['batch_size']
        sequence_length = config['train']['sequence_length']
        num_workers = config['train'].get('num_workers', 4)
        use_cache = config['train'].get('use_cache', False)
        shuffle_train = True
    else:  # test
        data_names = config['test']['data_names']
        batch_size = config['test']['batch_size']
        sequence_length = config['test'].get('sequence_length', config['train']['sequence_length'])
        num_workers = config['test'].get('num_workers', 0)
        use_cache = config['test'].get('use_cache', False)
        shuffle_train = False

    # Override num_workers if specified
    if uses_visualizer:
        num_workers = 0

    datasets = []
    for name in data_names:
        data_path = os.path.join(data_root, name)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        if use_sequences:
            if use_cache and 'cache-root' in config:
                cache_root = config['cache-root']
                cached_data_path = os.path.join(cache_root, name.replace("flow.h5", "encoded.npy"))
                dataset = M3EDSequenceRecurrent(h5_path=data_path, 
                                              sequence_length=sequence_length,
                                              cache_path=cached_data_path)
            else:
                dataset = M3EDSequenceRecurrent(h5_path=data_path, sequence_length=sequence_length)
        else:
            dataset = M3EDSequence(h5_path=data_path)
        
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

def train_recurrent(config):
    """Train the recurrent pose estimation model"""
    print("Starting recurrent model training...")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(config, mode='train', use_sequences=True)
    
    device = get_device(config)
    save_path = get_save_path(config)
    
    # Load encoding model
    encoder_checkpoint = config['checkpoints']['encoder']
    encoding_model = EncoderPipeline()
    encoding_model = load_model(encoding_model, encoder_checkpoint, device)
    
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

def test_recurrent(config):
    """Test the recurrent pose estimation model"""
    print("Starting recurrent model testing...")
    
    # Create test data loader with num_workers=0 for visualizer compatibility
    test_loader = create_data_loaders(config, mode='test', use_sequences=True, uses_visualizer=True)
    
    # Setup device
    device = get_device(config)
    save_path = get_save_path(config)
    
    # Setup models
    encoding_model = EncoderPipeline()
    visualizer = EgomotionVisualizer(save_path)

    # Load encoding model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        encoding_model = load_model(encoding_model, encoder_checkpoint, device)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained encoder model for testing...")
    
    # Load recurrent model
    sequence_length = config['test'].get('sequence_length', config['train']['sequence_length'])
    model = PoseEstimator(batch_size=config['train']['batch_size'], sequence_length=sequence_length)
    try:
        recurrent_checkpoint = config['checkpoints']['recurrent']
        model = load_model(model, recurrent_checkpoint, device)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained recurrent model for testing...")
    
    # Test the model
    tester = TestRecurrent(model, encoding_model.encoder, config, test_loader, visualizer=visualizer)
    tester.summary()
    tester._test()
    
    print("Recurrent model testing complete.")

def train_encoder(config):
    """Train the encoder model"""
    print("Starting encoder training...")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(config, mode='train', use_sequences=False)

    save_path = get_save_path(config)
    
    # Create and train encoder model
    model = EncoderPipeline()
    model.train()
    
    trainer = TrainEncoder(model, config, train_loader, test_loader, save_path)
    trainer.summary()
    trainer._train()
    
    print("Encoder training complete.")

def cache_encoder(config):
    """Cache encoder outputs for all test data"""
    print("Starting encoder caching...")
    
    # Create test data loader (single dataset processing happens in TestEncoderCache)
    test_loader = create_data_loaders(config, mode='test', use_sequences=False)
    
    # Setup device  
    device = get_device(config)
    
    # Setup model
    model = EncoderPipeline()
    
    # Load trained model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        model = load_model(model, encoder_checkpoint, device)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained model for caching...")
    
    # Cache the encodings using only the encoder part
    tester = TestEncoderCache(model.encoder, config, test_loader)
    tester.summary()
    cache_path = tester._test()
    
    print(f"Encoder caching complete. Files saved to: {cache_path}")

def test_encoder(config):
    """Test the encoder model"""
    print("Starting encoder testing...")
    
    # Create test data loader
    test_loader = create_data_loaders(config, mode='test', use_sequences=False)
    
    # Setup model
    model = EncoderPipeline()
    
    # Load trained model
    try:
        encoder_checkpoint = config['checkpoints']['encoder']
        model = load_model(model, encoder_checkpoint)
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: {e}")
        print("Using untrained model for testing...")
    
    # Test the model
    tester = TestEncoder(model.encoder, config, test_loader)
    tester.summary()
    tester._test()
    
    print("Encoder testing complete.")

def main():
    parser = argparse.ArgumentParser(description='Egomotion estimation training and testing')
    parser.add_argument('model_type', choices=['encoder', 'recurrent'],
                       help='Type of model to work with')
    parser.add_argument('action', choices=['train', 'test', 'cache'],
                       help='Action to perform')
    parser.add_argument('--config', 
                       help='Path to config file (default: configs/m3ed_{model_type}.json)')
    
    args = parser.parse_args()
    
    # Determine config file
    if args.config:
        config_path = args.config
    else:
        config_path = f'configs/m3ed_{args.model_type}.json'
    
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
            train_encoder(config)
        elif args.action == 'test':
            test_encoder(config)
        elif args.action == 'cache':
            cache_encoder(config)
    elif args.model_type == 'recurrent':
        if args.action == 'train':
            train_recurrent(config)
        elif args.action == 'test':
            test_recurrent(config)
        elif args.action == 'cache':
            print("Cache action is only available for encoder model type")
            print("Usage: python main.py encoder cache")

if __name__ == "__main__":
    main()