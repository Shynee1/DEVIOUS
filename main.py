from loaders.m3ed_loader import M3EDSequence, M3EDSequenceRecurrent
from loaders.kitti_loader import KITTISequence, KITTISequenceRecurrent
from loaders.airio_loader import SeqInfDataset
from model.encoder_pipeline import EncoderPipeline
from model.pose_estimator import PoseEstimator
from test import TestEncoder, TestRecurrent, TestEncoderCache
from train import TrainEncoder, TrainRecurrent
from utils.egomotion_visualizer import EgomotionVisualizer
from utils.helper_functions import get_save_path, load_model, get_num_gpus, load_pickle_results, get_device
from ekf.casADI_EKFrunner import run_ekf

import numpy as np
from pathlib import Path
import pickle
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

def create_train_loaders(config, dataset_name='m3ed', use_sequences=True):
    data_root = config['data-root']
    num_gpus = get_num_gpus()
    num_workers = 4 * num_gpus if num_gpus > 0 else 0
    data_names = config['train']['data_names']
    batch_size = config['train']['batch_size']
    sequence_length = config['train']['sequence_length']
    use_cache = config['train'].get('use_cache', False)

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
    
    # Split into train and test sets
    train_set, test_set = torch.utils.data.random_split(
        data, [int(0.8 * len(data)), len(data) - int(0.8 * len(data))]
    )
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader
        

def create_test_loaders(config, dataset_name='m3ed', use_sequences=True, uses_visualizer=False):
    data_root = config['data-root']
    num_gpus = get_num_gpus()
    num_workers = 4 * num_gpus if num_gpus > 0 else 0
    data_names = config['test']['data_names']
    batch_size = config['test']['batch_size']
    sequence_length = config['test']['sequence_length']
    use_cache = config['test'].get('use_cache', False)

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
        
        test_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        datasets.append(test_loader)

    return datasets

def train_recurrent(config, dataset_name):
    """Train the recurrent pose estimation model"""
    print("Starting recurrent model training...")

    # Create data loaders
    train_loader, test_loader = create_train_loaders(config, dataset_name=dataset_name, use_sequences=True)

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
    
    test_loaders = create_test_loaders(config, dataset_name=dataset_name, use_sequences=True, uses_visualizer=False)
    save_path = get_save_path(config)

    # Setup models
    encoding_model = EncoderPipeline()

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

    save_dict = {}
    output_path = os.path.join(save_path, "devious_output.pickle")
    for idx, test_loader in enumerate(test_loaders):
        dataset_name = config['test']['data_names'][idx]
        visualizer = EgomotionVisualizer(os.path.join(save_path, dataset_name))

        # Test the model
        tester = TestRecurrent(model, encoding_model.encoder, config, test_loader, save_path, visualizer=visualizer)
        tester.summary()
        _, outputs = tester._test()

        save_dict[dataset_name] = outputs

        # Save all outputs to a single file
        pickle.dump(save_dict, open(output_path, 'wb'))

    print("Recurrent model testing complete.")

def train_encoder(config, dataset_name):
    """Train the encoder model"""
    print("Starting encoder training...")
    
    # Create data loaders
    train_loader, test_loader = create_train_loaders(config, dataset_name=dataset_name, use_sequences=False)

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
    test_loaders = create_test_loaders(config, dataset_name=dataset_name, use_sequences=False)
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
    for test_loader in test_loaders:
        tester = TestEncoderCache(model.encoder, config, test_loader, save_path)
        tester.summary()
        cache_path = tester._test()

    print(f"Encoder caching complete. Files saved to: {cache_path}")

def test_encoder(config, dataset_name):
    """Test the encoder model"""
    print("Starting encoder testing...")
    
    # Create test data loader
    test_loaders = create_test_loaders(config, dataset_name=dataset_name, use_sequences=False)
    
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
    for test_loader in test_loaders:
        tester = TestEncoder(model.encoder, config, test_loader, save_path, save=True)
        tester.summary()
        tester._test()
    
    print("Encoder testing complete.")

def ekf(config, dataset_name):
    data_root = config['datasets']['dataset_root']
    airio_path = os.path.join(data_root, config['datasets']['airio'])
    airimu_path = os.path.join(data_root, config['datasets']['airimu'])
    devious_path = os.path.join(data_root, config['datasets']['devious'])
    airio_config = config['airio_settings']
    
    device = get_device(config)
    save_path = get_save_path(config)

    airio_state_load = load_pickle_results(airio_path)
    airimu_state_load = load_pickle_results(airimu_path)
    devious_state_load = pickle.load(open(devious_path, 'rb'))

    data_root = config['data-root']
    data_names = config['data_names']
    
    for name in data_names:
        seq_save_path = os.path.join(save_path, name)

        airimu_state = airimu_state_load[name]
        devious_state = devious_state_load[name]

        airimu_dataset = SeqInfDataset(
            data_root,
            name,
            airimu_state,
            device=device,
            name=config['name'],
            duration=1,
            step_size=1,
            conf=airio_config,
            drop_last=False
        )

        run_ekf(devious_state, airimu_dataset, name, airio_state_load, seq_save_path)

def main():
    parser = argparse.ArgumentParser(description='Egomotion estimation training and testing')
    subparsers = parser.add_subparsers(help='Choose to use the VO model or the VIO EKF', dest='model')

    model_parser = subparsers.add_parser('model', help='Train or test the VO model')
    model_parser.add_argument('model_type', choices=['encoder', 'recurrent'], help='Type of model to use')
    model_parser.add_argument('action', choices=['train', 'test', 'cache'], help='Action to perform')
    model_parser.add_argument('-d', '--dataset', choices=['m3ed', 'kitti'], default='m3ed', help='Dataset type to use (default: m3ed)')

    ekf_parser = subparsers.add_parser('ekf', help='Run the VIO EKF')
    ekf_parser.add_argument('-d', '--dataset', choices=['m3ed', 'kitti'], default='m3ed', help='Dataset type to use (default: m3ed)')

    args = parser.parse_args()
    
    # Determine config file
    if args.model == 'ekf':
        config_path = f'configs/{args.dataset}_ekf.json'
    else:
        config_path = f'configs/{args.dataset}_{args.model_type}.json'

    # Load configuration
    try:
        config = json.load(open(config_path))
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        print(f"Available configs: configs/encoder.json, configs/recurrent.json")
        return
    
    # Execute command
    if args.model == 'ekf':
        ekf(config, args.dataset)
        return
 
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
    
if __name__ == "__main__":
    main()