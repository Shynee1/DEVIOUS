import torch
import torch.nn as nn
import tqdm
import os
import json

from utils.loss_functions import RMSELoss
from test import TestEncoder, TestRecurrent

class Train:
    def __init__(self, model, config, train_loader, test_loader, save_path):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        if not config['cuda'] or not torch.cuda.is_available():
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:' + str(config['gpu']))

        self.model.to(self.device)

        self.save_path = save_path

        train_config = config['train']
        self.epochs = train_config['epochs']
        self.batch_size = train_config['batch_size']
        learning_rate = train_config['learning_rate']
        gamma = train_config['gamma']
        
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Multiply learning rate every 20 epochs by a constant
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=gamma)

    def summary(self):
        print("==================================== TRAIN SUMMARY ====================================")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Save Path: {self.save_path}")
        print(f"Test Set: {self.test_loader.__class__.__name__}")
        print(f"Dataset Length: {len(self.train_loader.dataset)}")
        print(f"Batch Size: {self.train_loader.batch_size}")
        print("======================================================================================")

    def _train(self):
        raise NotImplementedError
    
    def save_model(self, epoch, train_config, is_best=False, test_losses=None, train_losses=None):
        # Ensure save directory exists
        os.makedirs(self.save_path, exist_ok=True)

        # Save the losses
        if test_losses is not None and train_losses is not None:
            losses_path = os.path.join(self.save_path, 'losses.json')
            with open(losses_path, 'w') as f:
                json.dump({'train_losses': train_losses, 'test_losses': test_losses}, f)

        if epoch % train_config.get('save_interval', 1) == 0:
            save_path = os.path.join(self.save_path, f"model_{epoch}.ckpt")
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        if is_best:
            best_path = os.path.join(self.save_path, "best_model.ckpt")
            torch.save(self.model.state_dict(), best_path)
            print(f"Best model saved to {best_path}")

        newest_path = os.path.join(self.save_path, "newest_model.ckpt")
        torch.save(self.model.state_dict(), newest_path)
        print(f"Newest model saved to {newest_path}")
    
class TrainEncoder(Train):
    def __init__(self, model, config, train_loader, test_loader, save_path):
        super(TrainEncoder, self).__init__(model, config, train_loader, test_loader, save_path)
        self.loss_function = RMSELoss()
        self.tester = TestEncoder(model, config, test_loader)
    
    def _train(self):
        self.model.train()

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        save_best = False
        for epoch in range(self.epochs):

            epoch_loss = 0.0
            t_range = tqdm.tqdm(self.train_loader)

            for batch_idx, batch in enumerate(t_range):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                flow = batch['flow']
                valid_mask = batch['valid_mask'].bool()

                # Forward pass
                outputs = self.model(flow)

                # Compute loss
                loss = self.loss_function(outputs, flow, valid_mask)

                epoch_loss += loss.item()
                
                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.lr_scheduler.step()
            train_losses.append(epoch_loss / len(self.train_loader))

            test_loss_epoch = self.tester._test()
            test_losses.append(test_loss_epoch)

            if test_loss_epoch < best_loss:
                best_loss = test_loss_epoch
                save_best = True
            else:
                save_best = False

            self.save_model(epoch, self.config['train'], is_best=save_best)

            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss / len(self.train_loader)}")

        return train_losses, test_losses


class TrainRecurrent(Train):
    def __init__(self, recurrent_model, encoding_model, config, train_loader, test_loader, save_path):
        super(TrainRecurrent, self).__init__(recurrent_model, config, train_loader, test_loader, save_path)
        self.loss_function = nn.MSELoss()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoding_model = encoding_model
        self.tester = TestRecurrent(recurrent_model, encoding_model, config, test_loader)
        self.sequence_length = config['train']['sequence_length']

    def encode_flows(self, flow_sequence):
        batch_size, seq_len = flow_sequence.shape[:2]
        
        # Pre-allocate the output tensor with gradients enabled
        encoded_flows = torch.zeros(
            (batch_size, seq_len, 1024, 5, 10),
            device=flow_sequence.device,
            dtype=flow_sequence.dtype,
            requires_grad=True
        )
        
        # Process each frame without computing gradients for the encoder
        with torch.no_grad():
            for i in range(seq_len):
                flow = flow_sequence[:, i, :, :, :]
                encoded, _ = self.encoding_model(flow)
                pooled = self.pooling(encoded)
                # Copy the data without gradients
                encoded_flows.data[:, i] = pooled
                
                # Clear intermediate tensors
                del encoded, pooled
        
        return encoded_flows
    
    def _train(self):
        self.model.train()
        self.encoding_model.eval()

        train_losses = []
        test_losses = []
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            print(f"Starting epoch {epoch+1}/{self.epochs}")
            epoch_loss = 0.0
            t_range = tqdm.tqdm(self.train_loader)
            self.model.train()
    
            for batch_idx, batch in enumerate(t_range):
               
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                flow = batch['flow_sequence']                       # (batch, seq_length, C, H, W)
                ground_truth = batch['gt_transform']                # (batch, 6)
                
                if 'encodings' in batch:
                    encodings = batch['encodings']                  # (batch, seq_length, 1024, 5, 10)
                else:
                    encodings = self.encode_flows(flow)             # (batch, seq_length, 1024, 5, 10)

                # Forward pass
                outputs = self.model(encodings)                     # (batch, 6)

                # Compute loss
                loss = self.loss_function(outputs, ground_truth)
                epoch_loss += loss.item()

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update progress bar
                t_range.set_description(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.6f}")
                
                # Clear intermediate tensors to prevent memory accumulation
                del flow, ground_truth, encodings, outputs, loss
                

            self.lr_scheduler.step()
            train_losses.append(epoch_loss / len(self.train_loader))

            # Evaluate after each epoch
            test_loss_epoch = self.tester._test()
            test_losses.append(test_loss_epoch)

            if test_loss_epoch < best_loss:
                best_loss = test_loss_epoch
                save_best = True
            else:
                save_best = False

            self.save_model(epoch, self.config['train'], is_best=save_best, 
                            test_losses=test_losses, train_losses=train_losses)
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss / len(self.train_loader):.6f}")

        self.tester._test()

        return train_losses, test_losses