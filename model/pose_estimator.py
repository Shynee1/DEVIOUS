import torch
from torch import nn
import torch.nn.functional as F

class PoseEstimator(nn.Module):
    def __init__(self, dropout_rate=0.5, batch_size=1, sequence_length=5):
        super(PoseEstimator, self).__init__()
        self.batch_size = batch_size
        self.sequence_length = sequence_length

        rnn_input = 10 * 3 * 1024
        self.rnn = nn.LSTM(input_size=rnn_input, 
                           hidden_size=1000, 
                           num_layers=2,
                           dropout=0.5,
                           batch_first=True)

        self.fc1 = nn.Linear(1000, 128)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, 6)

        self.fc1_cov = nn.Linear(1000, 128)
        self.dropout_cov = nn.Dropout(p=dropout_rate)
        self.fc2_cov = nn.Linear(128, 6)

    def reset_cache(self):
        torch.cuda.empty_cache()

    def forward(self, x):
        # Flatten frames for RNN processing 
        batch_size, seq_length, channels, height, width = x.shape
        encoded_sequence = x.view(batch_size, seq_length, -1)

        rnn_out, _ = self.rnn(encoded_sequence)
        last_output = rnn_out[:, -1, :]  # Use last timestep

        final_output = F.relu(self.fc1(last_output))
        final_output = self.dropout(final_output)
        final_output = self.fc2(final_output)

        final_output_cov = F.relu(self.fc1_cov(last_output))
        final_output_cov = self.dropout_cov(final_output_cov)
        final_output_cov = self.fc2_cov(final_output_cov)

        final_output_cov = torch.exp(final_output_cov - 5.0)

        return final_output, final_output_cov