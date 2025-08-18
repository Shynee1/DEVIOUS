import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class EncoderPipeline(nn.Module):
    def __init__(self):
        super(EncoderPipeline, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        encoded, conv_cache = self.encoder(x)
        decoded = self.decoder(encoded, conv_cache)
        
        return decoded