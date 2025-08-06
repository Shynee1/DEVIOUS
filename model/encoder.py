import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3_1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        conv_cache = {}
    
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        conv_cache['conv2'] = x
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_1(x)
        conv_cache['conv3_1'] = x
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_1(x)
        conv_cache['conv4_1'] = x
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv5_1(x)
        conv_cache['conv5_1'] = x
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)

        return x, conv_cache