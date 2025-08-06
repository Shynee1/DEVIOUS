import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)  # 512 + 512 = 1024
        self.deconv3 = nn.ConvTranspose2d(1026, 256, kernel_size=4, stride=2, padding=1)  # 512 + 512 + 2 = 1026
        self.deconv2 = nn.ConvTranspose2d(514, 128, kernel_size=4, stride=2, padding=1)   # 256 + 256 + 2 = 514
    
        self.flow5 = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)   # 512 + 512 = 1024
        self.flow4 = nn.Conv2d(1026, 2, kernel_size=3, stride=1, padding=1)   # 512 + 512 + 2 = 1026
        self.flow3 = nn.Conv2d(514, 2, kernel_size=3, stride=1, padding=1)    # 256 + 256 + 2 = 514
        self.flow2 = nn.Conv2d(258, 2, kernel_size=3, stride=1, padding=1)    # 128 + 128 + 2 = 258

    def forward(self, x, conv_cache):
        deconv = self.deconv5(x)
        conv = conv_cache['conv5_1']
        
        concatenated = torch.cat([deconv, conv], dim=1)
        flow = self.flow5(concatenated)
        bilinear = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True)

        deconv = self.deconv4(concatenated)
        conv = conv_cache['conv4_1']
            
        concatenated = torch.cat([deconv, conv, bilinear], dim=1)
        flow = self.flow4(concatenated)
        bilinear = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True)
        
        deconv = self.deconv3(concatenated)
        conv = conv_cache['conv3_1']
            
        concatenated = torch.cat([deconv, conv, bilinear], dim=1)
        flow = self.flow3(concatenated)
        bilinear = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True)

        deconv = self.deconv2(concatenated)
        conv = conv_cache['conv2']
            
        concatenated = torch.cat([deconv, conv, bilinear], dim=1)
        flow = self.flow2(concatenated)
        bilinear = F.interpolate(flow, scale_factor=4, mode='bilinear', align_corners=True)

        return bilinear
