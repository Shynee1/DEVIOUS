import torch
import torch.nn as nn

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual, valid_mask):
        # Expand mask to match pred dimensions [B, C, H, W]
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(pred)
        pred = pred[valid_mask_expanded]
        actual = actual[valid_mask_expanded]
        return torch.sqrt(self.mse(torch.log(pred*pred + 1), torch.log(actual*actual + 1)))
    
class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual, valid_mask):
        # Expand mask to match pred dimensions [B, C, H, W]
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(pred)
        pred = pred[valid_mask_expanded]
        actual = actual[valid_mask_expanded]
        return torch.sqrt(self.mse(pred, actual))
    
class DiagLnCovLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, actual, cov):
        EPSILON = 1e-7
        error = (pred - actual).pow(2)
        l = (error / cov) + torch.log(cov + EPSILON)
        return l.mean()