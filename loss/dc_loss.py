import torch.nn as nn
import torch

def dcloss(phi1, phi2):
    # (N, 128)  -> (N, N)
    n = phi1.shape[0]
    cosine_sim = phi1@phi2.T
    summation = torch.sum(cosine_sim, dim=1) # N
    dc_loss = cosine_sim[range(n), range(n)]/summation
    dc_loss = dc_loss.mean()
    return dc_loss
    
    
    
class DCLoss(nn.Module):
    def __init__(self) :
        super().__init__()
    def forward(self, phi1, phi2):
        return dcloss(phi1, phi2)