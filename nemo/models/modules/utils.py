import torch
import torch.nn as nn

class NopLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(NopLayer,self).__init__()
    def forward(self,X):
        return X
    

AVAILABLE_LOSSES = {nn.MSELoss:1, nn.BCEWithLogitsLoss:1, nn.BCELoss:1}

try:    
    from scmdc_zinb import ZINBLoss, NBLoss
    AVAILABLE_LOSSES[NBLoss] = 2
    AVAILABLE_LOSSES[ZINBLoss] = 3
except ImportError:
    pass

def select_output_dimensionality_multiplier(tgt_dist_loss):
        for LossClass in AVAILABLE_LOSSES.keys():
            if isinstance(tgt_dist_loss,LossClass):
                return AVAILABLE_LOSSES[LossClass]
        raise ValueError(f"tgt_dist_loss must be a valid distribution {tuple(AVAILABLE_LOSSES.keys())}, but was {type(tgt_dist_loss)}")