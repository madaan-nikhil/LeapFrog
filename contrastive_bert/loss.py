import torch.nn as nn
import torch

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss,self).__init__()
        
    def forward(self, energy, positive_idx):
        """
            energy is torch.tensor with torch.Size(num_sources)
            positive_idx is list with indices of energy corresponding to pos sources
        """
        num_positives = len(positive_idx)

        assert num_positives >= 0

        p_positive = float(1.0 / num_positives)

        log_q_sum = torch.log(energy[positive_idx[0]])

        for i in range(1,len(positive_idx)):
            log_q_sum += torch.log(energy[positive_idx[i]])

        return  -p_positive * log_q_sum