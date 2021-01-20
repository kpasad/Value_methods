import torch
import torch.nn as nn
import torch.nn.functional as F

class ddqn(nn.Module):
    def __init__(self, state_size, action_size,seed):
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.shared_feats= nn.Sequential(
            nn.Linear(state_size,128),
            nn.ReLU()
        )

        self.value_nw = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  #State value is the single value
        )

        self.advnt_nw = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )


    def forward(self,state):
        x=self.shared_feats(state)
        val = self.value_nw(x)
        advt= self.advnt_nw(x)
        return val + advt  - advt.mean()

