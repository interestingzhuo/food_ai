import timm
import torch.nn as nn
from pooling import *



class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = timm.create_model('deit_small_patch16_224', pretrained = True)
        n_features = self.model.head.in_features
        self.model.head = nn.Linear(n_features, 128)
        self.name = 'deit_small_patch16_224'
        self.pool = SPoC()

    def forward(self, x):
        x = self.model(x)

        return nn.functional.normalize(x, dim=-1)