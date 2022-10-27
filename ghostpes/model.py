from ghostpes.config import *
from cvnets.models.classification.mobilevit import MobileViT
import torch 
import torch.nn as nn
from ghostpes.ghostnet import ghostnet

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

model_ghost = ghostnet()

model_ghost_git = MobileViT(opts)

print(model_ghost_git(torch.ones((2,3,224,224))).shape)




