from ghostpes.config import *
from cvnets.models.classification.mobilevit import MobileViT
import torch 
import torch.nn as nn
from ghostpes.ghostnet import ghostnet, module_ghost_1, module_ghost_2
from fvcore.nn import FlopCountAnalysis
import os

class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

## ghost 1
model_ghost = ghostnet()

def get_mobile_vit(pretrained = False):
    model =  MobileViT(opts)
    if pretrained:
        model.load_state_dict(torch.load("mobilevit_xs.pt"))
    return model

def get_ghost_vit_1(pretrained = False):
    model_ghost_git_1 = get_mobile_vit(pretrained)

    model_ghost_git_1.layer_1 = model_ghost.blocks[0]
    return model_ghost_git_1

## ghost 2
# model_ghost = ghostnet()
def get_ghost_vit_2(pretrained = False):
    model_ghost_git_2 = get_mobile_vit(pretrained)

    model_ghost_git_2.layer_1 = model_ghost.blocks[0]
    model_ghost_git_2.layer_2[0] = model_ghost.blocks[1]
    return model_ghost_git_2

## ghost 3
# model_ghost = ghostnet()

def get_ghost_vit_3(pretrained = False):
    model_ghost_git_3 = get_mobile_vit(pretrained)

    model_ghost_git_3.layer_1 = model_ghost.blocks[0]
    model_ghost_git_3.layer_2[0] = model_ghost.blocks[1]
    model_ghost_git_3.layer_2[1] = model_ghost.blocks[2]
    model_ghost_git_3.layer_2[2] = model_ghost.blocks[3]
    return model_ghost_git_3


model_vit = get_ghost_vit_2()

layer_3_vit = model_vit.layer_3[1]


print(model_vit)
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('cuong31120/MocoSau_fine_tune_12_final/model-2mi062rm:v3', type='model')
# artifact_dir = artifact.download()
# print(artifact_dir + "/model.ckpt")
# wandb.finish()

path_checkpoint = "artifacts/model-2mi062rm-v3/model.ckpt"
state_dict = torch.load(path_checkpoint,  map_location=torch.device('cpu'))

state_dict = state_dict["state_dict"]
new_state_dict = {}
for k, v in state_dict.items():
    k_new = k[len("model."):]
    if k[:len("model.layer_2.0.0")] == "model.layer_2.0.0":
        k_new = "layer_2.0" + k_new[len("layer_2.0.0"):] 
    print(k_new)
    new_state_dict.update({k_new:v})

is_ok = model_vit.load_state_dict(new_state_dict)
print(state_dict)

