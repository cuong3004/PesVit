from ghostpes.config import *
from cvnets.models.classification.mobilevit import MobileViT
import torch 
import torch.nn as nn
from ghostpes.ghostnet import ghostnet, module_ghost_1, module_ghost_2


class Identity(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

## ghost 1
model_ghost = ghostnet()

def get_ghost_vit_1():
    

    model_ghost_git_1 = MobileViT(opts)

    model_ghost_git_1.layer_1 = model_ghost.blocks[0]

## ghost 2
# model_ghost = ghostnet()
def get_ghost_vit_2():
    model_ghost_git_2 = MobileViT(opts)

    model_ghost_git_2.layer_1 = model_ghost.blocks[0]
    model_ghost_git_2.layer_2[0] = model_ghost.blocks[1]

## ghost 3
# model_ghost = ghostnet()

def get_ghost_vit_3():
    model_ghost_git_3 = MobileViT(opts)

    model_ghost_git_3.layer_1 = model_ghost.blocks[0]
    model_ghost_git_3.layer_2[0] = model_ghost.blocks[1]
    model_ghost_git_3.layer_2[1] = model_ghost.blocks[2]
    model_ghost_git_3.layer_2[2] = model_ghost.blocks[3]


# model_ghost_git.layer_1 = nn.Sequential(
#         *list(model_ghost.blocks.children())[:-5],
#     )
# # model_ghost_git.layer_2 = nn.Conv2d(40, 48, 1)
# model_ghost_git.layer_2 = Identity()
# model_ghost_git.layer_3[0] = Identity()
# model_ghost_git.layer_4[0] = module_ghost_1
# model_ghost_git.layer_5[0] = module_ghost_2

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# # print(count_parameters(model_ghost_git))
# # print(model_ghost_git)

# # print(nn.Sequential(
# #         *list(model_ghost.blocks.children())[:-5],
# #     ))

# print(model_ghost_git(torch.ones((2,3,224,224))).shape)




