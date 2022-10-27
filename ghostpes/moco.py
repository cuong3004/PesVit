import torch
import torch.nn as nn
from lightly.models.modules.heads import MoCoProjectionHead
from model import model_ghost_git
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
import pytorch_lightning as pl
import copy
import lightly  

from ghostpes.config import * 

class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        model = model_ghost_git
        # model.load_state_dict(torch.load("mobilevit_xxs.pt"))
        # model = mobilevit_xs()
        model.classifier.fc = nn.Linear(320, 512)
        self.backbone = model
        # self.backbone = nn.Sequential(
        #     *list(resnet.children())[:-1],
        #     nn.Linear(1280, 512),
        # )
        # mobilevit_s

        self.projection_head = MoCoProjectionHead(512, 512, 128)

        
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(self.backbone, self.backbone_momentum, 0.99)
        update_momentum(
            self.projection_head, self.projection_head_momentum, 0.99
        )

        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        loss = self.criterion(q, k)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.SGD(
        #     self.parameters(),
        #     lr=6e-3,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        optim = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0),
            'name': 'my_logging_name',
        }
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, max_epochs
        # )
        # return [optim], [lr_scheduler]
        return optim