import torch
import torch.nn as nn
from lightly.models.modules.heads import MoCoProjectionHead
from ghostpes.model import get_ghost_vit_2
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.utils import batch_shuffle
from lightly.models.utils import batch_unshuffle
import pytorch_lightning as pl
import copy
import lightly  

from ghostpes.config import * 

class IdentityLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, q, k):
        return q, k

def accuracy(output, target, topk=(1,5)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # print(correct[:k].shape)
        # print(correct[:k].view(-1).shape)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class MyNCELoss(lightly.loss.NTXentLoss):
    def __init__(self, temperature: float = 0.5, memory_bank_size: int = 0, gather_distributed: bool = False):
        super().__init__(temperature, memory_bank_size, gather_distributed)
        self.cross_entropy = IdentityLoss()
    def forward(self, out0: torch.Tensor, out1: torch.Tensor):
        return super().forward(out0, out1)

class MocoModel(pl.LightningModule):
    def __init__(self, temp=0.1, learning_rate=0.0005, momentum=0.99):
        super().__init__()

        self.save_hyperparameters()

        model = get_ghost_vit_2()
        # model.load_state_dict(torch.load("mobilevit_xxs.pt"))
        # model = mobilevit_xs()
        model.classifier.fc = nn.Linear(384, 512)
        self.backbone = model
        self.crossEntropy = nn.CrossEntropyLoss(reduction="mean")
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
        self.criterion = MyNCELoss(
            temperature=self.hparams.temp,
            memory_bank_size=memory_bank_size)

    def training_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        # update momentum
        update_momentum(
            self.backbone, self.backbone_momentum, self.hparams.momentum)
        update_momentum(
            self.projection_head, self.projection_head_momentum, self.hparams.momentum
        )

        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        logit, label = self.criterion(q, k)
        loss = self.crossEntropy(logit, label)
        
        # result = accuracy(logit, loss)
        # result = accuracy(logit, label)

        self.log("train_loss_ssl", loss,  on_step=False, on_epoch=True, batch_size=batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        (x_q, x_k), _, _ = batch

        q = self.backbone(x_q).flatten(start_dim=1)
        q = self.projection_head(q)

        k, shuffle = batch_shuffle(x_k)
        k = self.backbone_momentum(k).flatten(start_dim=1)
        k = self.projection_head_momentum(k)
        k = batch_unshuffle(k, shuffle)

        logit, label = self.criterion(q, k)
        loss = self.crossEntropy(logit, label)

        result = accuracy(logit, label)

        self.log("valid_loss_ssl", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("valid_acc_top1_ssl", result[0], on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("valid_acc_top5_ssl", result[1], on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def configure_optimizers(self):
        # optim = torch.optim.SGD(
        #     self.parameters(),
        #     lr=6e-3,
        #     momentum=0.9,
        #     weight_decay=5e-4,
        # )
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # lr_scheduler = {
        #     'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=10, eta_min=0),
        #     'name': 'my_logging_name',
        # }
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optim, max_epochs
        # )
        # return [optim], [lr_scheduler]
        return optim