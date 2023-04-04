from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from ghostpes.dataset import dataloader_train_moco, dataloader_valid_moco
from ghostpes.config import *
from ghostpes.moco import MocoModel
import pytorch_lightning as pl
import torch
import argparse

parser  = argparse.ArgumentParser()
parser.add_argument('--path_resume', type=str, default=None)
parser.add_argument('--name', type=str)
parser.add_argument('--temp', type=float, default=0.1)
parser.add_argument('--learning_rate', type=float, default=0.0005)
parser.add_argument('--momentum', type=float, default=0.99)

args = parser.parse_args()
print("-"*10)
print(args)
print("-"*10)

lr_monitor = LearningRateMonitor(logging_interval='step')

# use a GPU if available
gpus = 1 if torch.cuda.is_available() else 0

if args.path_resume:

    import wandb

    checkpoint_reference = args.path_resume

    # download checkpoint locally (if not already cached)
    run = wandb.init()
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    path_checkpoint = artifact_dir + '/model.ckpt'
    wandb.finish()


    wandb_logger = WandbLogger(project="MocoSau_ver2", name=args.name, log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss_ssl", mode="min")
    # model = MocoModel()
    # path_checkpoint = "/content/epoch=38-step=34164.ckpt"
    model = MocoModel.load_from_checkpoint(path_checkpoint)
    trainer = pl.Trainer(max_epochs=max_epochs,
    #  gpus=gpus,
                        accelerator='gpu', devices=1,

                        resume_from_checkpoint=path_checkpoint,
                        #  limit_train_batches=20,
                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
    # , precision=16
    )

else:


    wandb_logger = WandbLogger(project="MocoSau_ver2", name=args.name, log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="train_loss_ssl", mode="min")
    model = MocoModel(args.temp, args.learning_rate, args.momentum)
    # path_checkpoint = "/content/epoch=38-step=34164.ckpt"
    # model = MocoModel.load_from_checkpoint(path_checkpoint)
    trainer = pl.Trainer(max_epochs=max_epochs,
                        #  gpus=gpus,
                        accelerator='gpu', devices=1,

                        logger=wandb_logger,
                        callbacks=[checkpoint_callback, lr_monitor],
    # , precision=16
    )



trainer.fit(
    model,
    dataloader_train_moco,
    dataloader_valid_moco,
)