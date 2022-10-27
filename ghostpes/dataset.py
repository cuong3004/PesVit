import torch
import lightly
from ghostpes.config import *


collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=img_size
)

dataset_train_moco = lightly.data.LightlyDataset(
    input_dir=path_to_train,
)

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)