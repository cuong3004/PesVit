import torch
import lightly
from ghostpes.config import *


collate_fn_train = lightly.data.SimCLRCollateFunction(
    input_size=img_size
)

collate_fn_valid = lightly.data.SimCLRCollateFunction(
    input_size=img_size
)

dataset_full_moco = lightly.data.LightlyDataset(
    input_dir=path_to_train,
)
len_train = int(len(dataset_full_moco) * 0.8)
len_valid = len(dataset_full_moco) - len_train
train_set, val_set = torch.utils.data.random_split(dataset_full_moco, [len_train, len_valid], torch.Generator().manual_seed(42))

dataloader_train_moco = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn_train,
    drop_last=True,
    num_workers=num_workers
)

dataloader_valid_moco = torch.utils.data.DataLoader(
    val_set,
    batch_size=batch_size,
    # shuffle=True,
    collate_fn=collate_fn_valid,
    drop_last=True,
    num_workers=num_workers
)