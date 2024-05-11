import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning import Trainer
from lightning.pytorch.loggers import CSVLogger

import config
from data_module import get_default_loaders
from models import CustomResnet, LightningResNet

seed_everything(99)



def train():
    logger = CSVLogger("logs", name="benchmark")
    train_dl, val_dl, test_dl = get_default_loaders()
    chkpt_callback = ModelCheckpoint(monitor="val_loss", save_weights_only=True)
    es_callback = es = EarlyStopping(monitor="val_loss", patience=4)
    model = LightningResNet()
    trainer = Trainer(
        max_epochs=config.epochs,
        # fast_dev_run=2,
        logger=logger,
        callbacks=[chkpt_callback, es_callback]
    )
    trainer.fit(model, train_dl, val_dl)
    trainer.test(model=model, dataloaders=test_dl)
    torch.save(model.model.state_dict(), "last_chkpt.pth")

if __name__ == "__main__":
    train()
