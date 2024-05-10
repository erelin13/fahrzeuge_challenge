import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam
import torchvision.models as models
from torchmetrics.regression import MeanAbsoluteError
import lightning as L


def fc_module(in_features, num_classes, dropout=0.4):
    hidden = int(in_features/2)
    fc = nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.Dropout(dropout),
        nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, num_classes))
    return fc
    

class CustomResnet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.network = models.resnet50(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = fc_module(num_ftrs, num_classes)
    
    def forward(self, xb):
        return F.sigmoid(self.network(xb))
    
    def freeze(self): # we might try to train only last layers
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True


class LightningResNet(L.LightningModule):
    def __init__(self, num_classes=2, loss_fn=nn.CrossEntropyLoss(), metric=MeanAbsoluteError()):
        super().__init__()
        self.model = CustomResnet(num_classes)
        self.loss_fn = loss_fn
        self.metric = metric

    def training_step(self, batch):
        images, targets = batch
        out = self.model(images)
        loss = self.loss_fn(out, targets)
        self.log("train_loss", loss, prog_bar=True)
        return loss.float()
        
    def validation_step(self, batch):
        images, targets = batch
        out = self.model(images)
        loss = self.loss_fn(out, targets)
        score = self.metric(out, targets)
        self.log("val_loss", loss)
        self.log("val_score", score, prog_bar=True)
        return {'val_loss': loss.detach().float(), 'val_score': score.detach() }

    def test_step(self, batch):
        images, targets = batch
        out = self.model(images)
        loss = self.loss_fn(out, targets)
        score = self.metric(out, targets)
        # self.log("val_loss", loss)
        self.log("test_score", score, prog_bar=True)
        return {'val_score': score.detach() }


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer