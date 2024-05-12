import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms

import config
from data_module import valid_tfms
from models import CustomResnet, LightningResNet


def predict(model, image_path):
    img = Image.open(image_path).convert('RGB')
    img = valid_tfms(img)

    res = model(img.unsqueeze(0))
    res = res.detach().cpu().numpy()[0]
    return {
        "perspective_score_hood": res[0],
        "perspective_score_backdoor_left": res[1]
    }


def load_model(chkpt_path):
    if chkpt_path.endswith(".pth"):
        model = CustomResnet()
        ckpt = torch.load(chkpt_path, map_location="cpu")
        model.load_state_dict(ckpt)
    elif chkpt_path.endswith(".ckpt"):
        pl_model = LightningResNet.load_from_checkpoint(chkpt_path)
        model = pl_model.model
    model.eval()
    return model

def infer(img_path):
    model = load_model(config.model_checkpoint)
    res = predict(model, img_path)
    return res

if __name__ == "__main__":
    print(infer("CodingChallenge_v2/imgs/0a606358-bacc-4a71-909e-e28998bdceb4.jpg"))