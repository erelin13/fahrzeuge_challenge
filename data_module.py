import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from PIL import Image
from torchvision import transforms

import config


train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-90, 90)),
    transforms.ToTensor()])

valid_tfms = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


class CarsDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        class1_name="perspective_score_hood",
        class2_name="perspective_score_backdoor_left",
        transform=None):

        self.df = df
        self.df.reset_index(inplace=True)
        self.transform = transform
        self.img_dir = img_dir
        self.class1_name = class1_name
        self.class2_name = class2_name
        
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "filename"]
        img_path = os.path.join(self.img_dir, img_path)
        hood_score = self.df.loc[idx, self.class1_name]
        bdoor_score = self.df.loc[idx, self.class2_name]
        labels = torch.tensor([hood_score, bdoor_score])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img.float(), labels.float()

    def set_transforms(self, tfs):
        self.transforms = tfs


def get_default_loaders(device="cpu"):
    df = pd.read_csv(config.csv_path)
    dataset = CarsDataset(df, config.images_dir)
    train_set, val_set, test_set = random_split(
        df.index,
        [config.train_chunk, config.valid_chunk, config.test_chunk]
    )

    train_set = CarsDataset(
        df.loc[list(train_set)],
        config.images_dir,
        transform=train_tfms
    )
    val_set = CarsDataset(
        df.loc[list(val_set)],
        config.images_dir,
        transform=valid_tfms
    )
    test_set = CarsDataset(
        df.loc[list(test_set)],
        config.images_dir,
        transform=valid_tfms
    )

    train_set.set_transforms(train_tfms)
    val_set.set_transforms(valid_tfms)
    test_set.set_transforms(valid_tfms)

    train_loader = DataLoader(
        train_set, 
        config.batch_size,
        shuffle=True, 
        num_workers=4)
    val_loader = DataLoader(
        val_set, 
        1,
        shuffle=True)
    test_loader = DataLoader(
        test_set, 
        1,
        shuffle=True)

    return (
        DeviceDataLoader(train_loader, device),
        DeviceDataLoader(val_loader, device),
        DeviceDataLoader(test_loader, device),
    )