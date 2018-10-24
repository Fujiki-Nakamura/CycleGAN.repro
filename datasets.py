# coding: UTF-8
from glob import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_dataloader(args, train=True):
    if train:
        transform_A = get_transform(args)
        transform_B = get_transform(args)
    else:
        transform_A, transform_B = None, None

    dataset_A = DomainDataset(args.data_A_dir, transform=transform_A)
    dataset_B = DomainDataset(args.data_B_dir, transform=transform_B)
    dataloader_A = DataLoader(
        dataset_A, batch_size=args.batch_size, shuffle=train,
        num_workers=args.num_workers)
    dataloader_B = DataLoader(
        dataset_B, batch_size=args.batch_size, shuffle=train,
        num_workers=args.num_workers)

    return dataloader_A, dataloader_B


def get_transform(args):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform


class DomainDataset(Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.impaths = glob(os.path.join(data_dir, '*.jpg'))
        self.transform = transform

    def __len__(self):
        return len(self.impaths)

    def __getitem__(self, idx):
        impath = self.impaths[idx]
        im = Image.open(impath)
        if self.transform is not None:
            im = self.transform(im)
        return im
