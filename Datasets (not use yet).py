import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

"""

shape_dir = ShapeNet dataset PATH ("D:/3DShapeNets")
annotations_folders = Dataset이 Label명의 폴더로 구분되어 있는 PATH ("D:/3DShapeNets/volumetric_data")
shape


"""


class CustomShapeDataset(Dataset):
    def __init__(self, annotations_folders, shape_dir, transform=None, target_transform=None):
        self.shape_labels = os.listdir(annotations_folders)
        self.shape_dir = shape_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.shape_labels)

    def __getitem__(self, idx):

        shape_path = os.path.join(self.shape_dir, self.shape_labels[idx],'30','train')
        shape = read_image(shape_path)
        label = self.shape_labels[idx]
        if self.transform:
            shape = self.transform(shape)
        if self.target_transform:
            label = self.target_transform(label)
        return shape, label