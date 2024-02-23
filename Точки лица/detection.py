import os
import cv2
import torch
from torch.utils.data import DataLoader, Sampler, Dataset


def img_to_array(path):
    cv2.imread(path, cv2.IMREAD_COLOR)

class FacePointDataset(Dataset):
    def __init__(self, img_dir: str, coords_file: str, transforms):
        self.img_paths = list(os.listdir(img_dir))
        self.transforms = transforms

    def __getitem__(self, index):
