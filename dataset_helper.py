import random
import torch
import numpy as np

from PIL import Image, ImageOps
from torch.utils.data import Dataset

# user defined imports
from config import Config


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        if Config.run_type == "webcam":
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            # img0_tuple = self.imageFolderDataset.imgs[0]
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            # img1_tuple = self.imageFolderDataset.imgs[1]

        elif Config.run_type == "train" or Config.run_type == "test":
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            # we need to make sure approx 50% of images are in the same class
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                while True:
                    # keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if img0_tuple[1] == img1_tuple[1]:
                        break
            else:
                while True:
                    # keep looping till a different class image is found

                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                    if img0_tuple[1] != img1_tuple[1]:
                        break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class LFWDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=False):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):

        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class MnistDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        if Config.run_type == "webcam":
            img0_tuple = self.imageFolderDataset[0]
            img1_tuple = self.imageFolderDataset[1]

        elif Config.run_type == "train" or Config.run_type == "test":
            img0_tuple = random.choice(self.imageFolderDataset)
            # we need to make sure approx 50% of images are in the same class
            should_get_same_class = random.randint(0, 1)
            if should_get_same_class:
                while True:
                    # keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset)
                    if img0_tuple[1] == img1_tuple[1]:
                        break
            else:
                while True:
                    # keep looping till a different class image is found

                    img1_tuple = random.choice(self.imageFolderDataset)
                    if img0_tuple[1] != img1_tuple[1]:
                        break

        img0 = img0_tuple[0]
        img1 = img1_tuple[0]

        if self.should_invert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0,
            img1,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset)
