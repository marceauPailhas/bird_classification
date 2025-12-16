import zipfile
import os

import torchvision.transforms as transforms
from torchvision.transforms.v2 import ElasticTransform

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
def ensure_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

data_transforms = transforms.Compose([
    transforms.Lambda(ensure_rgb),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
])

data_transforms_training = transforms.Compose([ 
    transforms.Lambda(ensure_rgb),
    transforms.RandomResizedCrop(
        size=224,
        scale=(0.85, 1.),      # random zoom-in (crop)
        ratio=(0.9, 1.1)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    # ElasticTransform(alpha=20.0, sigma=4.0),
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    #transforms.RandomPosterize(bits=4, p=0.4),
    transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), shear= 5),
    transforms.Pad(padding=10, padding_mode='edge'),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
    #transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),

])

def stream_generator(dataset, transform):
    for sample in dataset:
        image = sample["image"] 
        image = image.convert("RGB") # PIL Image object
        label = sample["label"]

        # Apply transforms
        img_t = transform(image)
        yield img_t, label

from torch.utils.data import IterableDataset

class HFStreamDataset(IterableDataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __iter__(self):
        return stream_generator(self.dataset, self.transform)
    
import torch
import numpy as np

def mixup_data(x, y, alpha=0.2):
    '''Retourne les données mixées et les labels mixés.'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
