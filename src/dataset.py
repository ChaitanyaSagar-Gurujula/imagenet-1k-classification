import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet18
from ffcv.loader import Loader,OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import ToTensor, ToDevice, ToTorchImage, NormalizeImage, Squeeze
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder, CenterCropRGBImageDecoder
from tqdm import tqdm
import numpy as np

# Dataset paths
#train_beton = '/content/imagenet-mini/val.beton'
#val_beton = '/content/imagenet-mini/val.beton'
val_beton = '/Users/chaitanyasagargurujula/Documents/imagenet-mini/val.beton'

# Hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# Define data loaders
def create_loader(batch_size, beton_path, is_train):
    if is_train:
        # For training, use RandomResizedCropRGBImageDecoder
        decoders = [RandomResizedCropRGBImageDecoder(output_size =(224, 224),scale=(0.08, 1.0), ratio=(3./4., 4./3.))]
    else:
        # For validation, use CenterCropRGBImageDecoder
        decoders = [CenterCropRGBImageDecoder(output_size =(224, 224), ratio = 224/256)]
    
    # Define transforms
    transforms = decoders + [ToTensor(), ToDevice(device), ToTorchImage() , NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float32)]  # Combine decoder with ToTensor/ToDevice
        
    # Set the order based on training or validation
    order = OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL
    
    return Loader(
        beton_path,
        batch_size=batch_size,
        num_workers=1,
        order=order,  # Correct usage of TraversalOrder
        drop_last=is_train,
        pipelines={'image': transforms, 'label': [ToTensor(), Squeeze(), ToDevice(device)]},
        seed=42
    )

def get_imagenet_loaders(batch_size=128):
    """Create Imagenet 1k train and test data loaders using FFCV Loaders"""
    
    train_loader = create_loader(batch_size=batch_size, beton_path=val_beton, is_train=True)
    val_loader = create_loader(batch_size=batch_size, beton_path=val_beton, is_train=False)

    return train_loader, val_loader
