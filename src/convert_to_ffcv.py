import os
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField
from torchvision.datasets import ImageFolder

# Path to ImageNet directories
train_path = '/Users/chaitanyasagargurujula/Documents/imagenet-mini/train'
val_path = '/Users/chaitanyasagargurujula/Documents/imagenet-mini/val'

# Paths for .beton files
train_beton = '/Users/chaitanyasagargurujula/Documents/imagenet-mini/train.beton'
val_beton = '/Users/chaitanyasagargurujula/Documents/imagenet-mini/val.beton'

# Conversion function
def convert_to_beton(image_dir, beton_file):
    dataset = ImageFolder(image_dir)
    writer = DatasetWriter(
        beton_file,
        {
            'image': RGBImageField(max_resolution=256),
            'label': IntField()
        }
    )
    writer.from_indexed_dataset(dataset)

if __name__ == '__main__':
   # Convert datasets
   convert_to_beton(train_path, train_beton)
   convert_to_beton(val_path, val_beton)
