import os
from glob import glob
from PIL import Image
import torch
from torchvision import transforms

# Function to load and group images by resolution
def load_images_by_resolution(root_dir):
    # Initialize containers for each resolution
    images_32 = []
    images_48 = []
    images_64 = []
    labels_32 = []
    labels_48 = []
    labels_64 = []

    # Define transforms (convert to tensor)
    transform = transforms.ToTensor()

    # Iterate over the dataset
    for class_dir in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_dir)
        if os.path.isdir(class_path):
            for img_file in glob(os.path.join(class_path, "*.png")):  # Adjust extension if needed
                with Image.open(img_file) as img:
                    img_tensor = transform(img)
                    label = int(class_dir)  # Assuming class folders are named numerically
                    
                    # Group images by resolution
                    if img.size == (32, 32):
                        images_32.append(img_tensor)
                        labels_32.append(label)
                    elif img.size == (48, 48):
                        images_48.append(img_tensor)
                        labels_48.append(label)
                    elif img.size == (64, 64):
                        images_64.append(img_tensor)
                        labels_64.append(label)

    # Stack images and labels into tensors
    tensor_32 = torch.stack(images_32), torch.tensor(labels_32)
    tensor_48 = torch.stack(images_48), torch.tensor(labels_48)
    tensor_64 = torch.stack(images_64), torch.tensor(labels_64)

    return tensor_32, tensor_48, tensor_64


