import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class VariableInputNetwork(nn.Module):
    def __init__(self, num_classes=10, N=64, pooling_type='max'):
        """
        Args:
        - num_classes: Number of output classes (default=10 for MNIST).
        - N: Number of output channels in the final convolutional layer (default=64).
        - pooling_type: 'max' for global max pooling, 'avg' for global mean pooling.
        """
        super(VariableInputNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=N, kernel_size=3, stride=1, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Fixed 2x2 max pooling
        self.global_pool = nn.AdaptiveMaxPool2d((1, 1)) if pooling_type == 'max' else nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer
        self.fc = nn.Linear(N, num_classes)

    def forward(self, x):
        """
        Forward pass.
        Args:
        - x: Input tensor of shape (batch_size, 1, H, W).
        Returns:
        - logits: Output tensor of shape (batch_size, num_classes).
        """
        # Convolutional layers with ReLU activations and max pooling
        x = F.relu(self.conv1(x))  # Output: (batch, 16, H, W)
        x = self.pool(x)           # Output: (batch, 16, H/2, W/2)
        
        x = F.relu(self.conv2(x))  # Output: (batch, 32, H/2, W/2)
        x = self.pool(x)           # Output: (batch, 32, H/4, W/4)
        
        x = F.relu(self.conv3(x))  # Output: (batch, N, H/4, W/4)
        x = self.pool(x)           # Output: (batch, N, H/8, W/8)
        
        # Global pooling to reduce spatial dimensions to 1x1
        x = self.global_pool(x)    # Output: (batch, N, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch, N)
        
        # Fully connected layer
        logits = self.fc(x)        # Output: (batch, num_classes)
        return logits

# Calculate the total number of parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params
