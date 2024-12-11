import os
from glob import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# Custom Dataset to group images by resolution
class VariableResolutionDataset(Dataset):
    def __init__(self, root_dir, target_resolution=(28, 28), transform=None):
        self.root_dir = root_dir
        self.target_resolution = target_resolution
        self.image_paths = self._group_by_resolution()
        self.transform = transform or transforms.ToTensor()

    def _group_by_resolution(self):
        # Group images by resolution
        grouped_paths = {}
        all_images = glob(os.path.join(self.root_dir, "*", "*.png"))  # Change extension as needed
        for image_path in all_images:
            with Image.open(image_path) as img:
                resolution = img.size  # (width, height)
            if resolution not in grouped_paths:
                grouped_paths[resolution] = []
            grouped_paths[resolution].append(image_path)
        return grouped_paths

    def __len__(self):
        # Total number of images
        return sum(len(paths) for paths in self.image_paths.values())

    def __getitem__(self, idx):
        # Flatten grouped paths into a list for indexing
        all_paths = [path for paths in self.image_paths.values() for path in paths]
        image_path = all_paths[idx]

        # Load and resize image
        with Image.open(image_path) as img:
            img = img.convert("RGB")  # Ensure 3 channels
            resized_img = img.resize(self.target_resolution)

        # Apply transformations
        if self.transform:
            resized_img = self.transform(resized_img)

        # Extract label from folder name (assumes dataset is structured as root/class_name/image.png)
        label = os.path.basename(os.path.dirname(image_path))
        label = int(label)  # Convert to int if labels are numeric

        return resized_img, label

# Main Functionality
if __name__ == "__main__":
    root_dir = "path/to/mnist-varres"  # Update this to your dataset path
    target_resolution = (28, 28)  # Resize all images to this resolution

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # Normalize, modify based on dataset
    ])

    dataset = VariableResolutionDataset(root_dir, target_resolution, transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Inspect grouped paths
    grouped_paths = dataset._group_by_resolution()
    print(f"Grouped image resolutions: {grouped_paths.keys()}")

    # Loop through the dataloader
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: Images Shape {images.shape}, Labels {labels.shape}")
        if batch_idx == 2:  # Break after 3 batches
            break
