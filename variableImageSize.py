import os
from glob import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, num_classes=10, N=81, pooling_type='max'):
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


# Define a training function for one resolution
def train_on_resolution(model, data, labels, criterion, optimizer, device, batch_size=64):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Shuffle the data
    perm = torch.randperm(data.size(0))
    data, labels = data[perm], labels[perm]

    
    for i in range(0, data.size(0), batch_size):
        inputs = data[i:i+batch_size].to(device)
        targets = labels[i:i+batch_size].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Evaluation function
def evaluate_on_resolution(model, data, labels, criterion, device, batch_size=64):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        
        for i in range(0, data.size(0), batch_size):
            inputs = data[i:i+batch_size].to(device)
            targets = labels[i:i+batch_size].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def confusion_matrix(model, data, labels, device, batch_size=64):
    model.eval()
    confusion_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            inputs = data[i:i+batch_size].to(device)
            targets = labels[i:i+batch_size].to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    return confusion_matrix



# Training loop for all resolutions
def train_on_all_resolutions(tensors_by_resolution, model, criterion, optimizer, device, epochs=5):
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for res, (data, labels) in tensors_by_resolution.items():
            print(f"Training on resolution {res}x{res}:")
            train_loss, train_acc = train_on_resolution(model, data, labels, criterion, optimizer, device)
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print("-" * 50)

# Training loop for all resolutions with pooling comparison
def train_and_compare_pooling(train_data_by_resolution, test_data_by_resolution, num_epochs=5):
    pooling_methods = ['max', 'avg']
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for pooling_type in pooling_methods:
        print(f"\nTraining with {pooling_type.upper()} pooling:")
        
        # Initialize model, criterion, and optimizer
        model = VariableInputNetwork(pooling_type=pooling_type).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for res, (train_data, train_labels) in train_data_by_resolution.items():
                print(f"Training on resolution {res}x{res}:")
                train_loss, train_acc = train_on_resolution(model, train_data, train_labels, criterion, optimizer, device)
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print("-" * 50)
        
        # Evaluation
        print(f"\nEvaluating {pooling_type.upper()} pooling on test data:")
        pooling_results = {}
        for res, (test_data, test_labels) in test_data_by_resolution.items():
            test_loss, test_acc = evaluate_on_resolution(model, test_data, test_labels, criterion, device)
            print(f"Resolution {res}x{res}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            pooling_results[res] = (test_loss, test_acc)
        
        results[pooling_type] = pooling_results

    # Print Summary
    print("\nComparison of Pooling Methods:")
    for pooling_type, pooling_results in results.items():
        print(f"\n{pooling_type.upper()} Pooling:")
        for res, (loss, acc) in pooling_results.items():
            print(f"Resolution {res}x{res}: Test Loss = {loss:.4f}, Test Acc = {acc:.2f}%")
    
    return results

def train_max_pooling(train_data_by_resolution, test_data_by_resolution, num_epochs=5, batch_size=64):
    pooling_methods = ['max']
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    confusion_matrix_results = {}
    for pooling_type in pooling_methods:
        print(f"\nTraining with {pooling_type.upper()} pooling:")
        
        # Initialize model, criterion, and optimizer
        model = VariableInputNetwork(pooling_type=pooling_type,N=81).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0009)
        
        # Training
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            for res, (train_data, train_labels) in train_data_by_resolution.items():
                print(f"Training on resolution {res}x{res}:")
                train_loss, train_acc = train_on_resolution(model, train_data, train_labels, criterion, optimizer, device,batch_size)
                print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print("-" * 50)
        
        # Evaluation
        print(f"\nEvaluating {pooling_type.upper()} pooling on test data:")
        pooling_results = {}

        
        for res, (test_data, test_labels) in test_data_by_resolution.items():
            test_loss, test_acc = evaluate_on_resolution(model, test_data, test_labels, criterion, device, batch_size)
            print(f"Resolution {res}x{res}: Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            pooling_results[res] = (test_loss, test_acc)

            # Confusion Matrix
            confusion_matrix_results[res] = confusion_matrix(model, test_data, test_labels, device, batch_size)

        



        
        results[pooling_type] = pooling_results

    # Print Summary
    print("\nComparison of Pooling Methods:")
    for pooling_type, pooling_results in results.items():
        print(f"\n{pooling_type.upper()} Pooling:")
        for res, (loss, acc) in pooling_results.items():
            print(f"Resolution {res}x{res}: Test Loss = {loss:.4f}, Test Acc = {acc:.2f}%")

    #aggregate confusion matrix
    confusion_matrix_aggregate = torch.zeros(10, 10)

    for res, cm in confusion_matrix_results.items():
        confusion_matrix_aggregate += cm

    
    return results, confusion_matrix_aggregate, confusion_matrix_results

# Run the training and comparison
