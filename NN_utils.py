import numpy as np

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


# Calculate the total number of parameters
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params

def load_mnist_data():

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=len(train_data))
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    x_train_data, y_train_data = next(iter(train_loader))
    x_test_data, y_test_data = next(iter(test_loader))

    return (x_train_data, y_train_data), (x_test_data, y_test_data)

def load_mnist_data_with_augmentations():
    # Define augmentations for the training set
    train_transform = transforms.Compose([
        # transforms.RandomRotation(20),  # Randomly rotate images by ±10 degrees 
        # transforms.ToTensor(),  # Convert images to tensors
        # transforms.Normalize((0.5,), (0.5,))  # Normalize images to [-1, 1]
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # Rotation, translation, scaling
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),  # Perspective distortion
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),  # Add Gaussian blur
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]

    ])
    # Define transformations for the test set (no augmentations, just normalization)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=train_transform)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=len(test_data))
    x_train_data = []
    y_train_data = []
    for image,label in train_loader:
        x_train_data.append(image)
        y_train_data.append(label)
    x_train_data = torch.cat(x_train_data,dim=0)
    y_train_data = torch.cat(y_train_data,dim=0)
    print(x_train_data.shape)
    x_test_data, y_test_data = next(iter(test_loader))

    return (x_train_data, y_train_data), (x_test_data, y_test_data)
    #Split the training data into 50 000 training instances and 10 000 validation instances

def split_data(x_train_data, y_train_data):

    x_train_data, x_val_data = x_train_data[:50000], x_train_data[50000:]
    y_train_data, y_val_data = y_train_data[:50000], y_train_data[50000:]

    return (x_train_data, y_train_data), (x_val_data, y_val_data)


# Define the neural network
class MNISTConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # 1 input channel, 16 output channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 16 input channels, 32 output channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 32 input channels, 64 output channels
        self.fc1 = nn.Linear(64 * 3 * 3, 10)  # Flattened to a fully connected layer with 10 outputs

    def forward(self, x):
        x = F.relu(self.conv1(x))  # First convolution + ReLU
        x = F.max_pool2d(x, 2)     # Max pooling 2x2
        x = F.relu(self.conv2(x))  # Second convolution + ReLU
        x = F.max_pool2d(x, 2)     # Max pooling 2x2
        x = F.relu(self.conv3(x))  # Third convolution + ReLU
        x = F.max_pool2d(x, 2)     # Max pooling 2x2
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)            # Fully connected layer
        return x



def calculate_loss_and_accuracy(model, x_data, y_data, criterion, batch):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(x_data), batch):
            to = min(i + batch, len(x_data))
            x_batch = x_data[i:to].to("cuda")
            y_batch = y_data[i:to].to("cuda")
            output = model(x_batch)
            loss += criterion(output, y_batch).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_batch.view_as(pred)).sum().item()
    return loss / len(x_data), correct / len(x_data)

def confusion_matrix(model, x_data, y_data, batch):
    model.eval()
    confusion_matrix = np.zeros((10, 10))
    with torch.no_grad():
        for i in range(0, len(x_data), batch):
            to = min(i + batch, len(x_data))
            x_batch = x_data[i:to]
            y_batch = y_data[i:to]
            output = model(x_batch)
            pred = output.argmax(dim=1, keepdim=True)
            for j in range(len(pred)):
                confusion_matrix[y_batch[j]][pred[j]] += 1
    return confusion_matrix

# Training loop that computes the running loss per epoch and validation loss and accuracy per epoch
def train(model, x_train, y_train, x_val, y_val, optimizer, criterion, epochs=10, batch_size=64):

    first_epoch_running_loss = []

    train_evaluations = np.zeros((epochs, 2))
    val_evaluations = np.zeros((epochs, 2))


    for epoch in range(epochs):
        model.train()
        for i in tqdm(range(0, len(x_train), batch_size), desc=f'Batches for epoch {epoch + 1}/{epochs}'):

            x_batch = x_train[i:i + batch_size].to("cuda")
            y_batch = y_train[i:i + batch_size].to("cuda")

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            if epoch == 0:
                # get average loss and append to list
                first_epoch_running_loss.append(loss.item())

        
        train_loss, train_acc = calculate_loss_and_accuracy(model, x_train, y_train, criterion, batch_size)
        val_loss, val_acc = calculate_loss_and_accuracy(model, x_val, y_val, criterion, batch_size)

        train_evaluations[epoch] = [train_loss, train_acc]
        val_evaluations[epoch] = [val_loss, val_acc]

        print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

    return first_epoch_running_loss, train_evaluations, val_evaluations







