import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm


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
            x_batch = x_data[i:to]
            y_batch = y_data[i:to]
            output = model(x_batch)
            loss += criterion(output, y_batch).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(y_batch.view_as(pred)).sum().item()
    return loss / len(x_data), correct / len(x_data)

# Training loop that computes the running loss per epoch and validation loss and accuracy per epoch
def train(model, x_train, y_train, x_val, y_val, optimizer, criterion, epochs=10, batch_size=64):

    first_epoch_running_loss = []

    train_evaluations = np.zeros((epochs, 2))
    val_evaluations = np.zeros((epochs, 2))



    for epoch in range(epochs):
        model.train()
        for i in tqdm(range(0, len(x_train), batch_size), desc=f'Batches for epoch {epoch + 1}/{epochs}'):

            to = min(i + batch_size, len(x_train))
            x_batch = x_train[i:to]
            y_batch = y_train[i:to]

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



def plot_results(first_epoch_running_loss, train_evaluations, val_evaluations):

    # Plot first epoch running loss
    plt.figure(figsize=(10, 6))
    plt.plot(first_epoch_running_loss, label="Single batch-averaged Loss")
    conv_size = 100
    ma_loss = np.convolve(first_epoch_running_loss, np.ones(conv_size), 'valid') / conv_size
    plt.plot(np.arange(len(ma_loss))+conv_size/2, ma_loss, label="Moving Average Loss")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_evaluations[:, 0], label='Training Loss')
    plt.plot(val_evaluations[:, 0], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')

    # adjust x labels to start from 1 and be integers
    plt.xticks(np.arange(0, len(train_evaluations) , 1), np.arange(1, len(train_evaluations) + 1, 1))
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_evaluations[:, 1], label='Training Accuracy')
    plt.plot(val_evaluations[:, 1], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xticks(np.arange(0, len(train_evaluations) , 1), np.arange(1, len(train_evaluations) + 1, 1))
    plt.legend()
    plt.show()


