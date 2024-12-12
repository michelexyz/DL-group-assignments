import matplotlib.pyplot as plt
import numpy as np 

def plot_results(first_epoch_running_loss, train_evaluations, val_evaluations):

    ## Plot first epoch running loss ##
    plt.figure(figsize=(10, 6))
    plt.plot(first_epoch_running_loss, label="Single batch-averaged Loss")
    conv_size = 100
    ma_loss = np.convolve(first_epoch_running_loss, np.ones(conv_size), 'valid') / conv_size
    plt.plot(np.arange(len(ma_loss))+conv_size/2, ma_loss, label="Moving Average Loss", color='orangered')
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.tick_params(axis='y', labelcolor='tab:red')
    plt.legend()
    plt.show()

    ## Plot training and validation los ##
    plt.figure(figsize=(10, 6))
    plt.plot(train_evaluations[:, 0], label='Training Loss', color='lightcoral')
    plt.plot(val_evaluations[:, 0], label='Validation Loss', color='orangered')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')

    # adjust x labels to start from 1 and be integers
    plt.xticks(np.arange(0, len(train_evaluations) , 1), np.arange(1, len(train_evaluations) + 1, 1))
    plt.tick_params(axis='y', labelcolor='tab:red')
    plt.legend()
    plt.show()

    ## Plot training and validation accuracy ##
    plt.figure(figsize=(10, 6))
    plt.plot(train_evaluations[:, 1], label='Training Accuracy', color='lightblue')
    plt.plot(val_evaluations[:, 1], label='Validation Accuracy', color='dodgerblue')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xticks(np.arange(0, len(train_evaluations) , 1), np.arange(1, len(train_evaluations) + 1, 1))

    plt.tick_params(axis='y', labelcolor='tab:blue')
    plt.legend()
    plt.show()


def plot_param_comparison(results, parameter_name):
    plt.figure(figsize=(10, 6))

    for batch_size, (first_epoch_running_loss, train_evaluations, val_evaluations) in results.items():
        plt.plot(val_evaluations[:, 0], label=f'{parameter_name}: {batch_size}')

    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, len(val_evaluations) , 1), np.arange(1, len(val_evaluations) + 1, 1))
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

    
    plt.figure(figsize=(10, 6))

    for batch_size, (first_epoch_running_loss, train_evaluations, val_evaluations) in results.items():
    
        plt.plot(val_evaluations[:, 1], label=f'{parameter_name}: {batch_size}')

    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, len(val_evaluations) , 1), np.arange(1, len(val_evaluations) + 1, 1))
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()  
    plt.show()