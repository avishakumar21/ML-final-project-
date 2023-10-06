import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import time 


def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

    return train_dataset, test_dataset

def random_patches(K, p, dataset):
    num_samples = len(dataset)
    patches = []

    for _ in range(K):
        index = np.random.randint(num_samples)
        image, _ = dataset[index]
        num_rows, num_cols = image.shape[1], image.shape[2]
        top = np.random.randint(num_rows - p + 1)
        left = np.random.randint(num_cols - p + 1)
        patch = image[:, top:top+p, left:left+p]
        patches.append(patch)

    return patches

# Load the MNIST training and test data
train_dataset, test_dataset = load_data()

# Generate K patches of size p x p
K = 6500
p = 2
patches = random_patches(K, p, train_dataset)

# Convert patches to a single tensor of shape (K, 1, p, p)
patches = torch.stack(patches)

# Create convolutional layer with randomly initialized weights
conv_layer = nn.Conv2d(1, K, kernel_size=p, bias=False)
conv_layer.weight.data = patches  # Set weights to the generated patches

# Initialize lists to store averages and labels
train_averages = []
train_labels = []
test_averages = []
test_labels = []

# Perform convolution and ReLU activation for each image in the training set
start_time = time.time()
for image, label in train_dataset:
    image = image.unsqueeze(0)  # Add batch dimension
    conv_output = conv_layer(image)  # Perform convolution
    conv_output = F.relu(conv_output)  # Apply ReLU activation
    conv_output = conv_output.mean(dim=(2,3))  # Average of convolution output
    train_averages.append(conv_output.squeeze().tolist())
    train_labels.append(label)  # Append label to labels list
end_time = time.time()

train_time = end_time - start_time

# Perform convolution and ReLU activation for each image in the test set
start_time = time.time()
for image, label in test_dataset:
    image = image.unsqueeze(0)  # Add batch dimension
    conv_output = conv_layer(image)  # Perform convolution
    conv_output = F.relu(conv_output)  # Apply ReLU activation
    conv_output = conv_output.mean(dim=(2,3))  # Average of convolution output
    test_averages.append(conv_output.squeeze().tolist())
    test_labels.append(label)  # Append label to labels list
end_time = time.time()

test_time = end_time - start_time

total_time = (train_time + test_time) * 1000  # convert to milliseconds
print("Total computation time: {:.2f} ms".format(total_time))

# Convert averages to PyTorch tensors
train_averages = torch.tensor(train_averages)
test_averages = torch.tensor(test_averages)

#train and test labels for classic digits 0-9
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

#Save train and test averages and labels
torch.save(train_averages, 'train_averages.pt')
torch.save(train_labels, 'train_labels.pt')
torch.save(test_averages, 'test_averages.pt')
torch.save(test_labels, 'test_labels.pt')
torch.save(patches, 'patches.pt')

# Save the entire model
torch.save(conv_layer.state_dict(), 'my_model_weights.pth')

# Load the saved model weights
#saved_state_dict = torch.load('my_model_weights.pth')
#conv_layer.load_state_dict(saved_state_dict)

'''
print("Train Averages shape:", train_averages.shape)
print("Train Labels shape:", train_labels.shape)
print("Test Averages shape:", test_averages.shape)
print("Test Labels shape:", test_labels.shape)
'''