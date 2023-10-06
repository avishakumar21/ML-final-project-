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
from torchvision.io import read_image
import time 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt




#set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# defining paths of train, validation and test data
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root= 'train/', transform = transforms.Compose([transforms.Resize((102,102)),transforms.ToTensor()]))
    valid_dataset = datasets.ImageFolder(root= 'valid/', transform = transforms.Compose([transforms.Resize((102,102)),transforms.ToTensor()]))
    test_dataset = datasets.ImageFolder(root = 'test/', transform = transforms.Compose([transforms.Resize((102,102)),transforms.ToTensor()]))

    return train_dataset, valid_dataset, test_dataset



def random_patches(K, p, dataset):
    num_samples = len(dataset)
    patches = []

    for _ in range(K):
        index = np.random.randint(num_samples)
        image, _ = dataset[index]
        num_rows, num_cols = image.shape[1], image.shape[2]
        top = np.random.randint(num_rows - p + 1)
        left = np.random.randint(num_cols - p + 1)
        patch = image[:, top:top+p-1, left:left+p-1]
        patches.append(patch)

    return patches

# Load the satelite training validation, and test data
train_dataset, valid_dataset, test_dataset = load_data()

# Generate K patches of size p x p
# K = 2048
K_array = [1024, 2048, 4096, 8192]

K_time = []
for K in K_array:
    print(K)
    p = 3
    patches = random_patches(K, p, train_dataset)

    start_time = time.time()
    # Convert patches to a single tensor of shape (K, 1, p, p)
    patches = torch.stack(patches)
    # Create convolutional layer with randomly initialized weights
    conv_layer = nn.Conv2d(1, K, kernel_size=p, bias=False).to(device)
    conv_layer.weight.data = patches  # Set weights to the generated patches

    # Initialize lists to store averages and labels
    train_averages = []
    train_labels = []
    valid_averages = []
    valid_labels = []
    test_averages = []
    test_labels = []

    # Perform convolution, ReLU activation, and pooling for each image in the training set
    #i = 0 
    for image, label in train_dataset:
        #i = i + 1 
        #print('train image:')
        #print(i)
        image = image.unsqueeze(0)  # Add batch dimension
        conv_output = conv_layer(image)  # Perform convolution
        conv_output = F.relu(conv_output)  # Apply ReLU activation
        conv_output = F.max_pool2d(conv_output, kernel_size=2, stride=2)  # Max pooling
        conv_output = conv_output.mean(dim=(2,3))  # Average of convolution output
        train_averages.append(conv_output.squeeze().tolist())
        train_labels.append(label)  # Append label to labels list

    # Perform convolution, ReLU activation, and pooling for each image in the validation set
    #j = 0
    for image, label in valid_dataset:
        #j = j + 1 
        #print('valid image:')
        #print(j)
        image = image.unsqueeze(0)  # Add batch dimension
        conv_output = conv_layer(image)  # Perform convolution
        conv_output = F.relu(conv_output)  # Apply ReLU activation
        conv_output = F.max_pool2d(conv_output, kernel_size=2, stride=2)  # Max pooling
        conv_output = conv_output.mean(dim=(2,3))  # Average of convolution output
        valid_averages.append(conv_output.squeeze().tolist())
        valid_labels.append(label)  # Append label to labels list

    # Perform convolution, ReLU activation, and pooling for each image in the test set
    #l = 0 
    for image, label in test_dataset:
        #l = l + 1 
        #print('test image:')
        #print(l)
        image = image.unsqueeze(0)  # Add batch dimension
        conv_output = conv_layer(image)  # Perform convolution
        conv_output = F.relu(conv_output)  # Apply ReLU activation
        conv_output = conv_output.mean(dim=(2,3))  # Average of convolution output
        test_averages.append(conv_output.squeeze().tolist())
        test_labels.append(label)  # Append label to labels list

    end_time = time.time()
    time_features = end_time - start_time
    K_time.append(time_features)
    #print(time_features)
    # Convert averages to PyTorch tensors
    train_averages = torch.tensor(train_averages)
    test_averages = torch.tensor(test_averages)
    valid_averages = torch.tensor(valid_averages)

    #train, test, and validation labels for classic digits 0-9
    train_labels = torch.tensor(train_labels)
    test_labels = torch.tensor(test_labels)
    valid_labels = torch.tensor(valid_labels)

    #Save train, validation and test averages and labels
    file_name_TR_AV = 'train_averages_{}.pt'.format(K)  # construct the file name using string formatting
    torch.save(train_averages, file_name_TR_AV)
    file_name_TR_LA = 'train_labels_{}.pt'.format(K)  # construct the file name using string formatting
    torch.save(train_labels, file_name_TR_LA)
    file_name_TE_AV = 'test_averages_{}.pt'.format(K)
    torch.save(test_averages, file_name_TE_AV)
    file_name_TE_LA = 'test_labels_{}.pt'.format(K)
    torch.save(test_labels, file_name_TE_LA)
    file_name_VA_AV = 'valid_averages_{}.pt'.format(K)
    torch.save(valid_averages, file_name_VA_AV)
    file_name_VA_LA = 'valid_labels_{}.pt'.format(K)
    torch.save(valid_labels, file_name_VA_LA)
    file_name_patches = 'patches_{}.pt'.format(K)
    torch.save(patches, file_name_patches)


# Save the entire model
torch.save(conv_layer.state_dict(), 'featurization_model_weights.pth')

print(K_time)

plt.plot(K_array, K_time)
plt.xlabel('K')
plt.ylabel('Time (s)')
plt.show()

print('done')

