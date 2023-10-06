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


train_averages = torch.load('train_averages.pt')
train_labels = torch.load('train_labels.pt')
test_averages = torch.load('test_averages.pt')
test_labels = torch.load('test_labels.pt')

# Convert train and test labels to binary labels for parity: 0 = even, 1 = odd 
train_labels_parity = torch.where(train_labels % 2 == 0, torch.tensor(0), torch.tensor(1))
test_labels_parity = torch.where(test_labels % 2 == 0, torch.tensor(0), torch.tensor(1))

# Function to check if a number is prime
def is_prime(num):
    if num <= 1:
        return False
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Convert train and test labels to binary labels: 1 for prime, 0 for not prime
train_labels_prime = torch.tensor([1 if is_prime(label.item()) else 0 for label in train_labels])
test_labels_prime = torch.tensor([1 if is_prime(label.item()) else 0 for label in test_labels])

torch.save(train_labels_prime, 'train_labels_prime.pt')
torch.save(test_labels_prime, 'test_labels_prime.pt')
torch.save(train_labels_parity, 'train_labels_parity.pt')
torch.save(test_labels_parity, 'test_labels_parity.pt')




