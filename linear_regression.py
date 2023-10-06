import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import time
import sklearn 
from sklearn.metrics import roc_curve, auc, confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


# Load train and test averages and labels
train_averages = torch.load('train_averages_4096.pt')
train_labels = torch.load('train_labels_4096.pt')
test_averages = torch.load('test_averages_4096.pt')
test_labels = torch.load('test_labels_4096.pt')
patches = torch.load('patches_4096.pt')


# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return x

# Define the input size and output size for the linear regression model
input_size = len(patches)  # Number of patches
output_size = 2  # Number of classes 

# Instantiate the linear regression model
model = LinearRegression(input_size, output_size)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
#lr_array = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
#for lr in lr_array:
#    print('learning rate')
#    print(lr)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Convert train and test averages to float tensors
train_averages = train_averages.float()
test_averages = test_averages.float()

# Flatten the train and test averages for input to the linear regression model
train_averages_flattened = train_averages.view(train_averages.size(0), -1)
test_averages_flattened = test_averages.view(test_averages.size(0), -1)

# Training loop
epochs = 1000
train_start_time = time.time()
for epoch in range(epochs):
    # Zero the gradients
    optimizer.zero_grad()

    # Forward pass
    train_outputs = model(train_averages_flattened)
    # Calculate the loss
    loss = criterion(train_outputs, train_labels)
    # Backward pass
    loss.backward()
    # Update the weights
    optimizer.step()

    # Print loss for this epoch
    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, epochs, loss.item()))
train_end_time = time.time()
train_time = train_end_time - train_start_time
print('train time')
print(train_time)
# Evaluation
test_start_time = time.time()
with torch.no_grad():
    # Forward pass on test data
    test_outputs = model(test_averages_flattened)
    probabilities = F.softmax(test_outputs, dim=1)[:, 1]

    # Get the predicted labels
    _, test_predicted = torch.max(test_outputs.data, 1)
    # Calculate accuracy
    test_total = test_labels.size(0)
    test_correct = (test_predicted == test_labels).sum().item()
    test_accuracy = test_correct / test_total
    print('Test Accuracy: {:.4f}'.format(test_accuracy))


fpr, tpr, _ = roc_curve(test_labels, probabilities)
print('fpr')
print(fpr)
print('tpr')
print(tpr)

auc_score = auc(fpr, tpr)
print('auc score')
print(auc_score)
# Plot accuracy and time results

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

classes = ('Wildfire', 'No Wildfire')

# Build confusion matrix
cf_matrix = confusion_matrix(test_labels, test_predicted)
df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.savefig('output_reg.png')

test_end_time = time.time()
test_time = test_end_time - test_start_time
print('test time')
print(test_time)
