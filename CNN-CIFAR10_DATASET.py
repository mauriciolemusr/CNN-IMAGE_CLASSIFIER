#
#
# USING THE CIFAR10 DATASET FROM PYTORCH
# https://www.cs.toronto.edu/~kriz/cifar.html
#
#
# Code by Mauricio Lemus Rochin
# Built using Pytorch, the CIFAR10 dataset
#
#


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, fbeta_score
import seaborn as sns
import pandas as pd

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Download and load the dataset
train_data = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

# Split the training data into training and validation sets
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size
train_data, val_data = torch.utils.data.random_split(train_data, [train_size, val_size])

# Labels map
labels_map = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Defining the NN architecture using CrossEntropyLoss
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes in CIFAR-10 dataset
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

# Defining the NN architecture using NLLLoss
class CIFAR10CNN_NLLLoss(nn.Module):
    def __init__(self):
        super(CIFAR10CNN_NLLLoss, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes in CIFAR-10 dataset
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Flatten the output of the convolutional layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Apply log_softmax for NLLLoss
        x = F.log_softmax(x, dim=1)
        
        return x

# Initialize the models
model_nllloss = CIFAR10CNN_NLLLoss().to(device)
model = CIFAR10CNN().to(device)

# Define the loss functions and optimizers
criterion_nllloss = nn.NLLLoss()
optimizer_nllloss = torch.optim.Adam(model_nllloss.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create data loaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 20
epoch_losses = []
epoch_losses_nllloss = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_loss_nllloss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass CrossEntropyLoss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Forward pass NLLLoss
        outputs_nllloss = model_nllloss(inputs)
        loss_nllloss = criterion_nllloss(outputs_nllloss, labels)

        # Backward pass and optimization for CrossEntropyLoss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Backward pass and optimization for NLLLoss
        optimizer_nllloss.zero_grad()
        loss_nllloss.backward()
        optimizer_nllloss.step()

        running_loss += loss.item()
        running_loss_nllloss += loss_nllloss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, CrossEntropyLoss: {epoch_loss:.4f}")

    epoch_loss_nllloss = running_loss_nllloss / len(train_loader)
    epoch_losses_nllloss.append(epoch_loss_nllloss)
    print(f"Epoch {epoch + 1}/{num_epochs}, NLLLoss: {epoch_loss_nllloss:.4f}")

# Plot the loss curves
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-', label='CrossEntropyLoss')
plt.plot(range(1, num_epochs + 1), epoch_losses_nllloss, marker='o', linestyle='-', label='NLLLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.grid(True)
plt.show()

# Evaluation on test set for both models
def evaluate_model(model, criterion, test_loader):
    model.eval()
    predictions = []
    true_labels = []
    class_correct = [0] * 10
    class_total = [0] * 10
    class_pred_total = [0] * 10

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
                class_pred_total[prediction] += 1

    return predictions, true_labels, class_correct, class_total, class_pred_total

# Evaluate the models
predictions_nllloss, true_labels_nllloss, class_correct_nllloss, class_total_nllloss, class_pred_total_nllloss = evaluate_model(model_nllloss, criterion_nllloss, test_loader)
predictions, true_labels, class_correct, class_total, class_pred_total = evaluate_model(model, criterion, test_loader)

# Calculate metrics for both models
def calculate_metrics(predictions, true_labels, class_correct, class_total, class_pred_total, model_name):
    cm = confusion_matrix(true_labels, predictions)
    print(f"Confusion Matrix ({model_name}):")
    print(cm)

    f2_scores = fbeta_score(true_labels, predictions, average=None, beta=2)
    print(f"F2 Scores ({model_name}):")
    for i in range(10):
        print(f"Class {labels_map[i]}: F2 Score = {f2_scores[i]:.4f}")

    most_difficult_class = np.argmin(f2_scores)
    print(f"The most difficult class to predict ({model_name}) is: {labels_map[most_difficult_class]}")

    accuracy = 100 * np.mean(np.array(predictions) == np.array(true_labels))
    print(f'Overall Accuracy ({model_name}): {accuracy:.2f}%')

    class_precision = []
    class_recall = []

    for i in range(10):
        tp = class_correct[i]
        fp = class_pred_total[i] - class_correct[i]
        fn = class_total[i] - class_correct[i]

        precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0

        class_precision.append(precision)
        class_recall.append(recall)
        print(f'Class {labels_map[i]} - {model_name}: Precision = {precision:.2f}%, Recall = {recall:.2f}%')

    return class_precision, class_recall

# Calculate metrics for NLLLoss model
class_precision_nllloss, class_recall_nllloss = calculate_metrics(predictions_nllloss, true_labels_nllloss, class_correct_nllloss, class_total_nllloss, class_pred_total_nllloss, "NLLLoss")

# Calculate metrics for CrossEntropyLoss model
class_precision, class_recall = calculate_metrics(predictions, true_labels, class_correct, class_total, class_pred_total, "CrossEntropyLoss")

# Plot the precision vs recall bar graph for each class (NLLLoss)
plt.figure(figsize=(12, 8))
x = np.arange(10)
width = 0.35

plt.bar(x - width/2, class_precision_nllloss, width, label='Precision (NLLLoss)')
plt.bar(x + width/2, class_recall_nllloss, width, label='Recall (NLLLoss)')

plt.ylabel('Percentage')
plt.title('Precision vs Recall for Each Class (NLLLoss)')
plt.xticks(x, labels_map.values(), rotation=45)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Plot the precision vs recall bar graph for each class (CrossEntropyLoss)
plt.figure(figsize=(12, 8))
x = np.arange(10)
width = 0.35

plt.bar(x - width/2, class_precision, width, label='Precision (CrossEntropyLoss)')
plt.bar(x + width/2, class_recall, width, label='Recall (CrossEntropyLoss)')

plt.ylabel('Percentage')
plt.title('Precision vs Recall for Each Class (CrossEntropyLoss)')
plt.xticks(x, labels_map.values(), rotation=45)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()