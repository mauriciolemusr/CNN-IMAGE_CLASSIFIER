# Code by Mauricio Lemus Rochin
# Built using Pytorch, the Fashion MNIST dataset, and Claude AI

import torch
from torch.utils.data import Dataset
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

# Normalizing pixel values
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1] range
])

# Download and load the dataset
training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)

# Split the training data into training and validation sets
from torch.utils.data import random_split

train_size = int(0.8 * len(training_data))
val_size = len(training_data) - train_size
train_data, val_data = random_split(training_data, [train_size, val_size])

# Labels map
labels_map = {
    0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot",
}

# Defining the NN architecture using CrossEntropyLoss
class FashionCNN(nn.Module):
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 30)
        self.fc2 = nn.Linear(30, 10)
        
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
    
#Defining the NN architecture using NLLLoss    
    
class FashionCNN_NLLLoss(nn.Module):
    def __init__(self):
        super(FashionCNN_NLLLoss, self).__init__()
        
        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 30)
        self.fc2 = nn.Linear(30, 10)
        
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
    
# Initialize the model with NLLLoss
model_nllloss = FashionCNN_NLLLoss().to(device)

# Define the loss function for NLLLoss
criterion_nllloss = nn.NLLLoss()

# Define the optimizer for NLLLoss
optimizer_nllloss = torch.optim.Adam(model_nllloss.parameters(), lr=0.001)

# Initialize the model with CrossEntropyLoss
model = FashionCNN().to(device)

# Define the loss function for CrossEntropyLoss
criterion = nn.CrossEntropyLoss()

# Define the optimizer for CrossEntropyLoss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Create data loaders
batch_size = 64
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 20  # Set the number of epochs
epoch_losses = []  # List to store the loss value for each epoch
epoch_losses_nllloss = []  # List to store the NLLLoss value for each epoch

# Train the model
for epoch in range(num_epochs):
    running_loss = 0.0
    running_loss_nllloss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        
        # Forward pass CrossEntropyLoss
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Foward pass NLLLoss
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

        # Compute running loss
        running_loss += loss.item()
        running_loss_nllloss += loss_nllloss.item()

    epoch_loss = running_loss / len(train_loader)
    epoch_losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{num_epochs}, CrossEntropyLoss: {epoch_loss:.4f}")

    epoch_loss_nllloss = running_loss_nllloss / len(train_loader)
    epoch_losses_nllloss.append(epoch_loss_nllloss)
    print(f"Epoch {epoch + 1}/{num_epochs}, NLLLoss: {epoch_loss_nllloss:.4f}")


# Plot the loss curve for CrossEntropyLoss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (CrossEntropyLoss)')
plt.grid(True)
plt.show()

# Plot the loss curve for NLLLoss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses_nllloss, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (NLLLoss)')
plt.grid(True)
plt.show()


# Evaluation on test set for the model with CrossEntropyLoss
model.eval() 
predictions = []
true_labels = []
class_correct = [0] * 10
class_total = [0] * 10
class_pred_total = [0] * 10

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # Update class-wise statistics
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1
            class_pred_total[prediction] += 1

# Evaluation on test set for the model with NLLLoss
model_nllloss.eval()
predictions_nllloss = []
true_labels_nllloss = []
class_correct_nllloss = [0] * 10
class_total_nllloss = [0] * 10
class_pred_total_nllloss = [0] * 10

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs_nllloss = model_nllloss(inputs)
        _, predicted_nllloss = torch.max(outputs_nllloss.data, 1)
        predictions_nllloss.extend(predicted_nllloss.cpu().numpy())
        true_labels_nllloss.extend(labels.cpu().numpy())

        # Update class-wise statistics for NLLLoss model
        for label, prediction in zip(labels, predicted_nllloss):
            if label == prediction:
                class_correct_nllloss[label] += 1
            class_total_nllloss[label] += 1
            class_pred_total_nllloss[prediction] += 1

# Calculate confusion matrix for NLLLoss model
cm_nllloss = confusion_matrix(true_labels_nllloss, predictions_nllloss)
print("Confusion Matrix (NLLLoss):")
print(cm_nllloss)

# Create a DataFrame for the confusion matrix (NLLLoss)
cm_df_nllloss = pd.DataFrame(cm_nllloss, index=labels_map.values(), columns=labels_map.values())

# Plot the confusion matrix as a heatmap (NLLLoss)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df_nllloss, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (NLLLoss)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Calculate F2 score for each class (NLLLoss)
f2_scores_nllloss = fbeta_score(true_labels_nllloss, predictions_nllloss, average=None, beta=2)
print("F2 Scores (NLLLoss):")
for i in range(10):
    print(f"Class {i} ({labels_map[i]}): F2 Score = {f2_scores_nllloss[i]:.4f}")

# Find the class with the lowest F2 score (NLLLoss)
most_difficult_class_nllloss = np.argmin(f2_scores_nllloss)
print(f"The most difficult class to predict (NLLLoss) is: {labels_map[most_difficult_class_nllloss]}")

# Calculate overall accuracy for NLLLoss model
accuracy_nllloss = 100 * np.mean(np.array(predictions_nllloss) == np.array(true_labels_nllloss))
print(f'Overall Accuracy (NLLLoss): {accuracy_nllloss:.2f}%')

# Calculate precision and recall for each class (NLLLoss)
class_precision_nllloss = []
class_recall_nllloss = []

for i in range(10):
    tp = class_correct_nllloss[i]
    fp = class_pred_total_nllloss[i] - class_correct_nllloss[i]
    fn = class_total_nllloss[i] - class_correct_nllloss[i]

    precision = 100 * tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = 100 * tp / (tp + fn) if (tp + fn) > 0 else 0

    class_precision_nllloss.append(precision)
    class_recall_nllloss.append(recall)
    print(f'Class {i} ({labels_map[i]}) - NLLLoss: Precision = {precision:.2f}%, Recall = {recall:.2f}%')

# Plot the precision vs recall bar graph for each class (NLLLoss)
plt.figure(figsize=(10, 8))
x = np.arange(len(labels_map))
width = 0.35

plt.bar(x - width/2, class_precision_nllloss, width, label='Precision (NLLLoss)')
plt.bar(x + width/2, class_recall_nllloss, width, label='Recall (NLLLoss)')

plt.ylabel('Percentage')
plt.title('Precision vs Recall for Each Class (NLLLoss)')
plt.xticks(x, labels_map.values(), rotation=45)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()

# Evaluation on test set for the model with CrossEntropyLoss
model.eval() 
predictions = []
true_labels = []
class_correct = [0] * 10
class_total = [0] * 10
class_pred_total = [0] * 10

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        predictions.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

        # Update class-wise statistics
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                class_correct[label] += 1
            class_total[label] += 1
            class_pred_total[prediction] += 1

# Calculate confusion matrix (CrossEntropyLoss)
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix (CrossEntropyLoss):")
print(cm)

# Create a DataFrame for the confusion matrix (CrossEntropyLoss)
cm_df = pd.DataFrame(cm, index=labels_map.values(), columns=labels_map.values())

# Plot the confusion matrix as a heatmap (CrossEntropyLoss)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (CrossEntropyLoss)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Calculate F2 score for each class (CrossEntropyLoss)
f2_scores = fbeta_score(true_labels, predictions, average=None, beta=2)
print("F2 Scores (CrossEntropyLoss):")
for i in range(10):
    print(f"Class {i} ({labels_map[i]}): F2 Score = {f2_scores[i]:.4f}")

# Find the class with the lowest F2 score (CrossEntropyLoss)
most_difficult_class = np.argmin(f2_scores)
print(f"The most difficult class to predict (CrossEntropyLoss) is: {labels_map[most_difficult_class]}")

# Calculate overall accuracy (CrossEntropyLoss)
accuracy = 100 * np.mean(np.array(predictions) == np.array(true_labels))
print(f'Overall Accuracy (CrossEntropyLoss): {accuracy:.2f}%')

# Calculate precision and recall for each class (CrossEntropyLoss)
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
    print(f'Class {i} ({labels_map[i]}) - CrossEntropyLoss: Precision = {precision:.2f}%, Recall = {recall:.2f}%')

# Plot the precision vs recall bar graph for each class (CrossEntropyLoss)
plt.figure(figsize=(10, 8))
x = np.arange(len(labels_map))
width = 0.35

plt.bar(x - width/2, class_precision, width, label='Precision (CrossEntropyLoss)')
plt.bar(x + width/2, class_recall, width, label='Recall (CrossEntropyLoss)')

plt.ylabel('Percentage')
plt.title('Precision vs Recall for Each Class (CrossEntropyLoss)')
plt.xticks(x, labels_map.values(), rotation=45)
plt.legend(loc='lower right')
plt.grid(True)
plt.show()