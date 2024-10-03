import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import os
import cv2
import json
import numpy as np

# Parsing index file
def parse_index_file(index_path):
    with open(index_path, 'r') as file:
        lines = file.readlines()
    image_files = []
    json_files = []
    for line in lines:
        img_file, json_file = line.strip().split()
        image_files.append(img_file)
        json_files.append(json_file)
    return image_files, json_files

# Function to parse JSON annotation file
def parse_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    objects = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']  # List of points defining the polygon
        polygon = np.array(points, np.int32)  # Convert to NumPy array for OpenCV
        polygon = polygon.reshape((-1, 1, 2))  # Reshape for OpenCV polygon drawing
        objects.append({'name': label, 'polygon': polygon})
    return objects

# Label processing: Mapping multiple classes to binary (Good, No Good)
def get_label_from_annotations(annotations):
    for annotation in annotations:
        label = annotation['name']
        # If any annotation belongs to "Poor", "exc_solder", or "spike", classify it as "No Good"
        if label in ['No Good', 'Poor', 'exc_solder', 'spike']:
            return 0  # No Good
    return 1  # Good

# Load an image
def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Resize image for the CNN model input
    return image

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Output 2 classes (Good, No Good)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Set up dataset and dataloader
index_file = 'solindex.txt'
root_dir = 'SolDef_AI/Labeled'

# Custom Dataset class for loading images and annotations from lists
class SolderDataset(Dataset):
    def __init__(self, image_files, json_files, root_dir, transform=None):
        self.image_files = image_files
        self.json_files = json_files
        self.root_dir = root_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        json_path = os.path.join(self.root_dir, self.json_files[idx])
        
        # Load the image and annotations
        image = load_image(img_path)
        annotations = parse_json(json_path)
        
        # Get the label (binary classification)
        label = get_label_from_annotations(annotations)
        
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        
        # Convert the image to a tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W) format
        
        return image, torch.tensor(label, dtype=torch.long)


# Initialize the model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Splitting index into train and test sets (80% train, 20% test)
image_files, json_files = parse_index_file(index_file)
train_img, test_img, train_json, test_json = train_test_split(image_files, json_files, test_size=0.2, random_state=42)

# Creating train and test datasets
train_dataset = SolderDataset(train_img, train_json, root_dir, transform=None)
test_dataset = SolderDataset(test_img, test_json, root_dir, transform=None)

# Creating DataLoader for train and test sets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Training phase (same as before)
for epoch in range(10):  # Train for 10 epochs
    model.train()  # Set model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()  # Zero the parameter gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate the loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the weights
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Testing phase
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during testing
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {100 * correct / total}%")

# Save the trained model
torch.save(model.state_dict(), 'multi_label_model.pth')