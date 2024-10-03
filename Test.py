import torch
import torch.nn as nn
import cv2
import os
import numpy as np

# Load the trained model
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

# Function to load image and process it
def load_image(image_path):
    image = cv2.imread(image_path)
    original_image = image.copy()
    image = cv2.resize(image, (224, 224))  # Resize image for the CNN model input
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # Convert to (C, H, W) format
    return original_image, image

# Define a function to draw annotations
def draw_annotations(image, annotations, labels):
    for annotation in annotations:
        polygon = annotation['polygon']
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)  # Draw polygons
        cv2.putText(image, annotation['name'], tuple(polygon[0][0]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return image

# Define a mapping for the outputs
label_mapping = {
    0: "No Good",
    1: "Good",
}

# Load the model
model = SimpleCNN()
model.load_state_dict(torch.load('multi_label_model.pth'))
model.eval()  # Set the model to evaluation mode

# List of test images to manually test
test_images = ['WIN_20220330_13_28_47_Pro.jpg', 'WIN_20220330_13_29_03_Pro.jpg']  # Update with your test images

# Test script modification
for test_image_path in test_images:
    original_image, image = load_image(test_image_path)
    
    with torch.no_grad():
        classes, bboxes = model(image.unsqueeze(0))  # Get classes and bounding boxes
        predicted_classes = torch.argmax(classes, dim=1)

    # Iterate through predictions and draw boxes
    for i in range(len(predicted_classes)):
        label = label_mapping[predicted_classes[i].item()]
        x, y, w, h = bboxes[i].detach().numpy()  # Assuming the output is in (x, y, w, h)
        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
        cv2.putText(original_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Label the box

    # Display the image
    cv2.imshow("Test Image", original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
