import torch
import torch.nn as nn
import cv2
import os
import numpy as np

# Load the pre-trained model
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
        x_class = self.fc2(torch.relu(self.fc1(x)))  # Class prediction
        x_bbox = self.fc_bbox(torch.relu(self.fc1(x)))  # Bounding box prediction
        x = self.fc2(x)
        return x

# Load an image
def load_image(image_path):
    image = cv2.imread(image_path)
    return cv2.resize(image, (224, 224))  # Resize image for the CNN model input

# Annotate the image based on the predictions
def annotate_image(image, predictions):
    # Define positions for annotation
    height, width = image.shape[:2]
    for label, (x, y, w, h) in predictions:
        # Draw rectangle around the detected issues
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image

# Load model weights
model = SimpleCNN()
model.load_state_dict(torch.load('multi_label_model.pth'))
model.eval()

# Set the directory for test images
test_images = ['WIN_20220330_13_28_47_Pro.jpg', 'WIN_20220330_13_29_03_Pro.jpg']

# Process each image
for img_name in test_images:
    image = load_image(img_name)
    image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  # Get predicted class
        predictions = []
        
        # If the prediction indicates "No Good", classify further
        if predicted.item() == 0:  # No Good
            # Simulating specific issues detection (you might want to replace this with actual logic)
            # Here we're just annotating at random positions for demonstration
            predictions.append(('Poor', (50, 50, 100, 100)))  # Simulated coordinates
            predictions.append(('exc_solder', (150, 50, 100, 100)))  # Simulated coordinates
            predictions.append(('spike', (100, 150, 100, 100)))  # Simulated coordinates

        # Annotate the image based on predictions
        annotated_image = annotate_image(image, predictions)

        # Show or save the annotated image
        cv2.imshow('Annotated Image', annotated_image)
        cv2.waitKey(0)  # Wait for a key press to close the image window

# Cleanup
cv2.destroyAllWindows()
