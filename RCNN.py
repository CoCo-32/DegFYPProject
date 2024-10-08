import os
import json
import torch
import torch.utils
import torch.utils.data
import torchvision
import numpy as np
from PIL import Image

# Load a pre-trained model on the COCO dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weight = None)
# You can set `pretrained_backbone=True` if you want to fine-tune just the backbone

model.eval

# Number of classes (including background)
num_classes = 5  # (e.g., 1 for background + 4 defects)

# Get the number of input features from the pre-trained model
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the box predictor with a new one for your number of classes
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Parsing index file
def parse_index_file(txt_file):
    with open(txt_file, 'r') as file:
        lines = file.readlines()
    image_files = []
    json_files = []
    for line in lines:
        img_file, json_file = line.strip().split()
        image_files.append(img_file)
        json_files.append(json_file)
    return image_files, json_files

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Load an image and draw annotations
def load_image_with_annotations(image, annotations):
    #image = cv2.imread(image_path)
    #image = cv2.resize(image, (224, 224))  # Resize image for the CNN model input

    for annotation in annotations:
        label = annotation['name']
        polygon = annotation['polygon']
        # Draw polygon
        cv2.polylines(image, [polygon], isClosed=True, color=(0, 0, 255), thickness=3)
        # Calculate the position to place the label
        text_position = tuple(polygon[0][0])  # Use the first point of the polygon
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)
    # Display the image with annotations
    resized_image = cv2.resize(image, (512, 512))
    cv2.imshow('Image with Annotations', resized_image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

import torchvision.transforms as T

# Your dataset class must return images and corresponding target dictionaries
class SolderDataset(torch.utils.data.Dataset):
    def __init__(self, image_files, json_files, transforms=None):
        self.image_files = image_files
        self.json_files = json_files
        self.transform = transforms if transforms else T.Compose([T.ToTensor()])

    def __getitem__(self, index):
        # Load image
        img_path = self.image_files[index]
        img = Image.open(img_path)
        img = self.transform(img)

        json_file = self.json_files[index]
        with open(json_file, 'r') as file:
            data = json.load(file)

        # Extract polygons and labels
        boxes = []
        labels = []
        for shape in data['shapes']:
            label = shape['label']
            points = shape['points']  # List of points defining the polygon
            polygon = np.array(points, dtype=np.float32)  # Convert to NumPy array for OpenCV
             # Convert polygon to a bounding box (for Faster R-CNN)
            x_min = np.min(polygon[:, 0])
            y_min = np.min(polygon[:, 1])
            x_max = np.max(polygon[:, 0])
            y_max = np.max(polygon[:, 1])
            boxes.append([x_min, y_min, x_max, y_max])
            
            # Append label (you may need a mapping for label -> class index)
            labels.append(self.label_classification(label))

        # Convert to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
        }

        return img, target

    def __len__(self):
        return len(self.image_files)
    
    # Label processing: Mapping multiple classes to binary (Good, No Good)
    def label_classification(self, label):
            # If any annotation belongs to "Poor", "exc_solder", or "spike", classify it as "No Good"
            if label in ['No Good', 'Poor', 'exc_solder', 'spike']:
                return 0  # No Good
            return 1  # Good

# Assuming your text file is named 'data.txt' and is located in the same directory
txt_file = 'solindex.txt'
image_files, json_files = parse_index_file(txt_file)

# Example usage
dataset = SolderDataset(image_files, json_files)

# Create a data loader
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))


# Use GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Define optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {losses.item()}")


import matplotlib.pyplot as plt
import numpy as np
import cv2

# Put model in evaluation mode
model.eval()

# Load a test image
img, _ = dataset[0]  # Use a test image from your dataset
img = img.to(device)

# Perform inference
with torch.no_grad():
    prediction = model([img])[0]

# Visualize the bounding boxes
def visualize(img, boxes):
    img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert tensor to numpy array
    img_np = (img_np * 255).astype(np.uint8)  # Convert from float32 to uint8

    for box in boxes:
        box = box.cpu().numpy().astype(np.int32)
        cv2.rectangle(img_np, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
    
    plt.imshow(img_np)
    plt.show()

# Example of visualizing the results
visualize(img, prediction['boxes'])
