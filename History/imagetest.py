import os
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class ObjectDetectionDataset(Dataset):
    def __init__(self, txt_file, image_size=(640, 360), transform=ToTensor()):
        self.txt_file = txt_file
        self.transform = transform
        self.image_size = image_size  # New parameter for resizing
        with open(txt_file, 'r') as file:
            lines = file.readlines()
        self.images = []
        self.annotations = []

        for line in lines:
            img_file, json_file = line.strip().split()
            image_path = os.path.join('SolDef_AI/Labeled', img_file)
            json_path = os.path.join('SolDef_AI/Labeled', json_file)
            self.images.append(image_path)
            self.annotations.append(json_path)

    def __len__(self):
        return len(self.images)
    
    def label_classification(self, label):
            # If any annotation belongs to "Poor", "exc_solder", or "spike", classify it as "No Good"
            if label in ['No Good', 'Poor', 'exc_solder', 'spike']:
                return 0  # No Good
            return 1  # Good

    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Resize image
        original_size = image.size  # Save the original size for scaling annotations
        image = image.resize(self.image_size, Image.BILINEAR)

        # Load annotations
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        # Extract polygons and labels
        boxes = []
        labels = []
        for shape in annotation['shapes']:
            label = shape['label']
            points = shape['points']  # List of points defining the polygon
            polygon = np.array(points, dtype=np.float32)  # Convert to NumPy array for OpenCV
            
            # Resize bounding box to match the new image size
            scale_x = self.image_size[0] / original_size[0]
            scale_y = self.image_size[1] / original_size[1]
            polygon[:, 0] *= scale_x
            polygon[:, 1] *= scale_y
            
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

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)

        # Apply transformation if specified
        if self.transform:
            image = self.transform(image)

        # Visualize the final processed image and bounding boxes
        self.visualize_image(image, boxes)

        return image, target

    def visualize_image(self, image, boxes):
        """Visualize the image with bounding boxes."""
        image = image.permute(1, 2, 0).numpy()  # Convert from C x H x W to H x W x C
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        
        for box in boxes.numpy():
            x_min, y_min, x_max, y_max = box
            plt.gca().add_patch(plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                               fill=False, edgecolor='r', linewidth=2))

        plt.axis('off')
        plt.show()
