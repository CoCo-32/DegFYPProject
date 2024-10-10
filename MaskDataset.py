import os
import torch
import numpy as np
from PIL import Image, ImageDraw
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import ToTensor

class CustomDataset(Dataset):
    def __init__(self, txt_file, transforms=None):
        self.txt_file = txt_file
        self.transforms = transforms
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

    def __getitem__(self, idx):
        img_path = self.images[idx]
        ann_path = self.annotations[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Load annotations from JSON file
        with open(ann_path) as f:
            annotations = json.load(f)

        # Extract polygons and convert to masks and bounding boxes
        masks = []
        boxes = []
        for obj in annotations['shapes']:
            points = np.array(obj['points'])
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Create a binary mask for each polygon
            mask = Image.new('L', img.size, 0)
            draw = ImageDraw.Draw(mask)  # Create a drawing object
            draw.polygon(points.flatten().tolist(), outline=1, fill=1)
            mask = np.array(mask)
            masks.append(mask)

        # Convert data to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.ones((len(masks),), dtype=torch.int64)  # For simplicity, we assume all objects are of the same class.

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        if self.transform is None:
            image = ToTensor()(image)
        else:
            image = self.transform(image)

        return image, target
    


# Define transforms
transforms = T.Compose([T.ToTensor()])
