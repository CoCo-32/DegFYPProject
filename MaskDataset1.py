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
    def __init__(self, txt_file, transforms=None, target_size=(512, 512)):
        self.txt_file = txt_file
        self.transforms = transforms
        self.target_size = target_size  # New attribute for target size
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
        
        # Resize image
        image = img.resize(self.target_size)

        # Extract polygons and convert to masks and bounding boxes
        masks = []
        boxes = []
        for obj in annotations['shapes']:
            points = np.array(obj['points'])
            # Resize points to match the new image size
            points = points * np.array([self.target_size[0] / 2560, self.target_size[1] / 1440])
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            boxes.append([xmin, ymin, xmax, ymax])
            
            # Create a binary mask for each polygon
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)  # Create a drawing object
            draw.polygon(points.flatten().tolist(), outline=1, fill=1)
            masks.append(np.array(mask))

        # Convert data to PyTorch tensors
        masks = np.array(masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.ones((len(masks),), dtype=torch.int64)  # For simplicity, we assume all objects are of the same class.

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        if self.transforms is None:
            image = ToTensor()(image)
        else:
            image = self.transforms(image)

        return image, target
    


# Define transforms
transforms = T.Compose([T.ToTensor()])
