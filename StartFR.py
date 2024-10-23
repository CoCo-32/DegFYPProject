import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import cv2


class SolderDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        self.dataset_dir = os.path.join(dataset_dir)
        self.transform = transform
        self.class_dict = {"good": 1, "no_good": 2, "exc_solder": 3, "poor":4 , "spike":5}
        
        # Load annotations
        with open(os.path.join(self.dataset_dir, "labels/marbles_two_class_VGG_json_format.json"), "r") as f:
            annotations = json.load(f)
        self.annotations = [a for a in annotations.values() if a['regions']]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.dataset_dir, ann['filename'])
        image = Image.open(img_path).convert("RGB")
        
        # Create mask
        mask = np.zeros((image.size[1], image.size[0]), dtype=np.uint8)
        class_ids = []
        
        for region in ann['regions'].values():
            points = np.array([(region['shape_attributes']['all_points_x'][i], 
                                region['shape_attributes']['all_points_y'][i]) 
                               for i in range(len(region['shape_attributes']['all_points_x']))], np.int32)
            cv2.fillPoly(mask, [points], color=1)
            
            class_name = ['label']#region['region_attributes']['label']
            class_ids.append(self.class_dict[class_name])

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        mask = torch.from_numpy(mask).long()
        class_ids = torch.tensor(class_ids, dtype=torch.long)
        
        return image, mask, class_ids

# Example usage:
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = SolderDataset("path/to/dataset", "train", transform=transform)