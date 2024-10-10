import os
import json
import torch
from PIL import Image
from torchvision.transforms import functional as F

class Dataset(torch.utils.data.Dataset):
    def __init__(self, txt_file, transforms=None):
        self.txt_file = txt_file
        self.transforms = transforms
        #self.image_size = image_size
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
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load annotation
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        with open(ann_path, 'r') as f:
            annotation = json.load(f)

        # Process annotation
        num_objs = len(annotation['objects'])
        boxes = []
        masks = []
        labels = []

        for obj in annotation['objects']:
            xmin = obj['bbox']['xmin']
            ymin = obj['bbox']['ymin']
            xmax = obj['bbox']['xmax']
            ymax = obj['bbox']['ymax']
            boxes.append([xmin, ymin, xmax, ymax])
            
            mask = obj['segmentation']  # Assuming mask is in RLE format
            masks.append(mask)
            
            labels.append(obj['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    