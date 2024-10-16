import os
import json
import torch
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms as T
from PIL import Image, ImageDraw

class CustomDataset(Dataset):
    def __init__(self, txt_file, transforms=None, target_size=(512, 512)):
        self.txt_file = txt_file
        self.transforms = transforms
        self.target_size = target_size
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

        img = Image.open(img_path).convert("RGB")

        with open(ann_path) as f:
            annotations = json.load(f)
        
        image = img.resize(self.target_size)

        masks = []
        boxes = []
        for obj in annotations['shapes']:
            points = np.array(obj['points'])
            points = points * np.array([self.target_size[0] / 2560, self.target_size[1] / 1440])
            xmin, ymin = np.min(points, axis=0)
            xmax, ymax = np.max(points, axis=0)
            boxes.append([xmin, ymin, xmax, ymax])
            
            mask = Image.new('L', image.size, 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(points.flatten().tolist(), outline=1, fill=1)
            masks.append(np.array(mask))

        masks = np.array(masks)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        labels = torch.ones((len(masks),), dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks
        }

        if self.transforms is None:
            image = T.ToTensor()(image)
        else:
            image = self.transforms(image)

        return image, target

def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms = T.Compose([T.ToTensor()])

    dataset = CustomDataset(txt_file='solindex.txt', transforms=transforms)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    num_classes = 2
    model = get_model(num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            print(f'Loss: {losses.item()}')

    torch.save(model.state_dict(), 'mask_rcnn_model.pth')

if __name__ == "__main__":
    main()