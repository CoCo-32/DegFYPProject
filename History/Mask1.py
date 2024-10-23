import os
import json
import torch
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms as T
from PIL import Image
from History.MaskDataset1 import CustomDataset

def get_model(num_classes):
    # Load a pre-trained Mask R-CNN model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the mask predictor with a new one
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms to convert images to PyTorch tensors
    transforms = T.Compose([T.ToTensor()])


    # Set up the dataset and dataloader
    dataset = CustomDataset(txt_file='solindex.txt', transforms=transforms)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Initialize the model and optimizer
    num_classes = 2  # 1 class (object) + background
    model = get_model(num_classes)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 10
    model.train()

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            
            # Calculate total loss
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimizer step
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            print(f'Loss: {losses.item()}')
    # Save the model
    torch.save(model.state_dict(), 'mask_rcnn_model.pth')

if __name__ == "__main__":
    main()