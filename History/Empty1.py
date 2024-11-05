import os
import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from PIL import Image
import skimage.draw

class CustomDataset(Dataset):
    def __init__(self, dataset_dir, subset, transforms=None):
        self.transforms = transforms
        self.dataset_dir = dataset_dir
        self.subset = subset

        # Define classes
        self.class_info = [
            {"source": "custom", "id": 1, "name": "Blue_Marble"},
            {"source": "custom", "id": 2, "name": "Non_Blue_Marble"}
        ]
        self.class_names = [c["name"] for c in self.class_info]

        # Load annotations
        annotations_file = os.path.join(dataset_dir, subset, "labels/marbles_two_class_VGG_json_format.json")
        with open(annotations_file) as f:
            annotations = json.load(f)
        
        self.image_info = []
        for filename, annotation in annotations.items():
            if 'regions' in annotation:
                image_path = os.path.join(dataset_dir, subset, filename)
                self.image_info.append({
                    "id": filename,
                    "path": image_path,
                    "annotations": annotation['regions']
                })

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        img_info = self.image_info[idx]
        image = Image.open(img_info["path"]).convert("RGB")
        
        # Parse annotations
        num_objs = len(img_info['annotations'])
        boxes = []
        masks = []
        labels = []
        
        for _, anno in img_info['annotations'].items():
            poly = anno['shape_attributes']
            label = anno['region_attributes']['label']
            
            # Create binary mask
            mask = np.zeros(image.size[::-1], dtype=np.uint8)
            rr, cc = skimage.draw.polygon(poly['all_points_y'], poly['all_points_x'])
            mask[rr, cc] = 1
            masks.append(mask)
            
            # Bounding box
            x_min, y_min = np.min(poly['all_points_x']), np.min(poly['all_points_y'])
            x_max, y_max = np.max(poly['all_points_x']), np.max(poly['all_points_y'])
            boxes.append([x_min, y_min, x_max, y_max])
            
            # Label
            labels.append(self.class_names.index(label) + 1)  # +1 because 0 is background

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
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
            image, target = self.transforms(image, target)

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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define transforms
    def get_transform(train):
        transforms = []
        transforms.append(torchvision.transforms.ToTensor())
        if train:
            transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
        return torchvision.transforms.Compose(transforms)

    # Create dataset
    dataset_train = CustomDataset(dataset_dir="marble_dataset", subset="train", transforms=get_transform(train=True))
    dataset_val = CustomDataset(dataset_dir="marble_dataset", subset="val", transforms=get_transform(train=False))

    # Create data loaders
    train_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4,
                            collate_fn=lambda x: tuple(zip(*x)))

    # Create model
    num_classes = 3  # Background + Blue_Marble + Non_Blue_Marble
    model = get_model(num_classes)
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop
    num_epochs = 25

    for epoch in range(num_epochs):
        model.train()
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

    # Save the trained model
    torch.save(model.state_dict(), 'mask_rcnn_marble_model.pth')

if __name__ == "__main__":
    main()