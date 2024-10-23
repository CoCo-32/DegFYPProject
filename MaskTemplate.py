import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# 1. Define a custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, masks, labels):
        self.imgs = imgs
        self.masks = masks
        self.labels = labels
    
    def __getitem__(self, idx):
        img = self.imgs[idx]
        mask = self.masks[idx]
        label = self.labels[idx]
        
        return img, {"masks": mask, "labels": label}
    
    def __len__(self):
        return len(self.imgs)

# 2. Create a custom Mask R-CNN model
def get_model_instance_segmentation(num_classes):
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model

# 3. Prepare the data
# Assume you have your images, masks, and labels ready
imgs = [...]  # List of tensors
masks = [...]  # List of tensors
labels = [...]  # List of tensors

dataset = CustomDataset(imgs, masks, labels)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# 4. Initialize the model
num_classes = 2  # Background + 1 object class
model = get_model_instance_segmentation(num_classes)

# 5. Set up the optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# 6. Training loop
num_epochs = 10
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item()}")

# 7. Save the model
torch.save(model.state_dict(), 'mask_rcnn_model.pth')