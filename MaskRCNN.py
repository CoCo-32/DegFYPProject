import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights

# Use the 'weights' argument instead of 'pretrained'
weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  # You can also use .DEFAULT for the most recent version

model = maskrcnn_resnet50_fpn(weights=weights)
model.eval()  # Set to evaluation mode if not training

# Replace the model's head for the classifier
num_classes = 3  # Including background class

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Replace the box_predictor with a new one (classification and box regression heads)
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Replace the mask_predictor if segmentation masks are important
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
hidden_layer = 256  # You can experiment with this size

model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)