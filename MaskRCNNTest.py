import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.patches import Polygon

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
dataset = PennFudanDataset('data/PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=utils.collate_fn
)

# Load the model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
model.load_state_dict(torch.load('multi_label_model.pth'))
model.eval()  # Set the model to evaluation mode

# If running on GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Load the image for testing
img = Image.open('WIN_20220330_13_28_47_Pro.jpg').convert("RGB")

# Transform the image to tensor
transform = torchvision.transforms.ToTensor()
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

# Send image to device (GPU or CPU)
img_tensor = img_tensor.to(device)

# Run the model on the input image
with torch.no_grad():
    predictions = model(img_tensor)

# Get the output for the first image (as batch size is 1)
pred = predictions[0]

# Convert tensor to NumPy array
img_np = np.array(img)

# Plot the image
fig, ax = plt.subplots(1, figsize=(12, 12))
ax.imshow(img_np)

# Draw bounding boxes and masks
for i in range(len(pred['boxes'])):
    box = pred['boxes'][i].cpu().numpy()  # Bounding box coordinates
    score = pred['scores'][i].cpu().item()  # Confidence score
    
    if score > 0.5:  # Show only predictions with high confidence
        # Draw bounding box
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        
        # Draw mask
        mask = pred['masks'][i, 0].cpu().numpy()  # Get the mask
        mask = mask > 0.5  # Binarize the mask

        # Apply mask on the image
        ax.imshow(np.ma.masked_where(mask == 0, mask), cmap='jet', alpha=0.5)

plt.show()