import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# Load a pre-trained Mask R-CNN model
model_path = 'MaskRCNNModelV1.2.pth'
model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
model.load_state_dict(torch.load(model_path)) 
model.eval()  # Set the model to evaluation mode

# Define a transformation for the input image
transform = T.Compose([
    T.ToTensor(),
])

# Load your image
image_path = 'SolDef_AI/Test/WIN_20220329_14_32_22_Pro.jpg'
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Run inference
with torch.no_grad():
    predictions = model(image_tensor)

# Process predictions
boxes = predictions[0]['boxes']  # Bounding boxes
scores = predictions[0]['scores']  # Scores for each box
labels = predictions[0]['labels']  # Class labels
masks = predictions[0]['masks']  # Masks for each detected object

# Filter out predictions with a score below a threshold (e.g., 0.5)
threshold = 0.5
filtered_indices = scores > threshold
filtered_boxes = boxes[filtered_indices]
filtered_labels = labels[filtered_indices]
filtered_masks = masks[filtered_indices]

# Display the results (bounding boxes and labels)
for box, label in zip(filtered_boxes, filtered_labels):
    print(f'Label: {label.item()}, Box: {box.tolist()}')

# Optionally, you can visualize the image with the predictions
plt.imshow(image)
for box in filtered_boxes:
    plt.gca().add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, color='red'))
plt.axis('off')
plt.show()
