import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from Start import CustomDataset  # Assuming you've saved the previous code in CustomDataset.py

def show_sample(image, mask, class_ids):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Display the image
    ax1.imshow(image.permute(1, 2, 0))  # Convert from CxHxW to HxWxC
    ax1.set_title("Original Image")
    ax1.axis('off')
    
    # Display the mask
    ax2.imshow(mask, cmap='gray')
    ax2.set_title(f"Mask (Class IDs: {class_ids.tolist()})")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Set up the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset("path/to/dataset", "train", transform=transform)

# Display a few samples
num_samples = 5
for i in range(num_samples):
    image, mask, class_ids = dataset[i]
    show_sample(image, mask, class_ids)