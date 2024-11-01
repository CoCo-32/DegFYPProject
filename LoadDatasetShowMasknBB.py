import os
import json
import numpy as np
import cv2  
import torch
from torch.utils.data import Dataset
from pycocotools.mask import encode, decode
import matplotlib.pyplot as plt

class MaskRCNNDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load the JSON annotation file
        with open(json_file) as f:
            self.data = json.load(f)
        
        # Create a mapping from image_id to annotations
        self.image_info = {image['id']: image for image in self.data['images']}
        self.annotations = {image['id']: [] for image in self.data['images']}
        
        for annotation in self.data['annotations']:
            self.annotations[annotation['image_id']].append(annotation)

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        # Get the image info
        image_info = self.image_info[idx + 1]  # Adjusting index for zero-based
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Prepare the masks and target
        masks = []
        boxes = []
        labels = []
        
        for annotation in self.annotations[idx + 1]:
            # Create mask from segmentation
            segmentation = annotation['segmentation']
            
            # Check if segmentation is empty
            if len(segmentation) == 0:
                continue
            
            # Create an empty mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Fill the mask with the segmentation
            for poly in segmentation:
                cv2.fillPoly(mask, [np.array(poly).reshape((-1, 1, 2)).astype(np.int32)], 1)

            # Encode the mask to RLE
            rle = encode(np.asfortranarray(mask))  # Use Fortran order
            mask = decode(rle)  # Decode the mask

            masks.append(mask)
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])

        # Convert lists to numpy arrays
        masks = np.stack(masks, axis=-1) if masks else np.zeros((height, width, 0), dtype=np.uint8)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        # Apply transformations if any
        if self.transform:
            # Define your transformation logic here
            pass

        return {
            'image': image,
            'masks': masks,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx + 1
        }

# Example of how to use the dataset
if __name__ == "__main__":
    json_file = 'annotations_in_coco.json'
    img_dir = 'SolDef_AI/Labeled'
    dataset = MaskRCNNDataset(json_file=json_file, img_dir=img_dir)
    
    # Load the sample from the dataset (replace `dataset` and `0` with your actual dataset and index)
    sample = dataset[300]
    image = sample['image']
    masks = sample['masks']
    boxes = sample['boxes']
    labels = sample['labels']

    # Plot the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Original Image")
    plt.title("Original Image with Masks and Bounding Boxes")
    plt.axis("off")

    # Overlay each mask and bounding box
    for i in range(masks.shape[-1]):
        mask = masks[:, :, i]
        
        # Display the mask with transparency
        plt.imshow(mask, cmap='jet', alpha=0.3)  # Adjust alpha for transparency
        
        # Draw the bounding box
        box = boxes[i]
        x_min, y_min, width, height = box
        x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)
        rect = plt.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='yellow', facecolor='none')
        plt.gca().add_patch(rect)
        
        # Display label
        plt.text(x_min, y_min - 10, f"Label: {labels[i]}", color='yellow', fontsize=12, weight='bold')

    plt.show()