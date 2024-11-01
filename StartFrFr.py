import os
import json
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.mask import encode, decode
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm

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
            
        # Define default transforms
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])

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
        areas = []
        iscrowd = []
        
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
            rle = encode(np.asfortranarray(mask))
            mask = decode(rle)

            masks.append(mask)
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])
            areas.append(annotation['area'])
            iscrowd.append(annotation.get('iscrowd', 0))

        # Convert lists to tensors
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
            
            # Convert boxes from [x, y, width, height] to [x1, y1, x2, y2]
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            # Handle empty annotations
            masks = torch.zeros((0, height, width), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        # Convert image to tensor and normalize
        image = self.transform(image)

        # Create target dictionary in the format expected by Mask R-CNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx + 1]),
            'area': areas,
            'iscrowd': iscrowd
        }

        return image, target

def train_model(model, data_loader, optimizer, device, num_epochs=10):
    """Training function for the Mask R-CNN model"""
    model.train()
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        running_loss = 0.0
        
        # Progress bar for the training
        progress_bar = tqdm(data_loader, total=len(data_loader))
        
        for images, targets in progress_bar:
            # Move inputs to device
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Zero the optimizer gradients
            optimizer.zero_grad()
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass and optimization
            losses.backward()
            optimizer.step()
            
            # Update running loss
            running_loss += losses.item()
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {losses.item():.4f}")
        
        # Print epoch summary
        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
        
    return model

def collate_fn(batch):
    """Custom collate function to handle variable-sized images and annotations"""
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset parameters
    json_file = 'annotations_in_coco.json'
    img_dir = 'SolDef_AI/Labeled'
    batch_size = 2
    num_workers = 4
    num_epochs = 10
    learning_rate = 0.005
    
    # Create dataset
    dataset = MaskRCNNDataset(json_file=json_file, img_dir=img_dir)
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    model = maskrcnn_resnet50_fpn(weight='DEFAULT')
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train the model
    try:
        model = train_model(model, data_loader, optimizer, device, num_epochs=num_epochs)
        print("Training completed successfully!")
        
        # Save the trained model
        torch.save(model.state_dict(), 'mask_rcnn_model.pth')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"An error occurred during training: {str(e)}")
    
    # Test the trained model
    model.eval()
    with torch.no_grad():
        images, targets = next(iter(data_loader))
        images = [image.to(device) for image in images]
        predictions = model(images)
        
        print("\nTest Predictions:")
        print(f"Number of images processed: {len(images)}")
        print(f"Number of detections in first image: {len(predictions[0]['boxes'])}")