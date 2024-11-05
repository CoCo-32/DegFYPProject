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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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

def calculate_iou(pred_box, gt_box):
    """Calculate Intersection over Union (IoU) between two boxes."""
    xA = max(pred_box[0], gt_box[0])
    yA = max(pred_box[1], gt_box[1])
    xB = min(pred_box[2], gt_box[2])
    yB = min(pred_box[3], gt_box[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
    boxBArea = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def evaluate_model(model, data_loader, device, iou_threshold=0.5):
    """Evaluate the model on the dataset and calculate accuracy metrics."""
    model.eval()
    all_true_labels = []
    all_pred_labels = []
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, target in enumerate(targets):
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()

                pred_boxes = outputs[i]['boxes'].cpu()
                pred_labels = outputs[i]['labels'].cpu()
                pred_scores = outputs[i]['scores'].cpu()
                
                # Filter predictions by score threshold
                pred_boxes = pred_boxes[pred_scores > 0.5]
                pred_labels = pred_labels[pred_scores > 0.5]

                # Match predictions to ground truth
                matched_gt = set()
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    best_iou = 0.0
                    best_gt_idx = -1
                    for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if idx in matched_gt or pred_label != gt_label:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou and iou >= iou_threshold:
                            best_iou = iou
                            best_gt_idx = idx
                    
                    # If IoU is high enough, count as true positive
                    if best_gt_idx >= 0:
                        matched_gt.add(best_gt_idx)
                        all_true_labels.append(gt_labels[best_gt_idx].item())
                        all_pred_labels.append(pred_label.item())
                    else:
                        all_pred_labels.append(pred_label.item())
                        all_true_labels.append(0)  # False positive, no corresponding ground truth

                # Add any unmatched ground truths as false negatives
                for idx, gt_label in enumerate(gt_labels):
                    if idx not in matched_gt:
                        all_true_labels.append(gt_label.item())
                        all_pred_labels.append(0)  # No prediction for this ground truth

    # Calculate metrics
    accuracy = np.mean(np.array(all_true_labels) == np.array(all_pred_labels))
    precision = precision_score(all_true_labels, all_pred_labels, average='weighted')
    recall = recall_score(all_true_labels, all_pred_labels, average='weighted')
    f1 = f1_score(all_true_labels, all_pred_labels, average='weighted')
    conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
    
    print(f"Testing Accuracy: {accuracy:.4f}")
    print(f"Testing Precision: {precision:.4f}")
    print(f"Testing Recall: {recall:.4f}")
    print(f"Testing F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    return accuracy, precision, recall, f1, conf_matrix

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
    batch_size = 4  # Ori = 2
    num_workers = 8
    num_epochs = 10       # Ori = 10
    learning_rate = 0.05 # Ori = 0.005
    
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
    model = maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    # Train the model
    try:
        model.train()
        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")
            epoch_loss = 0  
            for images, targets in tqdm(data_loader, desc=f"Training Epoch {epoch + 1}"):
                images = [image.to(device) for image in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                
                # Forward pass
                loss_dict = model(images, targets)
                
                # Total loss
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()
                
                # Backward pass
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(data_loader):.4f}")
        
            # Set model to evaluation mode and calculate training accuracy
            model.eval()
            print(f"Evaluating training accuracy after epoch {epoch + 1}...")
            accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, data_loader, device)
            print(f"Training Accuracy: {accuracy:.4f}")
            print(f"Training Precision: {precision:.4f}")
            print(f"Training Recall: {recall:.4f}")
            print(f"Training F1 Score: {f1:.4f}")
            print("Training Confusion Matrix:")
            print(conf_matrix)
        # Save model
        torch.save(model.state_dict(), 'mask_rcnn_model.pth')
        print("Model saved as 'mask_rcnn_model.pth'")
    
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, data_loader, device)
