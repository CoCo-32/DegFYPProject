import os
import cv2
import json
import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torchvision.models.detection import maskrcnn_resnet50_fpn
from tqdm import tqdm
from RFTraining import ImageDataset

class EnsembleModel:
    def __init__(self, rf_model_path, mask_rcnn_model_path, device):
        # Load the Random Forest model
        self.rf = joblib.load(rf_model_path)
        
        # Load the Mask R-CNN model
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights=None)
        self.mask_rcnn.load_state_dict(torch.load(mask_rcnn_model_path))
        self.mask_rcnn.to(device)
        self.mask_rcnn.eval()  # Set to evaluation mode
        
        self.device = device

    def predict(self, image, boxes=None):
        """Make predictions using the ensemble model"""
        rf_pred = self.rf.predict([self.extract_features(image, boxes)])[0]
        
        # Get Mask R-CNN predictions
        with torch.no_grad():
            image_tensor = torch.tensor(image.transpose((2, 0, 1))).unsqueeze(0).to(self.device)  # Convert image to tensor
            outputs = self.mask_rcnn(image_tensor)
        
        # Extract Mask R-CNN predictions
        mask_rcnn_preds = outputs[0]['labels'].cpu().numpy() if len(outputs) > 0 else []
        
        return rf_pred, mask_rcnn_preds

    def extract_features(self, image, boxes=None):
        """Extract fixed-length feature vector from image for Random Forest"""
        # Similar feature extraction as done in RFImageClassifier
        if len(image.shape) == 3 and image.shape[0] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        features = []
        
        # Histogram feature
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).flatten() / cv2.calcHist([gray], [0], None, [32], [0, 256]).sum()
        features.extend(hist)
        
        # Basic statistics
        features.extend([np.mean(gray), np.std(gray), np.median(gray)])
        
        # Box features (if boxes are provided)
        if boxes is not None and len(boxes) > 0:
            box = boxes[0]  # Assume we take the first box
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            aspect_ratio = width / height if height != 0 else 0
            features.extend([width, height, area, aspect_ratio])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features, dtype=np.float32)

    def evaluate(self, data_loader):
        """Evaluate the ensemble model on the dataset"""
        all_true_labels = []
        all_rf_preds = []
        all_mask_rcnn_preds = []

        for images, targets in tqdm(data_loader):
            for image, target in zip(images, targets):
                # Convert tensor to numpy
                image_np = image.numpy().transpose((1, 2, 0))  # Convert to HWC format
                
                # Get boxes
                boxes = target['boxes'].numpy() if len(target['boxes']) > 0 else None
                
                rf_pred, mask_rcnn_preds = self.predict(image_np, boxes)
                
                all_rf_preds.append(rf_pred)
                all_mask_rcnn_preds.extend(mask_rcnn_preds)
                
                # Get ground truth labels (assuming single label per image)
                if len(target['labels']) > 0:
                    all_true_labels.append(target['labels'][0].item())
                else:
                    all_true_labels.append(0)

        # Combine predictions (simple majority voting)
        final_predictions = []
        for i in range(len(all_rf_preds)):
            final_predictions.append(np.argmax([all_rf_preds[i]] + list(all_mask_rcnn_preds[i])))

        # Calculate metrics
        accuracy = accuracy_score(all_true_labels, final_predictions)
        precision = precision_score(all_true_labels, final_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_true_labels, final_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_true_labels, final_predictions, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_true_labels, final_predictions)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    json_file = 'annotations_in_coco.json'
    img_dir = 'SolDef_AI/Labeled'
    batch_size = 2
    num_workers = 8

    # Create dataset and data loader for evaluation
    dataset = ImageDataset(json_file=json_file, img_dir=img_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    # Initialize the ensemble model
    rf_model_path = 'RandomForestModelV1.joblib'
    mask_rcnn_model_path = 'MaskRCNNModelV1.2.pth'
    ensemble_model = EnsembleModel(rf_model_path, mask_rcnn_model_path, device)

    # Evaluate the ensemble model
    print("Evaluating ensemble model...")
    ensemble_model.evaluate(data_loader)

    torch.save(ensemble_model.state_dict(), 'BaggingEnsemble.pth')
    print("Model saved as 'BaggingEnsemble.pth'")
