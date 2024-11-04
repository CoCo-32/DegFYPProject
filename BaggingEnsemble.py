import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2
import joblib
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from RFTraining import ImageDataset

class BaggingEnsemble:
    def __init__(self, maskrcnn_path, rf_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load Mask R-CNN
        self.maskrcnn = maskrcnn_resnet50_fpn(weights=None)
        self.maskrcnn.load_state_dict(torch.load(maskrcnn_path, map_location=self.device))
        self.maskrcnn.to(self.device)
        self.maskrcnn.eval()
        
        # Load Random Forest
        self.rf = joblib.load(rf_path)
        
        # Define transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Feature extraction parameters (matching RF training)
        self.n_histogram_bins = 32
        self.n_haralick_bins = 16
        
    def extract_features(self, image, boxes):
        """Extract features for Random Forest prediction"""
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        features = []
        
        # Extract histogram features
        hist = cv2.calcHist([gray], [0], None, [self.n_histogram_bins], [0, 256])
        hist = hist.flatten() / hist.sum()
        features.extend(hist)
        
        # Basic statistics
        features.extend([
            np.mean(gray),
            np.std(gray),
            np.median(gray)
        ])
        
        # Texture features
        haralick = cv2.calcHist([gray], [0], None, [self.n_haralick_bins], [0, 256])
        haralick = haralick.flatten() / haralick.sum()
        features.extend(haralick)
        
        # Box features
        if len(boxes) > 0:
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
            largest_box = boxes[np.argmax(areas)]
            width = largest_box[2] - largest_box[0]
            height = largest_box[3] - largest_box[1]
            area = width * height
            aspect_ratio = width / height if height != 0 else 0
            box_features = [width, height, area, aspect_ratio]
        else:
            box_features = [0, 0, 0, 0]
        
        features.extend(box_features)
        return np.array(features, dtype=np.float32)
    
    def predict(self, image, confidence_threshold=0.5, ensemble_method='weighted_average', weights=(0.6, 0.4)):
        """
        Make ensemble predictions using both models
        
        Args:
            image: RGB image array
            confidence_threshold: threshold for Mask R-CNN predictions
            ensemble_method: 'weighted_average' or 'max_confidence'
            weights: tuple of (maskrcnn_weight, rf_weight) for weighted averaging
            
        Returns:
            Dictionary containing combined predictions and individual model outputs
        """
        # Prepare image for Mask R-CNN
        if not isinstance(image, torch.Tensor):
            transformed_image = self.transform(image)
        else:
            transformed_image = image
            
        # Get Mask R-CNN predictions
        with torch.no_grad():
            maskrcnn_pred = self.maskrcnn([transformed_image.to(self.device)])[0]
            
        # Filter predictions by confidence
        mask_scores = maskrcnn_pred['scores'].cpu()
        mask_boxes = maskrcnn_pred['boxes'].cpu()
        mask_labels = maskrcnn_pred['labels'].cpu()
        mask_masks = maskrcnn_pred['masks'].cpu()
        
        confident_idx = mask_scores >= confidence_threshold
        confident_boxes = mask_boxes[confident_idx]
        confident_labels = mask_labels[confident_idx]
        confident_scores = mask_scores[confident_idx]
        confident_masks = mask_masks[confident_idx]
        
        # Get Random Forest predictions
        rf_features = self.extract_features(image, confident_boxes.numpy())
        rf_pred = self.rf.predict_proba([rf_features])[0]
        
        # Combine predictions based on ensemble method
        if ensemble_method == 'weighted_average':
            # Convert Mask R-CNN outputs to probability distribution
            maskrcnn_probs = torch.zeros(len(self.rf.classes_))
            for label, score in zip(confident_labels, confident_scores):
                maskrcnn_probs[label.item()] = max(maskrcnn_probs[label.item()], score.item())
            
            # Combine probabilities using weights
            combined_probs = (weights[0] * maskrcnn_probs + 
                            weights[1] * torch.tensor(rf_pred))
            final_prediction = int(torch.argmax(combined_probs))
            
        elif ensemble_method == 'max_confidence':
            # Take prediction from model with highest confidence
            maskrcnn_conf = torch.max(confident_scores) if len(confident_scores) > 0 else 0
            rf_conf = np.max(rf_pred)
            
            if maskrcnn_conf > rf_conf:
                final_prediction = confident_labels[0].item()
            else:
                final_prediction = np.argmax(rf_pred)
        
        return {
            'final_prediction': final_prediction,
            'maskrcnn_boxes': confident_boxes,
            'maskrcnn_labels': confident_labels,
            'maskrcnn_scores': confident_scores,
            'maskrcnn_masks': confident_masks,
            'rf_probabilities': rf_pred
        }
    
    def evaluate(self, data_loader, ensemble_method='weighted_average', weights=(0.6, 0.4)):
        """Evaluate the ensemble model on a dataset"""
        all_preds = []
        all_labels = []
        
        for images, targets in data_loader:
            for image, target in zip(images, targets):
                # Get ensemble prediction
                pred_result = self.predict(
                    image, 
                    ensemble_method=ensemble_method,
                    weights=weights
                )
                
                # Get ground truth label (assuming single label per image)
                true_label = target['labels'][0].item() if len(target['labels']) > 0 else 0
                
                all_preds.append(pred_result['final_prediction'])
                all_labels.append(true_label)
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': conf_matrix
        }
    
def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Example usage
    maskrcnn_path = 'MaskRCNNModelV1.2.pth'
    rf_path = 'RandomForestModelV1.joblib'
    
    # Create ensemble
    ensemble = BaggingEnsemble(
        maskrcnn_path=maskrcnn_path,
        rf_path=rf_path,
        device='cuda'
    )
    
    # Create dataset and dataloader (using your existing Dataset class)
    dataset = ImageDataset(
        json_file='annotations_in_coco.json',
        img_dir='SolDef_AI/Labeled'
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    # Evaluate ensemble with different methods
    print("Evaluating weighted average ensemble...")
    weighted_results = ensemble.evaluate(
        data_loader,
        ensemble_method='weighted_average',
        weights=(0.6, 0.4)
    )
    
    print("\nWeighted Average Results:")
    print(f"Accuracy: {weighted_results['accuracy']:.4f}")
    print(f"Precision: {weighted_results['precision']:.4f}")
    print(f"Recall: {weighted_results['recall']:.4f}")
    print(f"F1 Score: {weighted_results['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(weighted_results['confusion_matrix'])
    
    print("\nEvaluating max confidence ensemble...")
    max_conf_results = ensemble.evaluate(
        data_loader,
        ensemble_method='max_confidence'
    )
    
    print("\nMax Confidence Results:")
    print(f"Accuracy: {max_conf_results['accuracy']:.4f}")
    print(f"Precision: {max_conf_results['precision']:.4f}")
    print(f"Recall: {max_conf_results['recall']:.4f}")
    print(f"F1 Score: {max_conf_results['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(max_conf_results['confusion_matrix'])

if __name__ == "__main__":
    main()