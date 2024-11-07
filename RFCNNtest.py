import os
import json
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from pycocotools.mask import encode, decode
from torchvision.models.detection import maskrcnn_resnet50_fpn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

class RFMaskRCNNEnsemble:
    def __init__(self, n_estimators=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights='DEFAULT')
        self.mask_rcnn.to(self.device)
        self.rf = RandomForestClassifier(n_estimators=n_estimators)
         # Takes layers from Mask R-CNN backbone except the last one for feature extraction
        self.feature_extractor = torch.nn.Sequential(*list(self.mask_rcnn.backbone.body.children())[:-1])
        
    def extract_features(self, images):
        """Extract features using Mask R-CNN backbone"""
        self.mask_rcnn.eval()
        features = []
        
        with torch.no_grad():
            for image in images:
                # Get feature maps from backbone
                feat = self.feature_extractor(image.unsqueeze(0).to(self.device))
                # Global average pooling
                feat = torch.mean(feat, dim=[2, 3])
                features.append(feat.cpu().numpy().flatten())
                
        return np.array(features)
    
    def train(self, data_loader):
        """Train both Mask R-CNN and Random Forest"""
        # First train Mask R-CNN
        self.train_mask_rcnn(data_loader)
        
        # Then extract features and train Random Forest
        self.train_random_forest(data_loader)
        
    def train_mask_rcnn(self, data_loader, num_epochs=10, learning_rate=0.005):
        # Uses the full data from data_loader:
        # - images: The actual images
        # - targets: Dictionary containing:
        #   - boxes: Bounding boxes
        #   - labels: Class labels
        #   - masks: Segmentation masks
        #   - image_id: Image identifiers
        #   - area: Area of masks
        """Train Mask R-CNN"""
        self.mask_rcnn.train()
        optimizer = torch.optim.SGD(self.mask_rcnn.parameters(), lr=learning_rate, momentum=0.9)
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            running_loss = 0.0
            
            progress_bar = tqdm(data_loader, total=len(data_loader))
            for images, targets in progress_bar:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                optimizer.zero_grad()
                loss_dict = self.mask_rcnn(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                losses.backward()
                optimizer.step()
                
                running_loss += losses.item()
                progress_bar.set_description(f"Loss: {losses.item():.4f}")
            
            epoch_loss = running_loss / len(data_loader)
            print(f"Epoch {epoch+1} Loss: {epoch_loss:.4f}")
    
    def train_random_forest(self, data_loader):
        # Uses:
        # - images: To extract features through Mask R-CNN backbone
        # - targets: Only uses the first label from each target
        #   (labels[0] if exists, else 0)

        """Train Random Forest using features from Mask R-CNN"""
        print("Extracting features for Random Forest training...")
        features = []   # Features extracted from Mask R-CNN backbone
        labels = []     # Single label per image
        
        self.mask_rcnn.eval()
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                batch_features = self.extract_features(images)
                features.extend(batch_features)
                
                # Get labels from targets
                batch_labels = [t['labels'][0].item() if len(t['labels']) > 0 else 0 for t in targets]
                labels.extend(batch_labels)
        
        features = np.array(features)
        labels = np.array(labels)
        
        # Split data for Random Forest
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        
        # Evaluate on training set
        train_accuracy = self.rf.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Evaluate
        rf_score = self.rf.score(X_val, y_val)
        print(f"Random Forest Validation Score: {rf_score:.4f}")
    
    def predict(self, image):
        """Make predictions using both models"""
        self.mask_rcnn.eval()
        
        with torch.no_grad():
            # Get Mask R-CNN predictions
            image_tensor = image.to(self.device)
            mask_rcnn_pred = self.mask_rcnn([image_tensor])[0]
            
            # Extract features for Random Forest
            features = self.extract_features([image])
            rf_pred = self.rf.predict_proba(features)
            
            # Combine predictions
            combined_pred = {
                'mask_rcnn': mask_rcnn_pred,
                'rf_probabilities': rf_pred[0]
            }
            
            return combined_pred
        
    def save_model(self, save_path='ensemble_model.pth'):
        """
        Save both Mask R-CNN and Random Forest models in a single file
        
        Args:
            save_path (str): Path to save the combined model
        """
        # Create a state dictionary to save both models
        model_state = {
            'mask_rcnn_state': self.mask_rcnn.state_dict(),
            'rf_model': self.rf
        }
        
        # Save to a single file
        torch.save(model_state, save_path)
        print(f"Ensemble model saved to {save_path}")
    
    def load_model(self, load_path='ensemble_model.pth'):
        """
        Load both Mask R-CNN and Random Forest models from a single file
        
        Args:
            load_path (str): Path to load the combined model
        """
        # Load the state dictionary
        model_state = torch.load(load_path, map_location=self.device, weights_only=True)
        
        # Restore Mask R-CNN model
        self.mask_rcnn.load_state_dict(model_state['mask_rcnn_state'])
        self.mask_rcnn.to(self.device)
        
        # Restore Random Forest model
        self.rf = model_state['rf_model']
        
        print(f"Ensemble model loaded from {load_path}")

    def evaluate_metrics(self, data_loader):
        """Evaluate the ensemble model with accuracy, precision, recall, F1 score, and confusion matrix"""
        print("Evaluating the ensemble model...")

        all_true_labels = []
        all_rf_predictions = []

        self.mask_rcnn.eval()
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                # Extract true labels from targets
                true_labels = [t['labels'][0].item() if len(t['labels']) > 0 else 0 for t in targets]
                all_true_labels.extend(true_labels)

                # Get Random Forest predictions
                batch_features = self.extract_features(images)
                rf_predictions = self.rf.predict(batch_features)
                all_rf_predictions.extend(rf_predictions)

        # Convert lists to arrays for metrics calculation
        all_true_labels = np.array(all_true_labels)
        all_rf_predictions = np.array(all_rf_predictions)

        # Calculate metrics
        accuracy = accuracy_score(all_true_labels, all_rf_predictions)
        precision = precision_score(all_true_labels, all_rf_predictions, average='weighted')
        recall = recall_score(all_true_labels, all_rf_predictions, average='weighted')
        f1 = f1_score(all_true_labels, all_rf_predictions, average='weighted')
        conf_matrix = confusion_matrix(all_true_labels, all_rf_predictions)

        print(f"Testing Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Plot confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=True, yticklabels=True)
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix for Random Forest Predictions")
        plt.show()

class MaskRCNNDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, split="train"):
        self.img_dir = img_dir
        self.transform = transform
        self.split = split
        
        # Load the JSON annotation file
        with open(json_file) as f:
            self.data = json.load(f)
        
        # Create a mapping from image_id to annotations
        self.image_info = {image['id']: image for image in self.data['images']}
        self.annotations = {image['id']: [] for image in self.data['images']}
        
        for annotation in self.data['annotations']:
            self.annotations[annotation['image_id']].append(annotation)
        
        # Get a list of all image_ids
        all_image_ids = list(self.image_info.keys())
        
        # Split the data into train and test
        train_ids, test_ids = train_test_split(all_image_ids, test_size=0.2, random_state=42)
        
        # Assign images to either the train or test set based on the split
        if self.split == "train":
            self.image_ids = train_ids
        elif self.split == "test":
            self.image_ids = test_ids
        else:
            raise ValueError("split must be 'train' or 'test'")
        
        # Default transform if none is provided
        if transform is None:
            self.transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.image_info[image_id]
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        
        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare targets
        masks, boxes, labels, areas = [], [], [], []
        
        for annotation in self.annotations[image_id]:
            if len(annotation['segmentation']) == 0:
                continue
                
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for poly in annotation['segmentation']:
                cv2.fillPoly(mask, [np.array(poly).reshape((-1, 1, 2)).astype(np.int32)], 1)
            
            masks.append(mask)
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])
            areas.append(annotation['area'])

        # Convert to tensors
        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        else:
            masks = torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.uint8)
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        image = self.transform(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([image_id]),
            'area': areas
        }

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Dataset parameters
    json_file = 'annotations_in_coco.json'
    img_dir = 'SolDef_AI/Labeled'
    batch_size = 8
    num_workers = 8
    
    # Create dataset for training and testing
    train_dataset = MaskRCNNDataset(json_file=json_file, img_dir=img_dir, split="train")
    test_dataset = MaskRCNNDataset(json_file=json_file, img_dir=img_dir, split="test")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Create and train ensemble model
    ensemble = RFMaskRCNNEnsemble(n_estimators=10)
    
    try:
        print("Training ensemble model...")
        ensemble.train(train_loader)
        
        # Save the combined model
        ensemble.save_model('ensemble_model.pth')
        print("Models saved successfully!")
        
        # Evaluate the ensemble model on the test set
        ensemble.evaluate_metrics(test_loader)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
