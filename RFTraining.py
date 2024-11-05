import os
import json
import numpy as np
import cv2
import torch
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm

class RFImageClassifier:
    def __init__(self, n_estimators=10):
        self.rf = RandomForestClassifier(n_estimators=n_estimators)
        # Define fixed feature sizes
        self.n_histogram_bins = 32
        self.n_haralick_bins = 16
        self.n_box_features = 4  # Fixed number of box features
        
    def extract_features(self, image):
        """Extract fixed-length feature vector from image"""
        # Convert image to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        features = []
        
        # 1. Calculate histogram (fixed length)
        hist = cv2.calcHist([gray], [0], None, [self.n_histogram_bins], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize histogram
        features.extend(hist)
        
        # 2. Basic statistics (fixed length)
        mean = np.mean(gray)
        std = np.std(gray)
        median = np.median(gray)
        features.extend([mean, std, median])
        
        # 3. Texture features (fixed length)
        haralick = cv2.calcHist([gray], [0], None, [self.n_haralick_bins], [0, 256])
        haralick = haralick.flatten() / haralick.sum()  # Normalize
        features.extend(haralick)
        
        # 4. Box features (fixed length)
        if hasattr(self, 'current_boxes') and len(self.current_boxes) > 0:
            # Get features from the largest box
            areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in self.current_boxes]
            largest_box_idx = np.argmax(areas)
            box = self.current_boxes[largest_box_idx]
            
            # Calculate box features
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            aspect_ratio = width / height if height != 0 else 0
            
            box_features = [width, height, area, aspect_ratio]
        else:
            box_features = [0, 0, 0, 0]  # Default values if no boxes
            
        features.extend(box_features)
        
        return np.array(features, dtype=np.float32)

    def train(self, data_loader):
        """Train Random Forest using image features"""
        print("Extracting features for Random Forest training...")
        features = []
        labels = []
        
        for images, targets in tqdm(data_loader):
            for image, target in zip(images, targets):
                # Convert tensor to numpy array
                if hasattr(image, 'numpy'):
                    image = image.numpy()
                image = np.transpose(image, (1, 2, 0))
                
                # Convert boxes to numpy and correct format
                boxes = target['boxes'].numpy()
                if len(boxes) > 0:
                    # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
                    if boxes.shape[1] == 4:
                        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                self.current_boxes = boxes
                
                # Extract features
                image_features = self.extract_features(image)
                features.append(image_features)
                
                # Get label (assuming single label per image)
                if len(target['labels']) > 0:
                    labels.append(target['labels'][0].item())
                else:
                    labels.append(0)
        
        features = np.array(features)
        labels = np.array(labels)
        
        print(f"Feature vector shape: {features.shape}")
        print(f"Number of samples: {len(labels)}")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
        
        # Train Random Forest
        print("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        
        # Calculate training accuracy
        train_preds = self.rf.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_preds)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        
        # Evaluate on validation set
        rf_score = self.rf.score(X_val, y_val)
        print(f"Validation Accuracy: {rf_score:.4f}")

        # Make predictions on validation set
        val_preds = self.rf.predict(X_val)
        
        # Calculate validation metrics
        precision = precision_score(y_val, val_preds, average='weighted', zero_division=0)
        recall = recall_score(y_val, val_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_val, val_preds)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)

    
    def predict(self, image, boxes=None):
        """Make predictions using Random Forest"""
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if len(image.shape) == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))
            
        if boxes is not None:
            boxes = boxes.numpy() if hasattr(boxes, 'numpy') else boxes
            if len(boxes) > 0:
                # Convert [x, y, w, h] to [x1, y1, x2, y2] if needed
                if boxes.shape[1] == 4:
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        self.current_boxes = boxes if boxes is not None else []
        
        features = self.extract_features(image)
        
        # Get predictions and probabilities
        pred_label = self.rf.predict([features])[0]
        pred_proba = self.rf.predict_proba([features])[0]
        
        return {
            'predicted_class': pred_label,
            'class_probabilities': pred_proba
        }

# The ImageDataset class remains the same as in the previous code
class ImageDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        # Load the JSON annotation file
        with open(json_file) as f:
            self.data = json.load(f)
        
        # Create mappings
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
        # Get image info
        image_info = self.image_info[idx + 1]
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get annotations
        boxes, labels = [], []
        for annotation in self.annotations[idx + 1]:
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        image = self.transform(image)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx + 1])
        }

        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

if __name__ == "__main__":
    # Dataset parameters
    json_file = 'annotations_in_coco.json'
    img_dir = 'SolDef_AI/Labeled'
    batch_size = 16
    num_workers = 4
    
    # Create dataset and data loader
    dataset = ImageDataset(json_file=json_file, img_dir=img_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Create and train Random Forest model
    rf_classifier = RFImageClassifier(n_estimators=10)
    
    try:
        print("Training Random Forest model...")
        rf_classifier.train(data_loader)
        
        # Save the trained model
        joblib.dump(rf_classifier.rf, 'random_forest_classifier.joblib')
        print("Model saved successfully!")
        
        # Test the model
        #test_image, test_target = next(iter(data_loader))
        #predictions = rf_classifier.predict(test_image[0], test_target[0]['boxes'])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

