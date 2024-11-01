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
from tqdm import tqdm

class RFMaskRCNNEnsemble:
    def __init__(self, n_estimators=100):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights=None)
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
                feat = self.feature_extractor(image.unsqueeze(0).to(self.device))
                feat = torch.mean(feat, dim=[2, 3])
                features.append(feat.cpu().numpy().flatten())
                
        return np.array(features)
    
    def train(self, data_loader):
        """Train both Mask R-CNN and Random Forest"""
        self.train_mask_rcnn(data_loader)
        self.train_random_forest(data_loader)
        
    def train_mask_rcnn(self, data_loader, num_epochs=10, learning_rate=0.005):
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
        print("Extracting features for Random Forest training...")
        features = []
        labels = []
        
        self.mask_rcnn.eval()
        with torch.no_grad():
            for images, targets in tqdm(data_loader):
                batch_features = self.extract_features(images)
                features.extend(batch_features)
                
                batch_labels = [t['labels'][0].item() if len(t['labels']) > 0 else 0 for t in targets]
                labels.extend(batch_labels)
        
        features = np.array(features)
        labels = np.array(labels)
        
        X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2)
        
        print("Training Random Forest...")
        self.rf.fit(X_train, y_train)
        
        rf_score = self.rf.score(X_val, y_val)
        print(f"Random Forest Validation Score: {rf_score:.4f}")
    
    def predict(self, image):
        self.mask_rcnn.eval()
        
        with torch.no_grad():
            image_tensor = image.to(self.device)
            mask_rcnn_pred = self.mask_rcnn([image_tensor])[0]
            
            features = self.extract_features([image])
            rf_pred = self.rf.predict_proba(features)
            
            combined_pred = {
                'mask_rcnn': mask_rcnn_pred,
                'rf_probabilities': rf_pred[0]
            }
            
            return combined_pred

class CombinedModelWrapper:
    def __init__(self, mask_rcnn_model_path, rf_model_path, rf_threshold=0.5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mask_rcnn = maskrcnn_resnet50_fpn(weights=None)
        self.mask_rcnn.load_state_dict(torch.load(mask_rcnn_model_path, map_location=self.device))
        self.mask_rcnn.to(self.device)
        self.mask_rcnn.eval()
        
        import joblib
        self.rf = joblib.load(rf_model_path)
        
        self.feature_extractor = torch.nn.Sequential(*list(self.mask_rcnn.backbone.body.children())[:-1])
        self.rf_threshold = rf_threshold

    def extract_features(self, image):
        with torch.no_grad():
            feat = self.feature_extractor(image.unsqueeze(0).to(self.device))
            feat = torch.mean(feat, dim=[2, 3])
        return feat.cpu().numpy().flatten()

    def predict(self, image):
        with torch.no_grad():
            mask_rcnn_pred = self.mask_rcnn([image.to(self.device)])[0]
            
            features = self.extract_features(image)
            rf_prob = self.rf.predict_proba([features])[0]
            
            combined_detections = {
                'boxes': [],
                'labels': [],
                'masks': [],
                'scores': []
            }
            
            for i, score in enumerate(mask_rcnn_pred['scores']):
                if rf_prob[mask_rcnn_pred['labels'][i].item()] >= self.rf_threshold:
                    combined_detections['boxes'].append(mask_rcnn_pred['boxes'][i].cpu())
                    combined_detections['labels'].append(mask_rcnn_pred['labels'][i].cpu())
                    combined_detections['masks'].append(mask_rcnn_pred['masks'][i].cpu())
                    combined_detections['scores'].append(score.cpu())
                    
            combined_detections['boxes'] = torch.stack(combined_detections['boxes']) if combined_detections['boxes'] else torch.empty((0, 4))
            combined_detections['labels'] = torch.tensor(combined_detections['labels'], dtype=torch.int64)
            combined_detections['masks'] = torch.stack(combined_detections['masks']) if combined_detections['masks'] else torch.empty((0, image.shape[1], image.shape[2]))
            combined_detections['scores'] = torch.tensor(combined_detections['scores'])
        
        return combined_detections

    def save_combined_model(self, path):
        import joblib
        joblib.dump({
            'mask_rcnn_state_dict': self.mask_rcnn.state_dict(),
            'rf_model': self.rf,
            'rf_threshold': self.rf_threshold
        }, path)
        print(f"Combined model saved to {path}")

    @staticmethod
    def load_combined_model(path):
        import joblib
        data = joblib.load(path)
        combined_model = CombinedModelWrapper(mask_rcnn_model_path=None, rf_model_path=None)
        combined_model.mask_rcnn.load_state_dict(data['mask_rcnn_state_dict'])
        combined_model.rf = data['rf_model']
        combined_model.rf_threshold = data['rf_threshold']
        return combined_model

class MaskRCNNDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        
        with open(json_file) as f:
            self.data = json.load(f)
        
        self.image_info = {image['id']: image for image in self.data['images']}
        self.annotations = {image['id']: [] for image in self.data['images']}
        
        for annotation in self.data['annotations']:
            self.annotations[annotation['image_id']].append(annotation)
            
        if transform is None:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data['images'])

    def __getitem__(self, idx):
        image_info = self.image_info[idx + 1]
        image_path = os.path.join(self.img_dir, image_info['file_name'])
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        masks, boxes, labels, areas = [], [], [], []
        
        for annotation in self.annotations[idx + 1]:
            if len(annotation['segmentation']) == 0:
                continue
                
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            for poly in annotation['segmentation']:
                cv2.fillPoly(mask, [np.array(poly).reshape((-1, 1, 2)).astype(np.int32)], 1)
            
            masks.append(mask)
            boxes.append(annotation['bbox'])
            labels.append(annotation['category_id'])
            areas.append(annotation['area'])

        if len(masks) > 0:
            masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
        else:
            masks, boxes, labels, areas = torch.empty((0,)), torch.empty((0,)), torch.empty((0,)), torch.empty((0,))
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'area': areas,
            'image_id': torch.tensor([idx])
        }
        
        image = self.transform(image)
        
        return image, target

# Example usage
# combined_model = CombinedModelWrapper('mask_rcnn_model.pth', 'random_forest_model.joblib', rf_threshold=0.5)
# combined_model.save_combined_model('combined_model.joblib')

# loaded_model = CombinedModelWrapper.load_combined_model('combined_model.joblib')
# image, _ = next(iter(data_loader))
# prediction = loaded_model.predict(image[0])
# print("Combined prediction:", prediction)
