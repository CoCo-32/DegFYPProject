import os
import json
import numpy as np
import torch
import joblib
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MRCNNTraining import MaskRCNNDataset, collate_fn, calculate_iou
from RFTraining import RFImageClassifier, ImageDataset

class EnsembleClassifier:
    def __init__(self, mask_rcnn_model, rf_model, device):
        self.mask_rcnn_model = mask_rcnn_model
        self.rf_model = rf_model
        self.device = device

        # Create the bagging ensemble
        self.ensemble = BaggingClassifier(
            base_estimator=None,
            n_estimators=10,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=True,
            bootstrap_features=False,
            oob_score=True,
            n_jobs=-1,
            random_state=42
        )

    def train(self, train_loader, val_loader):
        # Train the individual models
        self.train_mask_rcnn(train_loader)
        self.train_random_forest(train_loader)

        # Combine the models into the bagging ensemble
        X_train, y_train = self.get_ensemble_features(train_loader)
        X_val, y_val = self.get_ensemble_features(val_loader)

        self.ensemble.fit(X_train, y_train)

        # Evaluate the ensemble
        ensemble_val_preds = self.ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, ensemble_val_preds)
        precision = precision_score(y_val, ensemble_val_preds, average='weighted', zero_division=0)
        recall = recall_score(y_val, ensemble_val_preds, average='weighted', zero_division=0)
        f1 = f1_score(y_val, ensemble_val_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_val, ensemble_val_preds)

        print(f"Ensemble Accuracy: {accuracy:.4f}")
        print(f"Ensemble Precision: {precision:.4f}")
        print(f"Ensemble Recall: {recall:.4f}")
        print(f"Ensemble F1 Score: {f1:.4f}")
        print("Ensemble Confusion Matrix:")
        print(conf_matrix)

    def train_mask_rcnn(self, train_loader):
        # Train the Mask R-CNN model
        self.mask_rcnn_model.train()
        for epoch in range(10):
            for images, targets in train_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.mask_rcnn_model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                self.mask_rcnn_model.optimizer.zero_grad()
                losses.backward()
                self.mask_rcnn_model.optimizer.step()

    def train_random_forest(self, train_loader):
        # Train the Random Forest model
        self.rf_model.train(train_loader)

    def get_ensemble_features(self, data_loader):
        all_features = []
        all_labels = []

        for images, targets in data_loader:
            mask_rcnn_preds = self.mask_rcnn_model(images)
            rf_preds = [self.rf_model.predict(image, target['boxes']) for image, target in zip(images, targets)]

            for mask_rcnn_pred, rf_pred, target in zip(mask_rcnn_preds, rf_preds, targets):
                # Combine features from the two models
                features = np.concatenate((mask_rcnn_pred['boxes'].flatten(),
                                          mask_rcnn_pred['labels'].flatten(),
                                          mask_rcnn_pred['scores'].flatten(),
                                          np.array([rf_pred['predicted_class']])))
                all_features.append(features)
                all_labels.append(target['labels'][0].item())

        return np.array(all_features), np.array(all_labels)

    def predict(self, image, boxes=None):
        mask_rcnn_pred = self.mask_rcnn_model([image.to(self.device)])[0]
        rf_pred = self.rf_model.predict(image, boxes)

        # Combine the predictions
        features = np.concatenate((mask_rcnn_pred['boxes'].flatten(),
                                  mask_rcnn_pred['labels'].flatten(),
                                  mask_rcnn_pred['scores'].flatten(),
                                  np.array([rf_pred['predicted_class']])))
        ensemble_pred = self.ensemble.predict([features])[0]
        ensemble_proba = self.ensemble.predict_proba([features])[0]

        return {
            'predicted_class': ensemble_pred,
            'class_probabilities': ensemble_proba
        }

if __name__ == "__main__":
    # Load the pre-trained Mask R-CNN and Random Forest models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_rcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    mask_rcnn_model.load_state_dict(torch.load('MaskRCNNModelV1.2.pth'))
    mask_rcnn_model.to(device)

    rf_model = RFImageClassifier(n_estimators=100)
    rf_model.rf = joblib.load('RandomForestModelV1.joblib')

    # Create the ensemble classifier
    ensemble = EnsembleClassifier(mask_rcnn_model, rf_model, device)

    # Load the training and validation data
    train_dataset = ImageDataset(json_file='annotations_in_coco.json', img_dir='SolDef_AI/Labeled')
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

    val_dataset = ImageDataset(json_file='annotations_in_coco.json', img_dir='SolDef_AI/Labeled')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Train the ensemble model
    ensemble.train(train_loader, val_loader)

    # Test the ensemble model
    test_image, test_target = next(iter(val_loader))
    predictions = ensemble.predict(test_image[0], test_target[0]['boxes'])
    print("Test predictions:")
    print(f"Predicted class: {predictions['predicted_class']}")
    print(f"Class probabilities: {predictions['class_probabilities']}")