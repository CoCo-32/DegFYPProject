import torch
from torchvision import transforms as T
import cv2
import numpy as np
from RFCNNTraining import RFMaskRCNNEnsemble

# Load the saved ensemble model
ensemble = RFMaskRCNNEnsemble(n_estimators=10)
ensemble.load_model('ensemble_model.pth')

# Define a function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

# Define the image path
image_path = '.jpg'
image_tensor = preprocess_image(image_path)

# Classify the image
prediction = ensemble.predict(image_tensor.unsqueeze(0))  # Add batch dimension

# Output the predictions
print("Mask R-CNN Prediction:", prediction['mask_rcnn'])
print("Random Forest Probabilities:", prediction['rf_probabilities'])
