import os
import cv2
import torch
import joblib
import numpy as np
import torchvision.transforms as T
from RFTraining import RFImageClassifier  # Import from your original file

def test_model(image_path, model_path='random_forest_classifier.joblib'):
    # Load the trained model
    rf_classifier = RFImageClassifier()
    rf_classifier.rf = joblib.load(model_path)
    
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply the same transforms as during training
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    
    # Make prediction
    predictions = rf_classifier.predict(image_tensor)
    
    return predictions

if __name__ == "__main__":
    # Example usage
    test_image_path = "SolDef_AI/Labeled/WIN_20220329_14_30_32_Pro.jpg"  # Replace with your test image path
    
    try:
        predictions = test_model(test_image_path)
        
        print("Prediction Results:")
        print(f"Predicted class: {predictions['predicted_class']}")
        print(f"Class probabilities: {predictions['class_probabilities']}")
        
        # If you have a class mapping, you can convert class indices to names
        # class_names = {0: 'class1', 1: 'class2', ...}  # Add your class mapping
        # predicted_class_name = class_names[predictions['predicted_class']]
        # print(f"Predicted class name: {predicted_class_name}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def test_multiple_images(image_dir, model_path='random_forest_classifier.joblib'):
    # Load the model once
    rf_classifier = RFImageClassifier()
    rf_classifier.rf = joblib.load(model_path)
    
    # Process all images in directory
    results = {}
    for image_name in os.listdir(image_dir):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            try:
                predictions = test_model(image_path, model_path)
                results[image_name] = predictions
            except Exception as e:
                print(f"Error processing {image_name}: {str(e)}")
    
    return results

# Usage
test_dir = "path/to/test/images"
results = test_multiple_images(test_dir)

for image_name, predictions in results.items():
    print(f"\nResults for {image_name}:")
    print(f"Predicted class: {predictions['predicted_class']}")
    print(f"Class probabilities: {predictions['class_probabilities']}")


from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(image_dir, true_labels, model_path='random_forest_classifier.joblib'):
    predictions = []
    rf_classifier = RFImageClassifier()
    rf_classifier.rf = joblib.load(model_path)
    
    for image_path, true_label in zip(image_paths, true_labels):
        pred = test_model(image_path, model_path)
        predictions.append(pred['predicted_class'])
    
    accuracy = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions)
    
    print(f"Accuracy: {accuracy}")
    print("\nClassification Report:")
    print(report)

# Add to test_model function
print(f"Image shape: {image.shape}")
print(f"Feature vector shape: {rf_classifier.extract_features(image).shape}")