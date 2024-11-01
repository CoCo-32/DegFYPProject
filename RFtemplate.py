import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def extract_image_features(image, annotations):
    """
    Extract features from image and annotations
    
    Parameters:
    image: np.array of shape (height, width, channels)
    annotations: dict containing bounding boxes, labels, or other metadata
    
    Returns:
    np.array of features
    """
    features = []
    
    # Basic image statistics
    features.extend([
        np.mean(image),  # Average pixel value
        np.std(image),   # Standard deviation of pixel values
        np.max(image),   # Maximum pixel value
        np.min(image)    # Minimum pixel value
    ])
    
    # Color channel statistics (if RGB)
    if len(image.shape) == 3:
        for channel in range(image.shape[2]):
            channel_data = image[:,:,channel]
            features.extend([
                np.mean(channel_data),
                np.std(channel_data)
            ])
    
    # Annotation-based features
    if 'bounding_box' in annotations:
        x, y, w, h = annotations['bounding_box']
        features.extend([
            w * h,          # Area
            w / h,          # Aspect ratio
            x / image.shape[1],  # Normalized x position
            y / image.shape[0]   # Normalized y position
        ])
    
    if 'keypoints' in annotations:
        keypoints = annotations['keypoints']
        # Calculate distances between keypoints
        for i in range(len(keypoints)):
            for j in range(i+1, len(keypoints)):
                dist = np.sqrt(np.sum((keypoints[i] - keypoints[j])**2))
                features.append(dist)
    
    return np.array(features)

def train_rf_classifier(images, annotations, labels):
    """
    Train a Random Forest classifier on image data with annotations
    """
    # Extract features for all images
    X = np.array([extract_image_features(img, ann) 
                  for img, ann in zip(images, annotations)])
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = rf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance plot
    feature_importance = rf.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance')
    plt.show()
    
    return rf

# Example usage
if __name__ == "__main__":
    # Example synthetic data
    n_samples = 100
    image_shape = (64, 64, 3)
    
    # Generate synthetic images and annotations
    images = np.random.rand(n_samples, *image_shape)
    annotations = [
        {
            'bounding_box': np.random.rand(4),
            'keypoints': np.random.rand(5, 2)
        }
        for _ in range(n_samples)
    ]
    labels = np.random.choice(['cat', 'dog'], n_samples)
    
    # Train classifier
    classifier = train_rf_classifier(images, annotations, labels)