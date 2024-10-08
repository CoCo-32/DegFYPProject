import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

def load_model(num_classes):
    # Load a pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Modify the classifier to match your number of classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    
    return model

def prepare_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor.unsqueeze(0)

def detect_objects(model, image_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
    
    return predictions[0]

def visualize_detections(image, predictions, threshold=0.5):
    image = image.squeeze().permute(1, 2, 0).numpy()
    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    
    boxes = predictions['boxes'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            x1, y1, x2, y2 = box
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2))
            plt.text(x1, y1, f'Class: {label}, Score: {score:.2f}', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.axis('off')
    plt.show()

def main():
    # Set the number of classes (including background)
    num_classes = 91  # COCO dataset has 90 classes + 1 for background

    # Load the model
    model = load_model(num_classes)

    # Prepare an image
    image_path = "images.jpeg"  # Replace with your image path
    image_tensor = prepare_image(image_path)

    # Perform object detection
    predictions = detect_objects(model, image_tensor)

    # Visualize the results
    visualize_detections(image_tensor, predictions)

if __name__ == "__main__":
    main()