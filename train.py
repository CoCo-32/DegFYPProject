import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
from dataset import ObjectDetectionDataset

def load_model(num_classes):
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def visualize_detections(image, predictions, threshold=0.5):
    image = image.permute(1, 2, 0).cpu().numpy()
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
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Set the number of classes (including background)
    num_classes = 2  # Modify this based on your dataset

    # Load the model
    model = load_model(num_classes)
    model.to(device)

    # Prepare the dataset
    dataset = ObjectDetectionDataset(txt_file='solindex.txt')
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Set up the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Train the model
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Print losses to debug
            print(f"Loss dict: {loss_dict}")
            print(f"Total loss: {losses.item()}")

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

    print(f"Loss: {losses.item()}")

    # Save the model
    torch.save(model.state_dict(), 'faster_rcnn_custom_model.pth')

    # Test the model on a sample image
    model.eval()
    sample_image, _ = dataset[0]
    sample_image = sample_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(sample_image)[0]

    visualize_detections(sample_image.squeeze(0), predictions)

if __name__ == "__main__":
    main()