from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Load the image
image_path = 'WIN_20220330_16_56_08_Pro.jpg'
image = Image.open(image_path)

# Define preprocessing steps
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply the transformation
image = transform(image).unsqueeze(0)


# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)  # Output 2 classes (Good, No Good)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Load the model
model = SimpleCNN()  # Replace with your actual model
model.load_state_dict(torch.load('multi_label_model.pth'))  # Load saved model
model.eval()

# Move to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
image = image.to(device)

# Make prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output, 1)

# Map the predicted class to a label
class_labels = ['Good', 'No Good', 'Too much', 'Too less']  # Adjust based on your dataset
predicted_label = class_labels[predicted.item()]

print(f'Predicted label: {predicted_label}')
