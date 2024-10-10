import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Load the image (replace 'image_path' with the actual path to your image)
image_path = 'SolDef_AI\Labeled\WIN_20220329_14_30_32_Pro.jpg'
image = cv2.imread(image_path)

# Load the annotation (replace 'annotation_file' with the actual path to your JSON annotation file)
annotation_file = 'SolDef_AI\Labeled\WIN_20220329_14_30_32_Pro.json'
with open(annotation_file) as f:
    data = json.load(f)

# Extract the shapes (polygons) from the JSON
shapes = data['shapes']

# Draw polygons on the image
for shape in shapes:
    points = np.array(shape['points'], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

# Convert BGR to RGB (OpenCV loads images in BGR format)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with annotations using Matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.axis('off')  # Hide axes
plt.show()