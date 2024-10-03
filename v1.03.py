import os
import cv2
import json
import numpy as np

def parse_index_file(index_path):
    with open(index_path, 'r') as file:
        lines = file.readlines()
    image_files = []
    json_files = []
    for line in lines:
        img_file, json_file = line.strip().split()
        image_files.append(img_file)
        json_files.append(json_file)
    return image_files, json_files

# Function to parse JSON annotation file
def parse_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    objects = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']  # List of points defining the polygon
        polygon = np.array(points, np.int32)  # Convert to NumPy array for OpenCV
        polygon = polygon.reshape((-1, 1, 2))  # Reshape for OpenCV polygon drawing
        objects.append({'name': label, 'polygon': polygon})
    return objects

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Function to display images with polygons and labels
def display_image_with_annotations(image, annotations):
    for annotation in annotations:
        label = annotation['name']
        polygon = annotation['polygon']
        # Draw polygon
        cv2.polylines(image, [polygon], isClosed=True, color=(255, 0, 0), thickness=2)
        # Calculate the position to place the label
        text_position = tuple(polygon[0][0])  # Use the first point of the polygon
        cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # Display the image with annotations
    cv2.imshow('Image with Annotations', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

index_file = 'solindex.txt'
image_files, json_files = parse_index_file(index_file)

for img_file, json_file in zip(image_files, json_files):
    print(os.path.join('SolDef_AI/Labeled', img_file))
    print(os.path.join('SolDef_AI/Labeled', json_file))
    image = load_image(os.path.join('SolDef_AI/Labeled', img_file))
    annotations = parse_json(os.path.join('SolDef_AI/Labeled', json_file))
    display_image_with_annotations(image, annotations)
