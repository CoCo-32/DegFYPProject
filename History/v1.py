import os
import cv2
import xml.etree.ElementTree as ET

def parse_index_file(index_path):
    with open(index_path, 'r') as file:
        lines = file.readlines()
    image_files = []
    xml_files = []
    for line in lines:
        img_file, xml_file = line.strip().split()
        image_files.append(img_file)
        xml_files.append(xml_file)
    return image_files, xml_files

def parse_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        bbox = {
            'xmin': int(bndbox.find('xmin').text),
            'ymin': int(bndbox.find('ymin').text),
            'xmax': int(bndbox.find('xmax').text),
            'ymax': int(bndbox.find('ymax').text),
        }
        objects.append({'name': name, 'bbox': bbox})
    return objects

def load_image(image_path):
    image = cv2.imread(image_path)
    return image

# Function to display images with bounding boxes and labels
def display_image_with_annotations(image, annotations):
    for annotation in annotations:
        label = annotation['name']
        bbox = annotation['bbox']
        # Draw bounding box
        cv2.rectangle(image, (bbox['xmin'], bbox['ymin']), (bbox['xmax'], bbox['ymax']), (255, 0, 0), 2)
        # Put label text above the bounding box
        cv2.putText(image, label, (bbox['xmin'], bbox['ymin'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    # Display the image with annotations
    cv2.imshow('Image with Annotations', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

index_file = 'index.txt'
image_files, xml_files = parse_index_file(index_file)

for img_file, xml_file in zip(image_files, xml_files):
    print(os.path.join('Sample', img_file))
    image = load_image(os.path.join('Sample', img_file))
    annotations = parse_xml(os.path.join('Sample', xml_file))
    display_image_with_annotations(image, annotations)
    # Now you have the image and the annotations (e.g., bounding boxes and labels)
