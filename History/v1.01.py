import os
import cv2

def parse_index_file(index_path):
    with open(index_path, 'r') as file:
        lines = file.readlines()
    image_files = []
    annot_files = []
    for line in lines:
        img_file, annot_file = line.strip().split()
        image_files.append(img_file)
        annot_files.append(annot_file)
    return image_files, annot_files

def parse_txt(annot_file):
    objects = []
    with open(annot_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = line.strip().split()
            label = parts[4]
            bbox = {
                'xmin': int(parts[0]),
                'ymin': int(parts[1]),
                'xmax': int(parts[2]),
                'ymax': int(parts[3]),
            }
            objects.append({'name': label, 'bbox': bbox})
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

index_file = 'trainval.txt'
image_files, annot_files = parse_index_file(index_file)

for img_file, annot_file in zip(image_files, annot_files):
    # Split and add '_temp' to the image path
    img_name, img_extension = os.path.splitext(img_file)
    new_img_file = f"{img_name}_test{img_extension}"
    image = load_image(new_img_file)
    annotations = parse_txt(annot_file)
    
    print(img_file, annot_file)
    display_image_with_annotations(image, annotations)
    # Now you have the image and the annotations (e.g., bounding boxes and labels)