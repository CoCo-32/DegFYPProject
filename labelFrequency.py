import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Load the original annotation file and declare output file
json_dir = 'SolDef_AI/Labeled'  # Replace with your file path

# Initialize COCO structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": []
}

# Unique ID counters for images and annotations
image_id = 1
annotation_id = 1

# Manually define category IDs based on the label
category_id_mapping = {
    "good": 1,
    "no_good": 2,
    "exc_solder": 3,
    "poor_solder": 4,
    "spike": 5
}

# Dictionary to hold label counts
label_counts = {label: 0 for label in category_id_mapping.keys()}

# Add the categories to the COCO structure based on the mapping
for label, category_id in category_id_mapping.items():
    category_info = {
        "id": category_id,
        "name": label,
        "supercategory": "none"
    }
    coco_output["categories"].append(category_info)

# Loop over all JSON files in the directory
for filename in os.listdir(json_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(json_dir, filename)

        with open(file_path, 'r') as f:
            labelme_data = json.load(f)

        # Step 1: Process image information
        image_info = {
            "id": image_id,
            "file_name": labelme_data["imagePath"],
            "height": labelme_data["imageHeight"],
            "width": labelme_data["imageWidth"]
        }
        coco_output["images"].append(image_info)

        # Step 2: Process shapes/annotations
        for shape in labelme_data["shapes"]:
            label = shape["label"]

            # Manually set the category ID based on the label
            if label in category_id_mapping:
                category_id = category_id_mapping[label]
                label_counts[label] += 1  # Increment the label count
            else:
                # If the label doesn't exist in the mapping, skip it or handle it accordingly
                print(f"Warning: Label '{label}' not found in category mapping. Skipping annotation.")
                continue

            # Convert polygon points to segmentation format (flattened list of coordinates)
            polygon_points = np.array(shape["points"]).flatten().tolist()

            # Compute the bounding box for the polygon
            x_coords = [point[0] for point in shape["points"]]
            y_coords = [point[1] for point in shape["points"]]
            bbox = [min(x_coords), min(y_coords), max(x_coords) - min(x_coords), max(y_coords) - min(y_coords)]

            # Create the annotation structure
            annotation_info = {
                "id": annotation_id,
                "iscrowd": 0,  # Assuming all annotations are non-crowd
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [polygon_points],  # List of polygons, each polygon is a list of x, y coordinates
                "bbox": bbox,  # COCO format requires [x_min, y_min, width, height]
                "area": bbox[2] * bbox[3]  # Area of the bounding box
            }
            coco_output["annotations"].append(annotation_info)
            annotation_id += 1

        # Increment image ID for the next file
        image_id += 1


# Step 4: Plot the label frequencies
labels = list(label_counts.keys())
frequencies = list(label_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(labels, frequencies, color='skyblue')
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Frequency of Labels in Annotations')
plt.show()
