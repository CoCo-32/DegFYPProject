import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def load_json_annotation(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def polygon_to_mask(polygon, image_height, image_width):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    polygon = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)
    return mask

def visualize_mask(image_path, mask, output_path):
    # Load the original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}. Using blank image.")
        image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a colored mask (red with 50% opacity)
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 255] = [255, 0, 0]  # Red color

    # Blend the image and the colored mask
    blended = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

    # Save the visualization
    plt.figure(figsize=(12, 8))
    plt.imshow(blended)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def process_annotation(annotation_file, output_dir):
    data = load_json_annotation(annotation_file)
    image_height = data['imageHeight']
    image_width = data['imageWidth']
    
    for shape_id, shape in enumerate(data['shapes']):
        category_name = shape['label']
        polygon = shape['points']
        
        # Generate mask
        mask = polygon_to_mask(polygon, image_height, image_width)
        
        # Visualize and save the mask
        image_path = os.path.join(os.path.dirname(annotation_file), data['imagePath'])
        output_path = os.path.join(output_dir, f"{os.path.splitext(data['imagePath'])[0]}_mask_{shape_id}.png")
        visualize_mask(image_path, mask, output_path)
        
        # Optionally, you can save the raw mask as well
        cv2.imwrite(os.path.join(output_dir, f"{os.path.splitext(data['imagePath'])[0]}_raw_mask_{shape_id}.png"), mask)

def process_all_annotations(annotation_files, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for annotation_file in annotation_files:
        process_annotation(annotation_file, output_dir)

# Example usage
annotation_files = ['SolDef_AI/Labeled/WIN_20220408_14_11_34_Pro.json']
output_dir = 'output_masks'
process_all_annotations(annotation_files, output_dir)

print(f"Processed {len(annotation_files)} annotations.")
print(f"Mask visualizations and raw masks saved in '{output_dir}' directory.")