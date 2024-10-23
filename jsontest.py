import json
import os

# Directory containing the JSON files
directory_path = 'SolDef_AI/Labeled'

# List to hold extracted data
extracted_data = []

# Loop through each file in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.json'):  # Process only JSON files
        file_path = os.path.join(directory_path, filename)

        # Load the JSON data from the file
        with open(file_path, 'r') as file:
            data = json.load(file)

            # Extracting components
            image_path = data['imagePath']
            shapes = data['shapes']

            # Loop through each shape to extract details
            for shape in shapes:
                label = shape['label']
                points = shape['points']
                shape_type = shape['shape_type']
                
                # Store the extracted information in a dictionary
                extracted_data.append({
                    'image_path': image_path, 
                    'shape_label': label,
                    'shape_type': shape_type,
                    'points': points
                })

# Optionally, print the extracted data
for entry in extracted_data:
    print(entry)
    print('\n')