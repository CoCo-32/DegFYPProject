import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes

from mrcnn.utils import Dataset
from matplotlib import pyplot as plt

from mrcnn.config import Config
from mrcnn.model import MaskRCNN


from mrcnn import model as modellib, utils

class SolderDataset(utils.Dataset):

    def load_dataset(self, dataset_dir):
        """Load a subset of the custom dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        #These lines add two classes to the dataset: "Blue_Marble" with ID 1 and "Non_Blue_Marble" with ID 2.
        # Add classes according to the numbe of classes required to detect
        self.add_class("custom", 1, "good")
        self.add_class("custom", 2, "no_good")
        self.add_class("custom", 3, "exc_solder")
        self.add_class("custom", 4, "poor")
        self.add_class("custom", 5, "spike")


        dataset_dir = 'SolDef_AI/Labeled'

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        #These lines load the annotations from a JSON file, convert them to a list, and filter out any annotations without regions.
        extracted_data = []
        for filename in os.listdir(dataset_dir):
            if filename.endswith('.json'):  # Process only JSON files
                file_path = os.path.join(dataset_dir, filename)

                # Load the JSON data from the file
                with open(file_path, 'r') as file:
                    data = json.load(file)

                    # Extracting components
                    image_path = data['imagePath']
                    shapes = data['shapes']

                    class_id=[]
                    # Loop through each shape to extract details
                    for shape in shapes:
                        label = shape['label']
                        try:
                            if label =='good':
                                class_id.append(1)
                            elif label =='no_good':
                                class_id.append(2)
                            elif label =='exc_solder':
                                class_id.append(3)
                            elif label =='poor':
                                class_id.append(4)
                            elif label =='spike':
                                class_id.append(5)
                        except:
                            pass
                        points = shape['points']
                        shape_type = shape['shape_type']
                        
                        self.add_image(
                        "custom",
                        image_id=filename,  # use file name as a unique image id
                        path=image_path,
                        shape_type=shape_type,
                        #width=width, height=height,
                        points=points,
                        class_id=class_id)

                        print(self.add_image)