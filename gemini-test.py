import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Or PyTorch equivalent

# Function to parse Pascal VOC XML (example)
def parse_pascal_voc_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []
    image_width = int(root.find('size/width').text)
    image_height = int(root.find('size/height').text)

    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(name)
    return {'boxes': boxes, 'labels': labels, 'width': image_width, 'height': image_height}

# Example of loading and parsing (adjust paths as needed)
dataset_root = 'One-Stage-TFS Thai One-Stage Fingerspelling Dataset'
image_dir = os.path.join(dataset_root, 'Training') # Adjust based on actual folder structure
# annotation_dir = os.path.join(dataset_root, 'annotations') # Adjust based on actual folder structure

all_image_paths = []
all_annotation_paths = []
for split_folder in ['train', 'test', 'unseen']: # Or however the dataset is split
    img_folder = os.path.join(image_dir, split_folder)
    anno_folder = os.path.join(annotation_dir, split_folder)
    if os.path.exists(img_folder):
        for img_name in os.listdir(img_folder):
            if img_name.endswith('.jpg'):
                all_image_paths.append(os.path.join(img_folder, img_name))
                xml_name = img_name.replace('.jpg', '.xml')
                all_annotation_paths.append(os.path.join(anno_folder, xml_name))

print(f"Found {len(all_image_paths)} images and annotations.")

# You would then typically load images and apply the parsing
# For a full system, you'd create a custom dataset loader that handles this for batches.

# Example for cropping hands (assuming you have hand_bbox and original_image)
# This would typically happen after your detection model predicts boxes

# def crop_hand(image, bbox):
#     xmin, ymin, xmax, ymax = bbox
#     cropped_hand = image[ymin:ymax, xmin:xmax]
#     return cv2.resize(cropped_hand, (desired_width, desired_height))

# Data augmentation setup (Keras example for classification)
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=False, # Fingerspelling might be asymmetric, so be careful with horizontal flip
#     fill_mode='nearest'
# )
# val_datagen = ImageDataGenerator(rescale=1./255)

from tensorflow.keras.applications import MobileNetV2 # Example
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

num_classes = 15 # For 15 Thai one-stage consonants

# Load MobileNetV2 pre-trained on ImageNet, excluding the top (classification) layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers (optional, unfreeze later for fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x) # Reduces spatial dimensions
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x) # Output layer for 15 classes

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy', # For one-hot encoded labels
              metrics=['accuracy'])

model.summary()