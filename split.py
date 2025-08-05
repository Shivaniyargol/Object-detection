import os
import shutil
import random

# Set paths
ANNOTATIONS_DIR = "C:\\Users\\Nayanaa\\Desktop\\minip\\dataset\\Annotations"  # Path to your annotations folder
IMAGES_DIR ="C:\\Users\\Nayanaa\\Desktop\\minip\\dataset\\Images"            # Path to your images folder
OUTPUT_DIR = "dataset_split"     # Output directory for train/test split
TRAIN_RATIO = 0.8                # Percentage of data to use for training

# Create output directories
train_images_dir = os.path.join(OUTPUT_DIR, "train", "images")
train_annotations_dir = os.path.join(OUTPUT_DIR, "train", "annotations")
test_images_dir = os.path.join(OUTPUT_DIR, "test", "images")
test_annotations_dir = os.path.join(OUTPUT_DIR, "test", "annotations")

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_annotations_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_annotations_dir, exist_ok=True)

# Get all image files (assuming they end in .jpg or .png)
image_files = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))]

# Shuffle the dataset for randomness
random.shuffle(image_files)

# Split dataset
train_count = int(len(image_files) * TRAIN_RATIO)
train_files = image_files[:train_count]
test_files = image_files[train_count:]

# Function to copy files
def copy_files(file_list, src_dir, dest_image_dir, dest_annotation_dir):
    for file in file_list:
        # Copy image file
        src_image_path = os.path.join(src_dir, file)
        dest_image_path = os.path.join(dest_image_dir, file)
        shutil.copy(src_image_path, dest_image_path)

        # Copy corresponding annotation file
        annotation_file = file.replace(".jpg", ".xml").replace(".png", ".xml")
        src_annotation_path = os.path.join(ANNOTATIONS_DIR, annotation_file)
        dest_annotation_path = os.path.join(dest_annotation_dir, annotation_file)

        if os.path.exists(src_annotation_path):
            shutil.copy(src_annotation_path, dest_annotation_path)

# Copy training files
copy_files(train_files, IMAGES_DIR, train_images_dir, train_annotations_dir)

# Copy testing files
copy_files(test_files, IMAGES_DIR, test_images_dir, test_annotations_dir)

print(f"Dataset split completed! Training set: {len(train_files)}, Testing set: {len(test_files)}")
