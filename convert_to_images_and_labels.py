import os
import numpy as np
from PIL import Image
from skimage.measure import label, regionprops

# Directory containing the 18 folders
base_dir = '.'

# Output directories for images and labels
images_dir = 'images'
labels_dir = 'labels'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

def enumerate_classes(base_dir):
    """Enumerate all folders and assign unique class IDs."""
    folders = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    class_ids = {}
    current_id = 0
    for folder in folders:
        if folder == "Extra Credit":
            # Handle subfolders in the Extra Credit folder
            subfolders = sorted(os.listdir(os.path.join(base_dir, folder)))
            for subfolder in subfolders:
                class_ids[subfolder] = current_id
                current_id += 1
        else:
            class_ids[folder] = current_id
            current_id += 1
    return class_ids

class_ids = enumerate_classes(base_dir)

# Function to convert mask to YOLO format labels
def mask_to_yolo_labels(mask, image_index, save_dir, class_id):
    """Convert mask to YOLO format and save."""
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    yolo_labels = []

    for region in regions:
        minr, minc, maxr, maxc = region.bbox
        x_center = ((minc + maxc) / 2) / mask.shape[1]
        y_center = ((minr + maxr) / 2) / mask.shape[0]
        width = (maxc - minc) / mask.shape[1]
        height = (maxr - minr) / mask.shape[0]
        yolo_labels.append(f"{class_id} {x_center} {y_center} {width} {height}")

    print("Writing image and label ", image_index)

    with open(os.path.join(save_dir, f'{image_index}.txt'), 'w') as file:
        file.writelines(s + '\n' for s in yolo_labels)

# Initialize image counter
img_count = 0

# Process each folder
for folder_name in sorted(class_ids.keys()):
    folder_path = os.path.join(base_dir, folder_name)
    samples_path = os.path.join(folder_path, 'samples.npy')
    labels_path = os.path.join(folder_path, 'labels.npy')

    # Check if the samples and labels files exist
    if not os.path.exists(samples_path) or not os.path.exists(labels_path):
        print(f"Missing files in {folder_path}. Skipping...")
        continue

    # Load samples and labels
    samples = np.load(samples_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)

    # Process each sample and label
    for sample, label in zip(samples, labels):
        # Convert the sample to an image and save
        img = Image.fromarray(sample)
        img.save(os.path.join(images_dir, f'{img_count}.png'))
        print(f"Image {img_count} loaded and saved")

        # Convert the label to YOLO format and save
        mask_to_yolo_labels(label, img_count, labels_dir, class_ids[folder_name])

        # Increment the image counter
        img_count += 1

print("Conversion complete.")