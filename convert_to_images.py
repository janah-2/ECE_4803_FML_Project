import os
import numpy as np
from PIL import Image

def find_npy_files(start_dir, file_name):
    """Recursively searches for files with the given name starting from start_dir."""
    for root, dirs, files in os.walk(start_dir):
        if file_name in files:
            return os.path.join(root, file_name)
    return None

# Directory containing the 18 folders
base_dir = '.'

# Output directories for images and labels
images_dir = 'images'
os.makedirs(images_dir, exist_ok=True)

# Initialize image and label counters
img_count = 0

# Process each folder
for i in range(18):
    folder_path = os.path.join(base_dir, f'folder_{i+1}')

    # Find samples.npy and labels.npy in the subfolders
    samples_path = find_npy_files(folder_path, 'samples.npy')

    if samples_path is None:
        print(f"Missing files in {folder_path}. Skipping...")
        continue

    # Load samples and labels with allow_pickle=True
    samples = np.load(samples_path, allow_pickle=True)

    # Process each sample and label in the folder
    for sample in samples:
        # Normalize and convert data type if necessary
        if sample.dtype == np.float32 or sample.dtype == np.float64:
            # Normalize to 0-255 and convert to uint8
            sample = 255 * (sample - sample.min()) / (sample.max() - sample.min())
            sample = sample.astype(np.uint8)
        
        # Convert sample to image
        img = Image.fromarray(sample)

        # Save the image
        img.save(os.path.join(images_dir, f'{img_count}.png'))
        print(f"Image {img_count} loaded and saved")

        img_count += 1

print("Conversion complete.")
