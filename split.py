import os
from sklearn.model_selection import train_test_split
import shutil

# Define base paths for images and labels
images_folder = 'images'
labels_folder = 'labels'

# Define subdirectories for training, validation, and testing within each folder
train_images_folder = os.path.join(images_folder, 'train')
val_images_folder = os.path.join(images_folder, 'val')
test_images_folder = os.path.join(images_folder, 'test')

train_labels_folder = os.path.join(labels_folder, 'train')
val_labels_folder = os.path.join(labels_folder, 'val')
test_labels_folder = os.path.join(labels_folder, 'test')

# Create the subdirectories if they don't exist
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(test_images_folder, exist_ok=True)

os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)
os.makedirs(test_labels_folder, exist_ok=True)

# List all images and labels
images = sorted(os.listdir(images_folder))
labels = sorted(os.listdir(labels_folder))

# Split the data
images_train, images_temp, labels_train, labels_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
images_val, images_test, labels_val, labels_test = train_test_split(images_temp, labels_temp, test_size=0.5, random_state=42)

# Function to copy files to new folders
def copy_files(source_folder, file_list, destination_folder):
    for file in file_list:
        shutil.copy(os.path.join(source_folder, file), os.path.join(destination_folder, file))

# Copy images to respective folders
copy_files(images_folder, images_train, train_images_folder)
copy_files(images_folder, images_val, val_images_folder)
copy_files(images_folder, images_test, test_images_folder)

# Copy labels to respective folders
copy_files(labels_folder, labels_train, train_labels_folder)
copy_files(labels_folder, labels_val, val_labels_folder)
copy_files(labels_folder, labels_test, test_labels_folder)

print("Data splitting and organization completed.")
