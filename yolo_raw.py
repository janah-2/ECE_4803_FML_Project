from ultralytics import YOLO
import os
import csv
import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 

# Load the model
model = YOLO('yolov8n.pt')

# Directory containing images
image_dir = 'images'

image_counter = 0

# Output directories for results
results_dir = 'yoloRaw'
output_images_dir = os.path.join(results_dir, 'images')
output_labels_dir = os.path.join(results_dir, 'labels')

# Output CSV file path
csv_path = os.path.join(results_dir, 'detection_results.csv')

# Ensure the output directories exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Get all image paths and sort them numerically
image_paths = sorted(
    [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.png', '.jpg', '.jpeg'))],
    key=lambda x: int(os.path.splitext(os.path.basename(x))[0])
)

# Batch size for processing
batch_size = 200

# Specify the desired classes
desired_classes = {34, 14, 5, 15, 74, 19, 16, 53, 11, 27}

# Open the CSV file to write the results
with open(csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Detected Classes', 'Preprocess Time (ms)', 'Inference Time (ms)', 'Postprocess Time (ms)'])

    # Function to process a batch of images
    def process_batch(batch_image_paths, image_counter):
        results = model(batch_image_paths)
        for idx, result in enumerate(results):
            image_path = batch_image_paths[idx]
            print(f"Analyzing image: {image_path}")

            # Prepare output paths for image and label
            image_output_path = os.path.join(output_images_dir, f"{image_counter}.png")
            label_output_path = os.path.join(output_labels_dir, f"{image_counter}.txt")

            if result.boxes and hasattr(result.boxes, 'cls') and result.names:
                class_indices = result.boxes.cls.to(dtype=torch.int32).tolist()
                class_names = [result.names[idx] for idx in class_indices if idx in desired_classes]

                if hasattr(result.boxes, 'orig_shape'):
                    image_height, image_width = result.boxes.orig_shape[:2]
                else:
                    # Fallback if orig_shape is not available
                    with ImageFile.open(image_path) as img:
                        image_width, image_height = img.size

                # Save label file with YOLO format labels
                with open(label_output_path, 'w') as label_file:
                    for cls, bbox in zip(class_indices, result.boxes.xyxy):
                        if cls in desired_classes:
                            x_min, y_min, x_max, y_max = bbox.cpu().numpy()
                            x_center = (x_min + x_max) / (2 * image_width)
                            y_center = (y_min + y_max) / (2 * image_height)
                            width = (x_max - x_min) / image_width
                            height = (y_max - y_min) / image_height
                            label_file.write(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                # Save the image with detections
                result.save(image_output_path)

                if hasattr(result, 'speed'):
                    preprocess_time = result.speed.get('preprocess', 'N/A')
                    inference_time = result.speed.get('inference', 'N/A')
                    postprocess_time = result.speed.get('postprocess', 'N/A')
                else:
                    preprocess_time = inference_time = postprocess_time = 'N/A'
                writer.writerow([image_output_path, ', '.join(class_names), preprocess_time, inference_time, postprocess_time])
            else:
                writer.writerow([image_path, 'No valid detections', 'N/A', 'N/A', 'N/A'])
            # print("counter in: ", image_counter)
            image_counter += 1
        return image_counter

    # Process images in batches
    for start_idx in range(0, len(image_paths), batch_size):
        end_idx = min(start_idx + batch_size, len(image_paths))
        batch_image_paths = image_paths[start_idx:end_idx]
        image_counter = process_batch(batch_image_paths, image_counter)
        # print("counter out: ", image_counter)

print("Inference complete and results are saved to:", csv_path)