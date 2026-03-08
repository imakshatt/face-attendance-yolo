import os
import shutil

# Path to dataset folder
dataset_path = "/home/akshat/attandance_system/dataset"

# Get all jpg files
images = [f for f in os.listdir(dataset_path) if f.lower().endswith(".jpg")]

# Sort images (optional but good practice)
images.sort()

# Starting folder number
start_id = 101

for i, image in enumerate(images):
    folder_id = start_id + i
    folder_path = os.path.join(dataset_path, str(folder_id))

    # Create folder (101,102,103...)
    os.makedirs(folder_path, exist_ok=True)

    # Move image
    src = os.path.join(dataset_path, image)
    dst = os.path.join(folder_path, image)

    shutil.move(src, dst)

print("Dataset structure created successfully!")