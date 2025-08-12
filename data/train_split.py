import os
import shutil

# Paths
train_folder = r"D:\projects\vehicle_detection\data\train"
images_folder = r"D:\projects\vehicle_detection\data\images"
annotations_folder = r"D:\projects\vehicle_detection\data\annotations"

# Create folders if they don't exist
os.makedirs(images_folder, exist_ok=True)
os.makedirs(annotations_folder, exist_ok=True)

# Loop through files in the train folder
for file in os.listdir(train_folder):
    src_path = os.path.join(train_folder, file)
    if file.endswith(".jpg"):
        shutil.move(src_path, os.path.join(images_folder, file))
    elif file.endswith(".JPG"):
        shutil.move(src_path, os.path.join(images_folder, file))
    elif file.endswith(".png"):
        shutil.move(src_path, os.path.join(images_folder, file))
    elif file.endswith(".PNG"):
        shutil.move(src_path, os.path.join(images_folder, file))
    elif file.endswith(".jpg"):
        shutil.move(src_path, os.path.join(images_folder, file))    
    elif file.endswith(".xml"):
        shutil.move(src_path, os.path.join(annotations_folder, file))
    elif file.endswith(".jpeg"):
        shutil.move(src_path, os.path.join(images_folder, file))
    
print("Files separated successfully!")