import os
import random
import shutil

# Paths
train_img_dir = "data/train/images"
train_ann_dir = "data/train/annotations"
val_img_dir = "data/val/images"
val_ann_dir = "data/val/annotations"

# Create val folders if they don't exist
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_ann_dir, exist_ok=True)

# List all image files
img_files = [f for f in os.listdir(train_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# Shuffle and select 20%
random.shuffle(img_files)
val_count = int(len(img_files) * 0.2)
val_files = img_files[:val_count]

# Move files
for img_file in val_files:
    # Move image
    src_img_path = os.path.join(train_img_dir, img_file)
    dst_img_path = os.path.join(val_img_dir, img_file)
    shutil.move(src_img_path, dst_img_path)

    # Move annotation (assuming XML with same filename but .xml)
    ann_file = os.path.splitext(img_file)[0] + ".xml"
    src_ann_path = os.path.join(train_ann_dir, ann_file)
    dst_ann_path = os.path.join(val_ann_dir, ann_file)
    if os.path.exists(src_ann_path):
        shutil.move(src_ann_path, dst_ann_path)

print(f" Moved {len(val_files)} images and annotations to validation set.")
