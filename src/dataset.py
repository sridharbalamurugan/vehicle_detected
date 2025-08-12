import os
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
import torchvision.transforms.functional as TF

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, classes=None, transforms=None, transform=None):
        """
        img_dir: path to images folder
        ann_dir: path to annotations folder (pascal VOC xml files)
        classes: list like ['__background__','car','bus','truck','bike']
        transforms/transform: callable that accepts (pil_image, target) and returns (image_tensor, target)
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        # build ordered list of image ids (without extension)
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        self.ids = [os.path.splitext(f)[0] for f in sorted(img_files)]
        # keep compatibility: user may pass 'transforms' or 'transform'
        self.transforms = transforms if transforms is not None else transform

        if classes is None:
            self.classes = ['__background__', 'car', 'bus', 'truck', 'bike']
        else:
            self.classes = classes
        # mapping label->index
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        # find actual file extension for this id
        img_path = None
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.PNG', '.JPEG'):
            candidate = os.path.join(self.img_dir, img_id + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            raise FileNotFoundError(f"No image found for id {img_id} in {self.img_dir}")

        ann_path = os.path.join(self.ann_dir, img_id + '.xml')
        image = Image.open(img_path).convert("RGB")

        boxes = []
        labels = []
        if os.path.exists(ann_path):
            tree = ET.parse(ann_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text.lower()
                if name not in self.class_to_idx:
                    # skip classes not in our list
                    continue
                xmlbox = obj.find('bndbox')
                xmin = float(xmlbox.find('xmin').text)
                ymin = float(xmlbox.find('ymin').text)
                xmax = float(xmlbox.find('xmax').text)
                ymax = float(xmlbox.find('ymax').text)
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(self.class_to_idx[name])

        # convert to tensors (handle empty case)
        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor([idx])
        # area calculation: (xmax-xmin)*(ymax-ymin)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.zeros((0,), dtype=torch.float32)
        iscrowd = torch.zeros((labels.shape[0],), dtype=torch.int64) if labels.numel() > 0 else torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        # If transforms is a callable that accepts (image, target) -> (image, target)
        if callable(self.transforms):
            image, target = self.transforms(image, target)
        else:
            # if transforms is None or image-only transforms, convert to tensor (image-only)
            image = TF.to_tensor(image)

        return image, target


# --- small detection-aware transforms ---
class ToTensor:
    def __call__(self, image, target):
        # image may be PIL Image or already a tensor; TF.to_tensor handles PIL Images.
        if isinstance(image, torch.Tensor):
            return image, target
        image = TF.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    """
    Detection-aware horizontal flip that supports both PIL Image and torch.Tensor (C,H,W).
    Adjusts bounding boxes accordingly: [xmin, ymin, xmax, ymax].
    """
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            # Flip image (works for both PIL.Image and torch.Tensor)
            image = TF.hflip(image)

            # get image width depending on type
            if isinstance(image, torch.Tensor):
                # Tensor shape: C,H,W
                _, h, w = image.shape
            else:
                # PIL.Image: size -> (width, height)
                w, h = image.size

            if target["boxes"].numel() > 0:
                boxes = target["boxes"].clone()
                # boxes: [xmin, ymin, xmax, ymax]
                # new_xmin = w - old_xmax
                # new_xmax = w - old_xmin
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
        return image, target

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


def get_transform(train):
    t = [ToTensor()]
    return Compose(t)




def collate_fn(batch):
    return tuple(zip(*batch))
