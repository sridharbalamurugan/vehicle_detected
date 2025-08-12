# D:\projects\vehicle_detection\src\train.py
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import VOCDataset, get_transform
from model import get_model
import utils
import evaluate

# ------------ Device ------------
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# ------------ Paths & classes ------------
train_img_dir = 'data/train/images'
train_ann_dir = 'data/train/annotations'
val_img_dir = 'data/val/images'
val_ann_dir = 'data/val/annotations'

classes = ['__background__', 'bus', 'car', 'bike', 'truck']  # background + 4 vehicle classes

# ------------ Datasets & Dataloaders ------------
train_dataset = VOCDataset(
    img_dir=train_img_dir,
    ann_dir=train_ann_dir,
    classes=classes,
    transforms=get_transform(train=True)
)

val_dataset = VOCDataset(
    img_dir=val_img_dir,
    ann_dir=val_ann_dir,
    classes=classes,
    transforms=get_transform(train=False)
)

# DataLoader settings: adjust num_workers for your machine (0 on Windows without issues)
num_workers = 0  # if Windows, or set 2/4 for Linux CUDA machines
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=utils.collate_fn, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=utils.collate_fn, num_workers=num_workers)

# ------------ Model ------------
num_classes = len(classes)
model = get_model(num_classes)
model.to(device)

# ------------ Optimizer & Scheduler ------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# ------------ Training loop ------------
num_epochs = 10
os.makedirs("outputs/checkpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0

    for images, targets in train_loader:
        # images: tuple of tensors, targets: tuple of dicts
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    lr_scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}] - Training Loss: {epoch_loss:.4f}")

    # Evaluate on validation set
    try:
        evaluate.evaluate(model, val_loader, device=device)
    except Exception as e:
        print(f"Evaluation failed: {e}")

    # Save checkpoint (including optimizer & scheduler states)
    checkpoint_path = f"outputs/checkpoints/model_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint: {checkpoint_path}")
