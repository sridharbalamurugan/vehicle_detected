import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from dataset import VOCDataset, get_transform
from model import get_model
import utils
from evaluate import evaluate
import matplotlib.pyplot as plt

train_losses = []
val_precisions = []
val_recalls = []
# Prepare datasets

full_train_dataset = VOCDataset(
    img_dir='data/train/images',
    ann_dir='data/train/annotations',
    classes=['__background__', 'bus', 'car', 'bike', 'truck'],
    transforms=get_transform(train=True)
)

full_val_dataset = VOCDataset(
    img_dir='data/val/images',
    ann_dir='data/val/annotations',
    classes=['__background__', 'bus', 'car', 'bike', 'truck'],
    transforms=get_transform(train=False)
)

full_test_dataset = VOCDataset(
    img_dir='data/test/images',
    ann_dir='data/test/annotations',
    classes=['__background__', 'bus', 'car', 'bike', 'truck'],
    transforms=get_transform(train=False)
)

# For quick runs, use subsets (optional)
subset_indices_train = list(range(min(10, len(full_train_dataset))))
train_dataset = Subset(full_train_dataset, subset_indices_train)

subset_indices_val = list(range(min(5, len(full_val_dataset))))
val_dataset = Subset(full_val_dataset, subset_indices_val)

# Data loaders
train_loader = DataLoader(
    train_dataset, batch_size=1, shuffle=True,
    collate_fn=utils.collate_fn, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False,
    collate_fn=utils.collate_fn, num_workers=0
)

test_loader = DataLoader(
    full_test_dataset, batch_size=1, shuffle=False,
    collate_fn=utils.collate_fn, num_workers=0
)


device = torch.device('cpu')
print(f"Using device: {device}")

# Prepare datasets and loaders (as before) ...
# [Your dataset and loader code here]

model = get_model(num_classes=5)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

num_epochs = 2

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} starting...")
    model.train()
    epoch_loss = 0.0

    for i, (images, targets) in enumerate(train_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        print(f"  Batch {i+1}, Loss: {losses.item():.4f}")

    lr_scheduler.step()
    print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {epoch_loss:.4f}")

    print("Evaluating on validation set...")
    avg_precision, avg_recall = evaluate(model, val_loader, device=device)

    # Print accuracy in percentage
    print(f"Validation Precision: {avg_precision * 100:.2f}%")
    print(f"Validation Recall: {avg_recall * 100:.2f}%")

    train_losses.append(epoch_loss)
    val_precisions.append(avg_precision)
    val_recalls.append(avg_recall)

    # Save checkpoint
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), f"outputs/checkpoints/minimal_model_epoch_{epoch+1}.pth")

print("Training complete.")

print("Evaluating on test set...")
evaluate(model, test_loader, device=device)

# Skip TorchScript export because it errors with detection model outputs
# If needed, can implement later with proper scripting or tracing alternatives.

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_precisions, label='Validation Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_recalls, label='Validation Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.legend()

plt.tight_layout()
plt.show()
