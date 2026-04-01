"""
Task I: Multi-Class Classification of Gravitational Lensing Images
Standalone Kaggle training script — paste into a Kaggle notebook cell or run as .py

Author: Rastin Aghighi
DEEPLENSE – GSoC 2026 (ML4SCI)

Usage on Kaggle:
  1. Upload dataset1.zip as a Kaggle dataset (or add the existing one)
  2. Adjust DATA_ROOT below to match the Kaggle dataset path
  3. Run with GPU accelerator enabled
"""

import os
import glob
import random
import copy

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — adjust these for your Kaggle environment
# ══════════════════════════════════════════════════════════════════════════════
DATA_ROOT = "/kaggle/input/deeplense-task1/dataset"  # <-- adjust this path
OUTPUT_DIR = "/kaggle/working"

SEED = 42
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7
NUM_CLASSES = 3
CLASS_NAMES = ["no substructure", "subhalo", "vortex"]

# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_classification_data(root, split):
    class_dirs = {"no": 0, "sphere": 1, "vort": 2}
    paths, labels = [], []
    for cls_name, cls_idx in class_dirs.items():
        cls_dir = os.path.join(root, split, cls_name)
        files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
        paths.extend(files)
        labels.extend([cls_idx] * len(files))
        print(f"  {split}/{cls_name}: {len(files)} samples")
    return paths, labels


print("Training set:")
train_paths, train_labels = load_classification_data(DATA_ROOT, "train")
print(f"  Total: {len(train_paths)}\n")

print("Validation set:")
val_paths, val_labels = load_classification_data(DATA_ROOT, "val")
print(f"  Total: {len(val_paths)}")

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class LensingClassificationDataset(Dataset):
    def __init__(self, paths, labels, augment=False):
        self.paths = paths
        self.labels = labels
        self.augment = augment
        self.aug_transform = T.Compose([
            T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        ]) if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = np.load(self.paths[idx]).astype(np.float32)
        img = torch.from_numpy(img)
        if self.aug_transform is not None:
            img = self.aug_transform(img)
        img = img.repeat(3, 1, 1)
        return img, self.labels[idx]


train_dataset = LensingClassificationDataset(train_paths, train_labels, augment=True)
val_dataset = LensingClassificationDataset(val_paths, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                  patience=3, verbose=True)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS} [Train]", leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

    train_loss = running_loss / total
    train_acc = correct / total
    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)

    # Validate
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    scheduler.step(val_loss)

    print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f"  ↑ New best val loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}")
            break

# Restore best and save
model.load_state_dict(best_model_state)
torch.save(best_model_state, os.path.join(OUTPUT_DIR, "resnet18_classifier.pth"))
print(f"Best validation loss: {best_val_loss:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs_range = range(1, len(history["train_loss"]) + 1)

ax1.plot(epochs_range, history["train_loss"], label="Train", linewidth=2)
ax1.plot(epochs_range, history["val_loss"], label="Validation", linewidth=2)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss"); ax1.set_title("Cross-Entropy Loss")
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history["train_acc"], label="Train", linewidth=2)
ax2.plot(epochs_range, history["val_acc"], label="Validation", linewidth=2)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy"); ax2.set_title("Classification Accuracy")
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
model.eval()
all_preds, all_labels_arr, all_probs = [], [], []
with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels_arr.append(labels.numpy())

all_preds = np.concatenate(all_preds)
all_labels_arr = np.concatenate(all_labels_arr)
all_probs = np.concatenate(all_probs)

# ── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(all_labels_arr, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Normalized Confusion Matrix")
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm[i, j]:.3f}\n({cm[i, j]})",
                ha="center", va="center", color=color, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()

print(classification_report(all_labels_arr, all_preds, target_names=CLASS_NAMES, digits=4))

# ── ROC Curves ───────────────────────────────────────────────────────────────
labels_bin = label_binarize(all_labels_arr, classes=[0, 1, 2])
fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0", "#DD8452", "#55A868"]

auc_scores = {}
for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores[cls_name] = roc_auc
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{cls_name} (AUC = {roc_auc:.4f})")

# Macro-average
all_fpr = np.linspace(0, 1, 200)
mean_tpr = np.zeros_like(all_fpr)
for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    mean_tpr += np.interp(all_fpr, fpr, tpr)
mean_tpr /= NUM_CLASSES
macro_auc = auc(all_fpr, mean_tpr)
auc_scores["macro-average"] = macro_auc
ax.plot(all_fpr, mean_tpr, color="navy", linewidth=2, linestyle="--",
        label=f"Macro-average (AUC = {macro_auc:.4f})")

ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves (One-vs-Rest)")
ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

print("\nAUC Scores:")
for name, score in auc_scores.items():
    print(f"  {name:20s}: {score:.4f}")

# ── Sample Predictions ───────────────────────────────────────────────────────
correct_mask = all_preds == all_labels_arr
incorrect_mask = ~correct_mask

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Predictions", fontsize=14, y=1.02)

correct_indices = np.where(correct_mask)[0]
shown = []
for cls_idx in range(NUM_CLASSES):
    cls_correct = correct_indices[all_labels_arr[correct_indices] == cls_idx]
    if len(cls_correct) > 0:
        shown.append(np.random.choice(cls_correct))
remaining = np.setdiff1d(correct_indices, shown)
shown.extend(np.random.choice(remaining, min(2, len(remaining)), replace=False))

for col, idx in enumerate(shown[:5]):
    img = np.load(val_paths[idx]).squeeze()
    axes[0, col].imshow(img, cmap="inferno", origin="lower")
    axes[0, col].set_title(f"True: {CLASS_NAMES[all_labels_arr[idx]]}\nPred: {CLASS_NAMES[all_preds[idx]]}",
                           fontsize=9, color="green")
    axes[0, col].axis("off")

incorrect_indices = np.where(incorrect_mask)[0]
if len(incorrect_indices) >= 5:
    wrong_samples = np.random.choice(incorrect_indices, 5, replace=False)
else:
    wrong_samples = incorrect_indices
for col, idx in enumerate(wrong_samples):
    img = np.load(val_paths[idx]).squeeze()
    axes[1, col].imshow(img, cmap="inferno", origin="lower")
    axes[1, col].set_title(f"True: {CLASS_NAMES[all_labels_arr[idx]]}\nPred: {CLASS_NAMES[all_preds[idx]]}",
                           fontsize=9, color="red")
    axes[1, col].axis("off")
for col in range(len(wrong_samples), 5):
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "task1_sample_predictions.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\nFinal validation accuracy: {correct_mask.mean():.4f}")
print(f"Incorrect predictions: {incorrect_mask.sum()} / {len(all_labels_arr)}")
print("\nDone! All figures and weights saved to", OUTPUT_DIR)
