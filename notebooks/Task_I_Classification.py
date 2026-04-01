# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# **Disclaimer:** This notebook was developed with the assistance of AI tools for code generation and formatting. The research strategy, architectural decisions, and experimental design were conducted by the author with guidance from academic advisors. All code was reviewed, tested, and validated by the author.


# %% [markdown]
# # Task I: Multi-Class Classification of Gravitational Lensing Images
# **Author:** Rastin Aghighi
# **DEEPLENSE - GSoC 2026 (ML4SCI)**
# **Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images
#
# ## Strategy
#
# We use a ResNet-18 backbone pretrained on ImageNet for multi-class classification
# of strong gravitational lensing images into three categories: **no substructure**,
# **subhalo substructure**, and **vortex substructure**. Transfer learning from
# ImageNet is effective here despite the domain gap because the early convolutional
# layers learn general edge and texture detectors that transfer well to grayscale
# astrophysical images. The single-channel lensing images are repeated across three
# channels to match the pretrained input expectations.
#
# We apply data augmentation that respects the physical symmetries of lensing images:
# random rotation (continuous, 0-360°), horizontal flip, and vertical flip.
# Gravitational lensing has no preferred orientation on the sky, so all rotations
# are physically valid augmentations. This effectively multiplies the training set
# without introducing unphysical distortions.
#
# This classification task is directly relevant to the GSoC project: super-resolved
# images should preserve or enhance the discriminative features that enable accurate
# classification. A strong classifier trained on native-resolution images provides
# a baseline against which we can later evaluate whether SR-enhanced images improve
# downstream classification accuracy.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import sys
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
from tqdm.auto import tqdm

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 7
NUM_CLASSES = 3
CLASS_NAMES = ["no substructure", "subhalo", "vortex"]

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_ROOT = os.path.join("..", "dataset1", "dataset")
FIGURES_DIR = os.path.join("..", "figures")
WEIGHTS_DIR = os.path.join("..", "weights")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# %% [markdown]
# ## 2. Data Loading and Exploration

# %%
# ── Load file paths ──────────────────────────────────────────────────────────
def load_classification_data(root, split):
    """Load .npy file paths and labels for a given split (train/val)."""
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

# %%
# ── Class distribution ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, (labels, title) in zip(axes, [(train_labels, "Train"), (val_labels, "Validation")]):
    counts = [labels.count(i) for i in range(NUM_CLASSES)]
    bars = ax.bar(CLASS_NAMES, counts, color=["#4C72B0", "#DD8452", "#55A868"])
    ax.set_title(f"{title} Class Distribution")
    ax.set_ylabel("Count")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                str(count), ha="center", va="bottom", fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_class_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Sample grid: 3 classes × 5 examples ─────────────────────────────────────
fig, axes = plt.subplots(3, 5, figsize=(12, 8))

for row, (cls_name, cls_idx) in enumerate(zip(CLASS_NAMES, range(NUM_CLASSES))):
    cls_indices = [i for i, l in enumerate(train_labels) if l == cls_idx]
    samples = random.sample(cls_indices, 5)
    for col, idx in enumerate(samples):
        img = np.load(train_paths[idx]).squeeze()
        axes[row, col].imshow(img, cmap="inferno", origin="lower")
        axes[row, col].axis("off")
        if col == 0:
            axes[row, col].set_ylabel(cls_name, fontsize=12, rotation=0, labelpad=80, va="center")

fig.suptitle("Sample Lensing Images by Class", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_sample_grid.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Intensity statistics ─────────────────────────────────────────────────────
sample_img = np.load(train_paths[0])
print(f"Image shape: {sample_img.shape}")
print(f"Dtype: {sample_img.dtype}")
print(f"Value range: [{sample_img.min():.4f}, {sample_img.max():.4f}]")

# %% [markdown]
# ## 3. Dataset and DataLoader

# %%
class LensingClassificationDataset(Dataset):
    """Dataset for gravitational lensing classification.

    Augmentation respects the rotational symmetry of lensing images:
    - Random continuous rotation (0-360°): no preferred orientation on the sky
    - Random horizontal flip: no preferred handedness
    - Random vertical flip: same reasoning
    """

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
        img = np.load(self.paths[idx]).astype(np.float32)  # (1, 150, 150)
        img = torch.from_numpy(img)  # (1, 150, 150)

        if self.aug_transform is not None:
            img = self.aug_transform(img)

        # Repeat grayscale to 3 channels for pretrained ResNet
        img = img.repeat(3, 1, 1)  # (3, 150, 150)

        label = self.labels[idx]
        return img, label


train_dataset = LensingClassificationDataset(train_paths, train_labels, augment=True)
val_dataset = LensingClassificationDataset(val_paths, val_labels, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=2, pin_memory=True)

# Sanity check
imgs, lbls = next(iter(train_loader))
print(f"Batch shape: {imgs.shape}, Labels: {lbls[:8]}")

# %% [markdown]
# ## 4. Model

# %%
def build_classifier(num_classes=3):
    """Build a ResNet-18 classifier adapted for single-channel 150×150 lensing images.

    - Pretrained ImageNet weights provide strong low-level feature extractors
    - Final FC layer replaced for 3-class output
    - Input: grayscale repeated 3× to match pretrained expectations
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


model = build_classifier(NUM_CLASSES).to(device)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## 5. Training

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                  patience=3, verbose=True)

# ── Training loop ────────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_loss = float("inf")
best_model_state = None
epochs_no_improve = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # ── Train ────────────────────────────────────────────────────────────
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

    # ── Validate ─────────────────────────────────────────────────────────
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

    # ── Early stopping ───────────────────────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = copy.deepcopy(model.state_dict())
        epochs_no_improve = 0
        print(f"  ↑ New best val loss: {val_loss:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# ── Restore best model ───────────────────────────────────────────────────────
model.load_state_dict(best_model_state)
save_path = os.path.join(WEIGHTS_DIR, "resnet18_classifier.pth")
torch.save(best_model_state, save_path)
print(f"\nBest model saved to {save_path}")
print(f"Best validation loss: {best_val_loss:.4f}")

# %% [markdown]
# ## 6. Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

epochs_range = range(1, len(history["train_loss"]) + 1)

ax1.plot(epochs_range, history["train_loss"], label="Train", linewidth=2)
ax1.plot(epochs_range, history["val_loss"], label="Validation", linewidth=2)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Cross-Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, history["train_acc"], label="Train", linewidth=2)
ax2.plot(epochs_range, history["val_acc"], label="Validation", linewidth=2)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Classification Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_training_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Evaluation: Confusion Matrix

# %%
# ── Collect predictions on validation set ────────────────────────────────────
model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for imgs, labels in tqdm(val_loader, desc="Evaluating"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(outputs.argmax(dim=1).cpu().numpy())
        all_labels.append(labels.numpy())

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
all_probs = np.concatenate(all_probs)

# ── Confusion matrix (row-normalized) ────────────────────────────────────────
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax)

ax.set_xticks(range(NUM_CLASSES))
ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Normalized Confusion Matrix")

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm_norm[i, j]:.3f}\n({cm[i, j]})",
                ha="center", va="center", color=color, fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_confusion_matrix.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── Classification report ────────────────────────────────────────────────────
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

# %% [markdown]
# ## 8. ROC Curves and AUC

# %%
# ── One-vs-Rest ROC curves ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))
colors = ["#4C72B0", "#DD8452", "#55A868"]

# Binarize labels for OvR
from sklearn.preprocessing import label_binarize
labels_bin = label_binarize(all_labels, classes=[0, 1, 2])

auc_scores = {}
for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)
    auc_scores[cls_name] = roc_auc
    ax.plot(fpr, tpr, color=color, linewidth=2,
            label=f"{cls_name} (AUC = {roc_auc:.4f})")

# Macro-average ROC
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
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_roc_curves.png"), dpi=150, bbox_inches="tight")
plt.show()

# ── Print AUC scores ─────────────────────────────────────────────────────────
print("\nAUC Scores:")
for name, score in auc_scores.items():
    print(f"  {name:20s}: {score:.4f}")

# %% [markdown]
# ## 9. Sample Predictions

# %%
# ── Correct and incorrect predictions ────────────────────────────────────────
correct_mask = all_preds == all_labels
incorrect_mask = ~correct_mask

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Sample Predictions", fontsize=14, y=1.02)

# Top row: correct predictions (one per class + 2 random)
correct_indices = np.where(correct_mask)[0]
shown = []
for cls_idx in range(NUM_CLASSES):
    cls_correct = correct_indices[all_labels[correct_indices] == cls_idx]
    if len(cls_correct) > 0:
        shown.append(np.random.choice(cls_correct))
remaining = np.setdiff1d(correct_indices, shown)
shown.extend(np.random.choice(remaining, min(2, len(remaining)), replace=False))

for col, idx in enumerate(shown[:5]):
    img = np.load(val_paths[idx]).squeeze()
    axes[0, col].imshow(img, cmap="inferno", origin="lower")
    axes[0, col].set_title(f"True: {CLASS_NAMES[all_labels[idx]]}\nPred: {CLASS_NAMES[all_preds[idx]]}",
                           fontsize=9, color="green")
    axes[0, col].axis("off")
axes[0, 0].set_ylabel("Correct", fontsize=12, rotation=0, labelpad=50, va="center")

# Bottom row: incorrect predictions
incorrect_indices = np.where(incorrect_mask)[0]
if len(incorrect_indices) >= 5:
    wrong_samples = np.random.choice(incorrect_indices, 5, replace=False)
else:
    wrong_samples = incorrect_indices

for col, idx in enumerate(wrong_samples):
    img = np.load(val_paths[idx]).squeeze()
    axes[1, col].imshow(img, cmap="inferno", origin="lower")
    axes[1, col].set_title(f"True: {CLASS_NAMES[all_labels[idx]]}\nPred: {CLASS_NAMES[all_preds[idx]]}",
                           fontsize=9, color="red")
    axes[1, col].axis("off")
# Hide unused axes if fewer than 5 incorrect
for col in range(len(wrong_samples), 5):
    axes[1, col].axis("off")
axes[1, 0].set_ylabel("Incorrect", fontsize=12, rotation=0, labelpad=50, va="center")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "task1_sample_predictions.png"), dpi=150, bbox_inches="tight")
plt.show()

val_acc_final = correct_mask.mean()
print(f"\nFinal validation accuracy: {val_acc_final:.4f}")
print(f"Incorrect predictions: {incorrect_mask.sum()} / {len(all_labels)}")

# %% [markdown]
# ## 10. Discussion
#
# **Architecture choice:** ResNet-18 is a well-established baseline for
# gravitational lensing classification (see Lanusse et al. 2018, Schaefer et al.
# 2018). Its relatively shallow depth prevents overfitting on the 30,000-image
# training set while still providing enough capacity to learn the subtle
# morphological differences between substructure classes. ImageNet pretraining
# gives a strong initialisation even for single-channel astrophysical images.
#
# **Hardest class:** We expect the "no substructure" vs "subhalo" distinction to
# be the most challenging, as subhalo perturbations can be subtle - small
# distortions in the arc morphology that are easy to miss. The vortex class
# typically produces more distinctive spiral-like features that the network can
# learn more readily.
#
# **Augmentation rationale:** Gravitational lensing images have full rotational
# symmetry (no preferred orientation on the sky) and no preferred handedness.
# Random rotations and flips exploit these symmetries to effectively multiply
# the training set without introducing unphysical distortions.
#
# **Connection to GSoC:** This classifier provides a quantitative framework for
# evaluating downstream impacts of super-resolution. If SR-enhanced images
# improve classification AUC, it demonstrates that the super-resolution model
# preserves and enhances physically meaningful substructure features - a key
# validation for the GSoC project goal of unsupervised SR on real lensing images.
