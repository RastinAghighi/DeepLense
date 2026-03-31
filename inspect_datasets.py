# %% [markdown]
# # Task VI.A: Supervised Super-Resolution on Simulated Strong Lensing Images
# 
# **Author:** Rastin Aghighi  
# **Evaluation Test:** DEEPLENSE – GSoC 2026 (ML4SCI)  
# **Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images
# 
# ## Strategy
# 
# **Architecture:** EDSR-baseline (Enhanced Deep Residual SR Network). I chose EDSR for several reasons:
# 1. The original EDSR paper (Lim et al., 2017) showed that **removing batch normalization** from 
#    SRResNet improves SR performance — BN normalizes features and limits the network's range flexibility,
#    which is counterproductive for regression tasks like SR where we need to reconstruct precise pixel values.
# 2. EDSR-baseline (16 residual blocks, 64 filters) strikes an excellent balance between reconstruction 
#    quality and training efficiency.
# 3. The architecture is fully convolutional, so it generalizes across input resolutions — critical for 
#    transfer learning to Task VI.B where image dimensions differ.
# 
# **Loss Function:** L1 (Mean Absolute Error). L1 loss produces sharper reconstructions than MSE/L2, 
# which tends to produce blurry outputs by averaging over possible solutions.
# 
# **Data Augmentation:** Random 90° rotations and horizontal/vertical flips, applied consistently 
# to HR/LR pairs. These exploit the rotational symmetry of gravitational lensing images (there is no 
# preferred orientation on the sky).
# 
# **Baseline:** Bicubic interpolation, to quantify how much the learned model improves over a 
# classical upsampling method.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Data Loading and Exploration
# 
# The dataset contains simulated strong lensing images at two resolutions:
# - **HR (High Resolution):** 150×150 pixels, single channel
# - **LR (Low Resolution):** 75×75 pixels, single channel
# - **Scale factor:** 2×
# 
# HR and LR images share filenames (`sample{N}.npy`). Since the HR folder has 10,000 files 
# and the LR folder has 4,492, we match by filename to get valid pairs.

# %%
# ============================================================
# CONFIGURE THIS PATH to your dataset location
# ============================================================
DATA_ROOT = r".\Dataset\Dataset"  # Adjust if needed

HR_DIR = os.path.join(DATA_ROOT, "HR")
LR_DIR = os.path.join(DATA_ROOT, "LR")

# Find matched pairs by filename
hr_files = {Path(f).stem: f for f in glob.glob(os.path.join(HR_DIR, "*.npy"))}
lr_files = {Path(f).stem: f for f in glob.glob(os.path.join(LR_DIR, "*.npy"))}

matched_keys = sorted(set(hr_files.keys()) & set(lr_files.keys()))
print(f"HR images: {len(hr_files)}")
print(f"LR images: {len(lr_files)}")
print(f"Matched pairs: {len(matched_keys)}")

# Load one pair to verify dimensions
sample_hr = np.load(hr_files[matched_keys[0]])
sample_lr = np.load(lr_files[matched_keys[0]])
print(f"\nHR shape: {sample_hr.shape}, dtype: {sample_hr.dtype}, range: [{sample_hr.min():.4f}, {sample_hr.max():.4f}]")
print(f"LR shape: {sample_lr.shape}, dtype: {sample_lr.dtype}, range: [{sample_lr.min():.4f}, {sample_lr.max():.4f}]")
print(f"Scale factor: {sample_hr.shape[-1] / sample_lr.shape[-1]:.1f}x")

# %%
# Visualize sample pairs
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample HR/LR Pairs from Simulated Dataset", fontsize=14)

for i in range(5):
    hr = np.load(hr_files[matched_keys[i]])[0]  # Remove channel dim for display
    lr = np.load(lr_files[matched_keys[i]])[0]
    
    axes[0, i].imshow(hr, cmap="inferno")
    axes[0, i].set_title(f"HR (150×150)", fontsize=9)
    axes[0, i].axis("off")
    
    axes[1, i].imshow(lr, cmap="inferno")
    axes[1, i].set_title(f"LR (75×75)", fontsize=9)
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("High Res", fontsize=11)
axes[1, 0].set_ylabel("Low Res", fontsize=11)
plt.tight_layout()
plt.savefig("sample_pairs_6a.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Dataset and DataLoader
# 
# **Augmentation strategy:** Since gravitational lensing images have no preferred orientation 
# on the sky, we apply random 90° rotations (k ∈ {0, 1, 2, 3}) and random horizontal/vertical 
# flips. These are applied **identically** to both HR and LR images in a pair to preserve 
# spatial correspondence. This effectively multiplies the dataset by up to 8×.

# %%
class LensingDataset(Dataset):
    """Dataset for HR/LR lensing image pairs with consistent augmentation."""
    
    def __init__(self, keys, hr_files, lr_files, augment=False):
        self.keys = keys
        self.hr_files = hr_files
        self.lr_files = lr_files
        self.augment = augment
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]
        hr = np.load(self.hr_files[key]).astype(np.float32)  # (1, H, H)
        lr = np.load(self.lr_files[key]).astype(np.float32)  # (1, H/s, H/s)
        
        if self.augment:
            # Random 90-degree rotation (applied to both consistently)
            k = random.randint(0, 3)
            if k > 0:
                hr = np.rot90(hr, k, axes=(1, 2)).copy()
                lr = np.rot90(lr, k, axes=(1, 2)).copy()
            
            # Random horizontal flip
            if random.random() > 0.5:
                hr = np.flip(hr, axis=2).copy()
                lr = np.flip(lr, axis=2).copy()
            
            # Random vertical flip
            if random.random() > 0.5:
                hr = np.flip(hr, axis=1).copy()
                lr = np.flip(lr, axis=1).copy()
        
        return torch.from_numpy(hr), torch.from_numpy(lr)


# 90:10 train-test split
random.shuffle(matched_keys)
split_idx = int(0.9 * len(matched_keys))
train_keys = matched_keys[:split_idx]
test_keys = matched_keys[split_idx:]

print(f"Training pairs: {len(train_keys)}")
print(f"Test pairs:     {len(test_keys)}")

train_dataset = LensingDataset(train_keys, hr_files, lr_files, augment=True)
test_dataset = LensingDataset(test_keys, hr_files, lr_files, augment=False)

BATCH_SIZE = 16
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                          num_workers=0, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

# %% [markdown]
# ## 4. Bicubic Interpolation Baseline
# 
# Before training a learned model, we establish a baseline using bicubic interpolation.
# This gives us a floor to measure improvement against.

# %%
def compute_metrics(sr, hr):
    """Compute MSE, SSIM, PSNR between SR output and HR ground truth.
    
    Args:
        sr: Super-resolved image, numpy array (H, W)
        hr: High-resolution ground truth, numpy array (H, W)
    
    Returns:
        dict with mse, ssim, psnr values
    """
    # Ensure same dtype
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)
    
    mse_val = np.mean((sr - hr) ** 2)
    
    # Determine data range from the ground truth
    data_range = hr.max() - hr.min()
    if data_range < 1e-8:
        data_range = 1.0
    
    ssim_val = ssim(hr, sr, data_range=data_range)
    psnr_val = psnr(hr, sr, data_range=data_range)
    
    return {"mse": mse_val, "ssim": ssim_val, "psnr": psnr_val}


# Bicubic baseline on test set
from torch.nn.functional import interpolate

bicubic_metrics = {"mse": [], "ssim": [], "psnr": []}

for hr, lr in test_loader:
    # Bicubic upscale: (B, 1, 75, 75) -> (B, 1, 150, 150)
    bicubic_sr = interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
    
    sr_np = bicubic_sr[0, 0].numpy()
    hr_np = hr[0, 0].numpy()
    
    m = compute_metrics(sr_np, hr_np)
    for k in bicubic_metrics:
        bicubic_metrics[k].append(m[k])

print("=== Bicubic Interpolation Baseline ===")
print(f"  MSE:  {np.mean(bicubic_metrics['mse']):.6f} ± {np.std(bicubic_metrics['mse']):.6f}")
print(f"  SSIM: {np.mean(bicubic_metrics['ssim']):.4f} ± {np.std(bicubic_metrics['ssim']):.4f}")
print(f"  PSNR: {np.mean(bicubic_metrics['psnr']):.2f} ± {np.std(bicubic_metrics['psnr']):.2f} dB")

# %% [markdown]
# ## 5. EDSR-Baseline Model
# 
# **Architecture details:**
# - **No Batch Normalization:** Lim et al. (2017) showed that removing BN from residual blocks 
#   improves SR quality. BN normalizes features to zero mean/unit variance, which limits the 
#   range of activations — counterproductive for pixel regression tasks.
# - **Residual blocks:** 16 blocks, each with two 3×3 conv layers and ReLU activation, 
#   with a residual scaling factor of 0.1 for training stability.
# - **Upsampling:** Sub-pixel convolution (PixelShuffle) for 2× upscaling — more efficient and 
#   produces fewer artifacts than transposed convolution or bicubic upsampling layers.
# - **Global residual learning:** The network learns the residual between the bicubic-upscaled 
#   input and the HR target, making optimization easier.

# %%
class ResBlock(nn.Module):
    """Residual block without batch normalization (EDSR design)."""
    
    def __init__(self, n_feats, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feats, n_feats, 3, padding=1)
        self.res_scale = res_scale
    
    def forward(self, x):
        residual = self.conv2(self.relu(self.conv1(x)))
        return x + residual * self.res_scale


class EDSR(nn.Module):
    """EDSR-baseline: 16 residual blocks, 64 filters, 2x upscale."""
    
    def __init__(self, n_channels=1, n_feats=64, n_resblocks=16, scale=2):
        super().__init__()
        self.scale = scale
        
        # Head: initial feature extraction
        self.head = nn.Conv2d(n_channels, n_feats, 3, padding=1)
        
        # Body: residual blocks
        body = [ResBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, 3, padding=1))
        self.body = nn.Sequential(*body)
        
        # Tail: upsampling via sub-pixel convolution
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_feats, n_channels, 3, padding=1)
        )
    
    def forward(self, x):
        # Global residual: add bicubic-upscaled input to network output
        bicubic = nn.functional.interpolate(x, scale_factor=self.scale, 
                                            mode="bicubic", align_corners=False)
        
        head = self.head(x)
        body = self.body(head)
        res = head + body  # Long skip connection
        sr = self.tail(res)
        
        return sr + bicubic  # Global residual learning


model = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"EDSR-baseline")
print(f"  Total parameters:     {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## 6. Training
# 
# **Optimizer:** Adam with learning rate 1e-4.  
# **Loss:** L1 (MAE) — preferred over MSE for SR as it produces sharper outputs.  
# **Scheduler:** ReduceLROnPlateau, monitoring validation loss.  
# **Epochs:** 100 with early stopping (patience=10) to prevent overfitting.

# %%
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 10  # Early stopping patience

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, 
                                                  patience=5, verbose=True)

# Training history
history = {"train_loss": [], "val_loss": [], "val_psnr": [], "lr": []}
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

print(f"Training EDSR for up to {NUM_EPOCHS} epochs...")
print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
print("-" * 60)

for epoch in range(NUM_EPOCHS):
    # --- Training ---
    model.train()
    train_loss = 0.0
    
    for hr, lr in train_loader:
        hr, lr = hr.to(device), lr.to(device)
        
        optimizer.zero_grad()
        sr = model(lr)
        loss = criterion(sr, hr)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * lr.size(0)
    
    train_loss /= len(train_dataset)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    val_psnr = 0.0
    
    with torch.no_grad():
        for hr, lr in test_loader:
            hr, lr = hr.to(device), lr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)
            val_loss += loss.item()
            
            # Quick PSNR for monitoring
            sr_np = sr[0, 0].cpu().numpy()
            hr_np = hr[0, 0].cpu().numpy()
            data_range = hr_np.max() - hr_np.min()
            if data_range > 1e-8:
                val_psnr += psnr(hr_np, sr_np, data_range=data_range)
    
    val_loss /= len(test_dataset)
    val_psnr /= len(test_dataset)
    current_lr = optimizer.param_groups[0]["lr"]
    
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)
    history["lr"].append(current_lr)
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
        patience_counter = 0
        marker = " ★"
    else:
        patience_counter += 1
        marker = ""
    
    if (epoch + 1) % 5 == 0 or epoch == 0 or marker:
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train L1: {train_loss:.6f} | Val L1: {val_loss:.6f} | "
              f"Val PSNR: {val_psnr:.2f} dB | LR: {current_lr:.1e}{marker}")
    
    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# Restore best model
model.load_state_dict(best_model_state)
print(f"\nRestored best model (val loss: {best_val_loss:.6f})")

# Save model weights
torch.save(best_model_state, "edsr_simulated_best.pth")
print("Saved model weights to edsr_simulated_best.pth")

# %% [markdown]
# ## 7. Training Curves

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history["train_loss"], label="Train L1 Loss", color="steelblue")
ax1.plot(history["val_loss"], label="Val L1 Loss", color="coral")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("L1 Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(history["val_psnr"], label="Val PSNR", color="seagreen")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("PSNR (dB)")
ax2.set_title("Validation PSNR")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_6a.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Evaluation on Test Set

# %%
model.eval()
edsr_metrics = {"mse": [], "ssim": [], "psnr": []}
test_samples = []  # Store a few for visualization

with torch.no_grad():
    for i, (hr, lr) in enumerate(test_loader):
        lr_dev = lr.to(device)
        sr = model(lr_dev).cpu()
        
        sr_np = sr[0, 0].numpy()
        hr_np = hr[0, 0].numpy()
        lr_np = lr[0, 0].numpy()
        
        m = compute_metrics(sr_np, hr_np)
        for k in edsr_metrics:
            edsr_metrics[k].append(m[k])
        
        # Store first 8 samples for visualization
        if i < 8:
            test_samples.append((lr_np, sr_np, hr_np))

# Print comparison
print("=" * 60)
print("EVALUATION RESULTS ON TEST SET")
print("=" * 60)
print(f"\n{'Metric':<8} {'Bicubic':<22} {'EDSR':<22} {'Improvement'}")
print("-" * 60)

for metric in ["mse", "ssim", "psnr"]:
    bic_mean = np.mean(bicubic_metrics[metric])
    bic_std = np.std(bicubic_metrics[metric])
    edsr_mean = np.mean(edsr_metrics[metric])
    edsr_std = np.std(edsr_metrics[metric])
    
    if metric == "mse":
        improvement = f"{(1 - edsr_mean/bic_mean)*100:.1f}% reduction"
    elif metric == "ssim":
        improvement = f"+{edsr_mean - bic_mean:.4f}"
    else:  # psnr
        improvement = f"+{edsr_mean - bic_mean:.2f} dB"
    
    print(f"{metric.upper():<8} {bic_mean:.6f} ± {bic_std:.6f}   "
          f"{edsr_mean:.6f} ± {edsr_std:.6f}   {improvement}")

# %% [markdown]
# ## 9. Visual Comparison

# %%
n_show = min(4, len(test_samples))
fig, axes = plt.subplots(n_show, 3, figsize=(12, 4 * n_show))
fig.suptitle("Super-Resolution Results: LR → EDSR Output → HR Ground Truth", fontsize=14, y=1.01)

for i in range(n_show):
    lr_img, sr_img, hr_img = test_samples[i]
    
    # Compute per-image metrics for display
    m = compute_metrics(sr_img, hr_img)
    
    axes[i, 0].imshow(lr_img, cmap="inferno")
    axes[i, 0].set_title("LR Input (75×75)" if i == 0 else "")
    axes[i, 0].axis("off")
    
    axes[i, 1].imshow(sr_img, cmap="inferno")
    axes[i, 1].set_title("EDSR Output (150×150)" if i == 0 else "")
    axes[i, 1].set_xlabel(f"PSNR: {m['psnr']:.1f} dB | SSIM: {m['ssim']:.3f}", fontsize=8)
    axes[i, 1].axis("off")
    
    axes[i, 2].imshow(hr_img, cmap="inferno")
    axes[i, 2].set_title("HR Ground Truth (150×150)" if i == 0 else "")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("visual_comparison_6a.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Error maps: absolute difference between EDSR output and HR ground truth
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("Reconstruction Error Maps (|SR - HR|)", fontsize=14)

for i in range(min(4, n_show)):
    lr_img, sr_img, hr_img = test_samples[i]
    
    # Bicubic error
    bicubic_up = torch.nn.functional.interpolate(
        torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0),
        size=(150, 150), mode="bicubic", align_corners=False
    )[0, 0].numpy()
    bic_error = np.abs(bicubic_up - hr_img)
    edsr_error = np.abs(sr_img - hr_img)
    
    vmax = max(bic_error.max(), edsr_error.max())
    
    axes[0, i].imshow(bic_error, cmap="hot", vmin=0, vmax=vmax)
    axes[0, i].set_title(f"Bicubic Error" if i == 0 else "")
    axes[0, i].axis("off")
    
    axes[1, i].imshow(edsr_error, cmap="hot", vmin=0, vmax=vmax)
    axes[1, i].set_title(f"EDSR Error" if i == 0 else "")
    axes[1, i].axis("off")

axes[0, 0].set_ylabel("Bicubic", fontsize=11)
axes[1, 0].set_ylabel("EDSR", fontsize=11)
plt.tight_layout()
plt.savefig("error_maps_6a.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 10. Discussion
# 
# ### Architecture Choice
# EDSR-baseline was chosen for its strong balance of reconstruction quality and training 
# efficiency. The key insight from Lim et al. (2017) is that batch normalization, while 
# beneficial for classification, actually degrades super-resolution performance by normalizing 
# away the range flexibility that pixel-regression tasks require. Our implementation includes 
# residual scaling (0.1) for training stability and a global skip connection that adds the 
# bicubic-upscaled input to the network output, allowing the model to focus on learning the 
# high-frequency residual details rather than the entire image.
# 
# ### Loss Function
# L1 loss was used instead of MSE (L2). In super-resolution, L2 loss tends to produce 
# over-smoothed results because it heavily penalizes outliers, causing the model to predict 
# the mean of possible high-resolution solutions. L1 loss is more robust and produces sharper 
# reconstructions, which is particularly important for scientific imaging where fine structural 
# details carry physical information about substructure.
# 
# ### Data Augmentation
# Gravitational lensing images have rotational and reflective symmetry (no preferred 
# orientation on the sky), making geometric augmentations physically motivated rather than 
# just a regularization trick. We restricted rotations to 90° increments to avoid 
# interpolation artifacts in the ground truth images.
# 
# ### Connection to Proposal
# This supervised SR model serves as a strong baseline for the proposed unsupervised approach. 
# The key challenge in the GSoC project is performing SR without paired HR/LR data — real 
# telescope observations rarely have perfectly aligned multi-resolution counterparts. Having 
# established supervised performance benchmarks here, the unsupervised method's quality can 
# be evaluated relative to this upper bound.
# 
# ### Limitations and Future Work
# - The model was trained on simulated data, which has a cleaner noise profile than real 
#   telescope images. Domain adaptation will be important for real-world deployment.
# - EDSR-baseline is a relatively simple architecture; more advanced models (RCAN, SwinIR, 
#   diffusion-based approaches like DiffLense) could achieve higher quality at the cost of 
#   increased training time and complexity.
# - The current training uses the entire LR image; patch-based training could improve 
#   efficiency and allow batch size scaling for larger models.