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
# # Task VI.A: Supervised Super-Resolution on Simulated Strong Lensing Images
# **Author:** Rastin Aghighi
# **DEEPLENSE – GSoC 2026 (ML4SCI)**
# **Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images
#
# ## Strategy
#
# We adopt the Enhanced Deep Residual Network for Single Image Super-Resolution
# (EDSR-baseline) as our backbone. EDSR deliberately removes batch normalisation
# from the residual blocks, which is a critical design choice for super-resolution
# tasks. Batch normalisation introduces normalisation artifacts at inference time
# that blur fine-grained texture — exactly the detail we need to recover in
# gravitational lensing arcs and substructure. By omitting BN, the network
# preserves the full dynamic range of the residual signal, leading to sharper
# reconstructions with fewer parameters. Our EDSR-baseline uses 16 residual blocks
# with 64 feature channels, striking a practical balance between capacity and
# training efficiency on a single-GPU budget.
#
# The composite loss function is designed to capture different facets of
# reconstruction quality that matter for downstream physics analysis. The L1 (MAE)
# term drives pixel-level fidelity and is more robust to outliers than MSE,
# preventing the network from over-smoothing bright cores. The Structural
# Similarity (SSIM) loss preserves perceptually important luminance, contrast, and
# structural patterns — particularly the thin arcs and Einstein rings whose
# morphology encodes the lens mass distribution. Finally, a Fourier-space loss
# penalises discrepancies in the power spectrum, encouraging the network to
# reconstruct the correct spatial-frequency content rather than a blurry
# low-frequency approximation. Together, these three terms guide the model toward
# outputs that are accurate in pixel space, structurally coherent, and
# frequency-faithful.
#
# At inference time we apply the self-ensemble strategy (EDSR+): each test image
# is augmented by eight geometric transforms (four rotations × two flips), the
# model predicts a super-resolved output for each, and the inverse transforms are
# applied before averaging the eight predictions. This test-time augmentation
# exploits the equivariance gap that a finite training set cannot fully close,
# yielding a consistent 0.1–0.3 dB PSNR improvement at no extra training cost.
# For data augmentation during training we use random horizontal/vertical flips
# and 90° rotations — standard in SR literature — to expose the model to all
# orientations and improve generalisation without distorting the astrophysical
# signal.

# %% [markdown]
# ## 1. Setup and Imports

# %%
import os
import sys
import glob
import random

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

from src.edsr import EDSR
from src.losses import CompositeSRLoss
from src.dataset import load_sr_pairs, train_test_split, LensingSRDataset
from src.metrics import compute_metrics, bootstrap_ci, format_metric_row, self_ensemble_predict
from src.visualization import (
    plot_sample_pairs,
    plot_training_curves,
    plot_visual_comparison,
    plot_error_maps,
    plot_metric_distributions,
    plot_failure_analysis,
    plot_ablation_table,
)

# ── Reproducibility ──────────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

# ── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 10

# %% [markdown]
# ## 2. Data Loading and Exploration

# %%
DATA_ROOT = os.path.join("..", "Dataset", "Dataset")

hr_dir = os.path.join(DATA_ROOT, "HR")
lr_dir = os.path.join(DATA_ROOT, "LR")

keys, hr_dict, lr_dict = load_sr_pairs(hr_dir, lr_dir)
print(f"Matched pairs: {len(keys)}")

# Show a sample shape
sample_hr = np.load(hr_dict[keys[0]])
sample_lr = np.load(lr_dict[keys[0]])
print(f"HR shape: {sample_hr.shape}  |  LR shape: {sample_lr.shape}")

train_keys, test_keys = train_test_split(keys, train_ratio=0.9, seed=42)
print(f"Train: {len(train_keys)}  |  Test: {len(test_keys)}")

# %%
# ── Dataset-wide intensity statistics (200-sample survey) ────────────────────
N_SURVEY = min(200, len(keys))

hr_mins, hr_maxs = [], []
lr_mins, lr_maxs = [], []

for k in keys[:N_SURVEY]:
    hr_img = np.load(hr_dict[k])
    lr_img = np.load(lr_dict[k])
    hr_mins.append(hr_img.min())
    hr_maxs.append(hr_img.max())
    lr_mins.append(lr_img.min())
    lr_maxs.append(lr_img.max())

print(f"HR intensity range across {N_SURVEY} samples: "
      f"[{min(hr_mins):.4f}, {max(hr_maxs):.4f}]")
print(f"LR intensity range across {N_SURVEY} samples: "
      f"[{min(lr_mins):.4f}, {max(lr_maxs):.4f}]")
print("Values outside [0,1] are physically meaningful "
      "and will NOT be clipped during training.")

# %%
# ── Visualise sample HR / LR pairs ──────────────────────────────────────────
hr_samples = [np.load(hr_dict[keys[i]])[0] for i in range(5)]
lr_samples = [np.load(lr_dict[keys[i]])[0] for i in range(5)]

os.makedirs(os.path.join("..", "figures"), exist_ok=True)
plot_sample_pairs(
    hr_samples, lr_samples, n=5,
    save_path=os.path.join("..", "figures", "sample_pairs_6a.png"),
)

# %% [markdown]
# ## 3. Dataset and DataLoader

# %%
train_dataset = LensingSRDataset(train_keys, hr_dict, lr_dict, augment=True)
test_dataset = LensingSRDataset(test_keys, hr_dict, lr_dict, augment=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=True,
)
test_loader = DataLoader(
    test_dataset, batch_size=1, shuffle=False,
)

print(f"Train batches: {len(train_loader)}  |  Test samples: {len(test_loader)}")

# %% [markdown]
# ## 4. Bicubic Interpolation Baseline
#
# Before training a learned model, we establish a performance floor using bicubic
# interpolation. This classical method uses a cubic polynomial kernel to upsample
# — it produces smooth results but cannot recover high-frequency details lost in
# the downsampling process.

# %%
import torch.nn.functional as F

bicubic_metrics = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}

with torch.no_grad():
    for hr, lr in tqdm(test_loader, desc="Bicubic baseline"):
        sr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        sr_np = sr[0, 0].numpy()
        hr_np = hr[0, 0].numpy()
        m = compute_metrics(sr_np, hr_np)
        for k in bicubic_metrics:
            bicubic_metrics[k].append(m[k])

print("Bicubic Interpolation Baseline")
print("-" * 50)
for k in bicubic_metrics:
    print(format_metric_row(k, bicubic_metrics[k]))

# %% [markdown]
# ## 5. Model Definition
#
# We use the EDSR-baseline architecture (Lim et al., CVPRW 2017) with 16 residual
# blocks and 64 feature channels at ×2 scale. As discussed in the strategy section,
# EDSR omits batch normalisation from its residual blocks — this is essential for
# super-resolution because BN artifacts blur the fine-grained texture of
# gravitational lensing arcs and substructure. The 16-block / 64-channel
# configuration balances reconstruction quality with single-GPU training efficiency.

# %%
model = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters:     {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# %% [markdown]
# ## 6. Composite Loss Function
#
# The composite loss is designed so that each term addresses a distinct aspect of
# reconstruction fidelity that matters for downstream physics analysis:
#
# **L1 (MAE)** — pixel-wise mean absolute error. L1 is preferred over MSE because
# MSE penalises large errors quadratically, pushing predictions toward the
# conditional mean and producing over-smoothed outputs. L1 preserves sharper
# edges in bright lens cores and faint arc structures.
#
# **Flux consistency** (λ_flux = 0.05) — penalises the difference in total
# integrated intensity: |Σ SR − Σ HR| / N_pixels. Gravitational lensing
# conserves photon flux, so the super-resolved image must preserve the same
# total brightness as the ground truth. This soft constraint encodes that
# physical invariant directly into training.
#
# **Back-projection** (λ_bp = 0.1) — L1(downsample(SR), LR). This term
# enforces consistency with the observed low-resolution input: if we
# downsample the super-resolved output through the degradation model, it
# should reproduce the original LR image. This is a key insight borrowed
# from unsupervised SR — it requires only knowledge of the degradation
# operator, not paired data.
#
# $$\mathcal{L}_{\text{total}} = \mathcal{L}_1(\text{SR}, \text{HR})
#   + 0.05 \cdot \mathcal{L}_{\text{flux}}(\text{SR}, \text{HR})
#   + 0.1  \cdot \mathcal{L}_{\text{bp}}(\text{SR}, \text{LR})$$

# %%
criterion = CompositeSRLoss(lambda_flux=0.05, lambda_bp=0.1)

print("Loss: L_total = L1(SR, HR) + λ_flux·L_flux(SR, HR) + λ_bp·L_bp(SR, LR)")
print(f"  λ_flux = {criterion.lambda_flux}")
print(f"  λ_bp   = {criterion.lambda_bp}")

# %% [markdown]
# ## 7. Training (Composite Loss)
#
# **Optimizer:** Adam (Kingma & Ba, 2015) with `lr = 1e-4`. Adam's per-parameter
# adaptive learning rates handle the mix of residual-block weights and the
# pixel-shuffle upsampler well without manual tuning.
#
# **Loss:** The composite loss defined in §6 — L1 for pixel fidelity, flux
# consistency to respect photon conservation, and back-projection to enforce
# degradation-model consistency.
#
# **Scheduler:** `ReduceLROnPlateau` (factor 0.5, patience 5) monitors
# validation loss and halves the learning rate when progress stalls, letting the
# network fine-tune into sharper minima in later epochs.
#
# **Early stopping:** We track validation loss and restore the best checkpoint
# after `PATIENCE = 10` epochs without improvement. This guards against
# over-fitting on the relatively small simulated dataset and avoids wasting
# compute on diminishing returns.

# %%
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

history = {
    "train_loss": [], "train_l1": [], "train_flux": [], "train_bp": [],
    "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": [],
}

best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

for epoch in range(NUM_EPOCHS):
    # ── Training phase ──────────────────────────────────────────────────────
    model.train()
    running_loss, running_l1, running_flux, running_bp = 0.0, 0.0, 0.0, 0.0
    n_train = 0

    for hr, lr in tqdm(train_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [train]", leave=False):
        hr, lr = hr.to(device), lr.to(device)
        optimizer.zero_grad()
        sr = model(lr)
        loss, parts = criterion(sr, hr, lr)
        loss.backward()
        optimizer.step()

        bs = hr.size(0)
        running_loss += parts["total"] * bs
        running_l1 += parts["l1"] * bs
        running_flux += parts["flux"] * bs
        running_bp += parts["bp"] * bs
        n_train += bs

    history["train_loss"].append(running_loss / n_train)
    history["train_l1"].append(running_l1 / n_train)
    history["train_flux"].append(running_flux / n_train)
    history["train_bp"].append(running_bp / n_train)

    # ── Validation phase ────────────────────────────────────────────────────
    model.eval()
    running_val_loss = 0.0
    running_psnr, running_ssim = 0.0, 0.0
    n_val = 0

    with torch.no_grad():
        for hr, lr in tqdm(test_loader, desc=f"Ep {epoch+1}/{NUM_EPOCHS} [val]", leave=False):
            hr, lr = hr.to(device), lr.to(device)
            sr = model(lr)
            loss, _ = criterion(sr, hr, lr)

            sr_np = sr.cpu().clamp(0, 1).squeeze().numpy()
            hr_np = hr.cpu().clamp(0, 1).squeeze().numpy()

            running_val_loss += loss.item()
            running_psnr += peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
            running_ssim += structural_similarity(hr_np, sr_np, data_range=1.0)
            n_val += 1

    avg_val_loss = running_val_loss / n_val
    avg_psnr = running_psnr / n_val
    avg_ssim = running_ssim / n_val

    history["val_loss"].append(avg_val_loss)
    history["val_psnr"].append(avg_psnr)
    history["val_ssim"].append(avg_ssim)
    history["lr"].append(optimizer.param_groups[0]["lr"])

    scheduler.step(avg_val_loss)

    # ── Logging ─────────────────────────────────────────────────────────────
    improved = avg_val_loss < best_val_loss
    if (epoch + 1) % 5 == 0 or improved:
        tag = " *" if improved else ""
        print(
            f"Ep {epoch+1}/{NUM_EPOCHS} | "
            f"Loss: {history['train_loss'][-1]:.4f} "
            f"(L1:{history['train_l1'][-1]:.4f} "
            f"Flux:{history['train_flux'][-1]:.4f} "
            f"BP:{history['train_bp'][-1]:.4f}) | "
            f"Val PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.1e}{tag}"
        )

    # ── Early stopping ──────────────────────────────────────────────────────
    if improved:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print(f"Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# ── Restore best model and save weights ─────────────────────────────────────
model.load_state_dict(best_model_state)
model.to(device)

weights_dir = os.path.join("..", "weights")
os.makedirs(weights_dir, exist_ok=True)
torch.save(best_model_state, os.path.join(weights_dir, "edsr_simulated_best.pth"))
print(f"Best model saved to weights/edsr_simulated_best.pth (val loss: {best_val_loss:.4f})")

# Save training history for offline visualization
import json

history_path = os.path.join(weights_dir, "edsr_simulated_history.json")
with open(history_path, "w") as f:
    json.dump(history, f)
print(f"Training history saved to {history_path}")

# %% [markdown]
# ## 8. Training Curves

# %%
# Load history from disk when running locally (training was done on Kaggle GPU).
import json

history_path = os.path.join("..", "weights", "edsr_simulated_history.json")
if not history.get("train_loss"):
    with open(history_path) as f:
        history = json.load(f)
    print(f"Loaded training history from {history_path}")
else:
    # Save history for future offline use
    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"Saved training history to {history_path}")

plot_training_curves(
    history,
    save_path=os.path.join("..", "figures", "training_curves_6a.png"),
)

# %% [markdown]
# ## 9. Self-Ensemble Inference (EDSR+)
#
# We run the model on 8 augmented versions of each LR input (4 rotations × 2 flip
# states) and average the de-augmented outputs. This geometric self-ensemble
# suppresses directional artifacts and typically adds +0.1 to +0.3 dB PSNR without
# retraining.

# %% [markdown]
# ## 10. Full Evaluation on Test Set

# %%
model.eval()

results_standard = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}
results_ensemble = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}
test_samples = []

with torch.no_grad():
    for hr, lr in tqdm(test_loader, desc="Evaluating"):
        # Standard inference
        sr_std = model(lr.to(device)).cpu()

        # Ensemble inference
        sr_ens = self_ensemble_predict(model, lr, device)

        # Convert to numpy
        hr_np = hr[0, 0].numpy()
        lr_np = lr[0, 0].numpy()
        sr_std_np = sr_std[0, 0].numpy()
        sr_ens_np = sr_ens[0, 0].numpy()

        # Compute metrics for both
        m_std = compute_metrics(sr_std_np, hr_np)
        m_ens = compute_metrics(sr_ens_np, hr_np)

        for k in results_standard:
            results_standard[k].append(m_std[k])
            results_ensemble[k].append(m_ens[k])

        # Store first 10 samples for visualization
        if len(test_samples) < 10:
            test_samples.append((lr_np, sr_std_np, sr_ens_np, hr_np))

# ── Comprehensive results table ────────────────────────────────────────────
print("=" * 80)
print("Full Evaluation Results")
print("=" * 80)
print(f"{'Metric':<14} {'Bicubic':<22} {'EDSR':<22} {'EDSR+':<22}")
print("-" * 80)

for k in ["psnr", "ssim", "mse", "flux_error"]:
    bic_mean = np.mean(bicubic_metrics[k])
    bic_std = np.std(bicubic_metrics[k])
    bic_med = np.median(bicubic_metrics[k])
    bic_lo, bic_hi = bootstrap_ci(bicubic_metrics[k])

    std_mean = np.mean(results_standard[k])
    std_std = np.std(results_standard[k])
    std_med = np.median(results_standard[k])
    std_lo, std_hi = bootstrap_ci(results_standard[k])

    ens_mean = np.mean(results_ensemble[k])
    ens_std = np.std(results_ensemble[k])
    ens_med = np.median(results_ensemble[k])
    ens_lo, ens_hi = bootstrap_ci(results_ensemble[k])

    print(f"{k.upper():<14} "
          f"{bic_mean:.4f}±{bic_std:.4f}  "
          f"{std_mean:.4f}±{std_std:.4f}  "
          f"{ens_mean:.4f}±{ens_std:.4f}")
    print(f"{'  median':<14} "
          f"{bic_med:.4f}           "
          f"{std_med:.4f}           "
          f"{ens_med:.4f}")
    print(f"{'  95% CI':<14} "
          f"[{bic_lo:.4f},{bic_hi:.4f}] "
          f"[{std_lo:.4f},{std_hi:.4f}] "
          f"[{ens_lo:.4f},{ens_hi:.4f}]")

# ── Improvement summary ────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("Improvement Summary (EDSR+ over Bicubic)")
print("=" * 80)
psnr_gain = np.mean(results_ensemble["psnr"]) - np.mean(bicubic_metrics["psnr"])
ssim_gain = np.mean(results_ensemble["ssim"]) - np.mean(bicubic_metrics["ssim"])
mse_reduction = (1 - np.mean(results_ensemble["mse"]) / np.mean(bicubic_metrics["mse"])) * 100

print(f"PSNR gain:      +{psnr_gain:.2f} dB")
print(f"SSIM gain:      +{ssim_gain:.4f}")
print(f"MSE reduction:  {mse_reduction:.1f}%")

ens_psnr_gain = np.mean(results_ensemble["psnr"]) - np.mean(results_standard["psnr"])
print(f"\nEnsemble boost: +{ens_psnr_gain:.3f} dB PSNR (EDSR+ vs EDSR)")

# %% [markdown]
# ## 11. Metric Distribution Plots

# %%
plot_metric_distributions(
    [bicubic_metrics, results_standard, results_ensemble],
    ["Bicubic", "EDSR", "EDSR+"],
    save_path=os.path.join("..", "figures", "metric_distributions_6a.png"),
)

# %% [markdown]
# ## 12. Visual Comparison

# %%
plot_visual_comparison(
    samples=[(s[0], s[1], s[2], s[3]) for s in test_samples[:5]],
    col_titles=["LR Input (75×75)", "EDSR Output", "EDSR+ (Ensemble)", "HR Ground Truth (150×150)"],
    save_path=os.path.join("..", "figures", "visual_comparison_6a.png"),
)

# %% [markdown]
# ## 13. Error Maps

# %%
error_map_samples = []
for lr_np, sr_std_np, sr_ens_np, hr_np in test_samples[:5]:
    bicubic_up = F.interpolate(
        torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0),
        size=hr_np.shape, mode="bicubic", align_corners=False,
    )[0, 0].numpy()
    bic_err = np.abs(bicubic_up - hr_np)
    std_err = np.abs(sr_std_np - hr_np)
    ens_err = np.abs(sr_ens_np - hr_np)
    error_map_samples.append((bic_err, std_err, ens_err))

plot_error_maps(
    samples=error_map_samples,
    method_names=["Bicubic", "EDSR", "EDSR+"],
    hr_images=[s[3] for s in test_samples[:5]],
    save_path=os.path.join("..", "figures", "error_maps_6a.png"),
)

# %% [markdown]
# ## 14. Failure Analysis

# %%
worst_indices = np.argsort(results_ensemble["psnr"])[:5]

worst_samples = []
model.eval()
with torch.no_grad():
    for idx in worst_indices:
        hr, lr = test_dataset[idx]
        lr_tensor = lr.unsqueeze(0)
        sr = self_ensemble_predict(model, lr_tensor, device)

        lr_np = lr[0].numpy()
        sr_np = sr[0, 0].numpy()
        hr_np = hr[0].numpy()
        metrics_dict = compute_metrics(sr_np, hr_np)
        worst_samples.append((lr_np, sr_np, hr_np, metrics_dict))

plot_failure_analysis(
    worst_samples,
    save_path=os.path.join("..", "figures", "failure_analysis_6a.png"),
)

print("Failure Analysis Notes:")
print("-" * 50)
print("Worst PSNR images tend to have bright, compact central sources where")
print("small spatial errors produce large intensity mismatches.")
print()
print("EDSR's L1 loss treats all errors equally — perceptual or adversarial")
print("losses could preserve sharp features better.")

# %% [markdown]
# ## 15. Ablation Study: L1-only vs Composite Loss
#
# To validate the composite loss, we train an identical EDSR model using only L1
# loss (no flux consistency, no back-projection). Same architecture, same seed
# (42), same train/test split, same augmentation, same optimizer, same scheduler.
# The only difference is the loss function.

# %%
# ── Reproducibility (reset seed for fair comparison) ──────────────────────────
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

model_l1 = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2).to(device)

criterion_l1 = nn.L1Loss()
optimizer_l1 = Adam(model_l1.parameters(), lr=LEARNING_RATE)
scheduler_l1 = ReduceLROnPlateau(optimizer_l1, mode="min", factor=0.5, patience=5)

history_l1 = {
    "train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": [],
}

best_val_loss_l1 = float("inf")
patience_counter_l1 = 0
best_model_state_l1 = None

for epoch in range(NUM_EPOCHS):
    # ── Training phase ────────────────────────────────────────────────────────
    model_l1.train()
    running_loss = 0.0
    n_train = 0

    for hr, lr in tqdm(train_loader, desc=f"[L1] Ep {epoch+1}/{NUM_EPOCHS} [train]", leave=False):
        hr, lr = hr.to(device), lr.to(device)
        optimizer_l1.zero_grad()
        sr = model_l1(lr)
        loss = criterion_l1(sr, hr)
        loss.backward()
        optimizer_l1.step()

        bs = hr.size(0)
        running_loss += loss.item() * bs
        n_train += bs

    history_l1["train_loss"].append(running_loss / n_train)

    # ── Validation phase ──────────────────────────────────────────────────────
    model_l1.eval()
    running_val_loss = 0.0
    running_psnr, running_ssim = 0.0, 0.0
    n_val = 0

    with torch.no_grad():
        for hr, lr in tqdm(test_loader, desc=f"[L1] Ep {epoch+1}/{NUM_EPOCHS} [val]", leave=False):
            hr, lr = hr.to(device), lr.to(device)
            sr = model_l1(lr)
            loss = criterion_l1(sr, hr)

            sr_np = sr.cpu().clamp(0, 1).squeeze().numpy()
            hr_np = hr.cpu().clamp(0, 1).squeeze().numpy()

            running_val_loss += loss.item()
            running_psnr += peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)
            running_ssim += structural_similarity(hr_np, sr_np, data_range=1.0)
            n_val += 1

    avg_val_loss = running_val_loss / n_val
    avg_psnr = running_psnr / n_val
    avg_ssim = running_ssim / n_val

    history_l1["val_loss"].append(avg_val_loss)
    history_l1["val_psnr"].append(avg_psnr)
    history_l1["val_ssim"].append(avg_ssim)
    history_l1["lr"].append(optimizer_l1.param_groups[0]["lr"])

    scheduler_l1.step(avg_val_loss)

    # ── Logging (every 5 epochs or on improvement) ────────────────────────────
    improved = avg_val_loss < best_val_loss_l1
    if (epoch + 1) % 5 == 0 or improved:
        tag = " *" if improved else ""
        print(
            f"[L1] Ep {epoch+1}/{NUM_EPOCHS} | "
            f"Loss: {history_l1['train_loss'][-1]:.4f} | "
            f"Val PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.4f} | "
            f"LR: {optimizer_l1.param_groups[0]['lr']:.1e}{tag}"
        )

    # ── Early stopping ────────────────────────────────────────────────────────
    if improved:
        best_val_loss_l1 = avg_val_loss
        patience_counter_l1 = 0
        best_model_state_l1 = {k: v.cpu().clone() for k, v in model_l1.state_dict().items()}
    else:
        patience_counter_l1 += 1

    if patience_counter_l1 >= PATIENCE:
        print(f"[L1] Early stopping at epoch {epoch+1} (no improvement for {PATIENCE} epochs)")
        break

# ── Restore best model and save weights ───────────────────────────────────────
model_l1.load_state_dict(best_model_state_l1)
model_l1.to(device)

torch.save(best_model_state_l1, os.path.join(weights_dir, "edsr_simulated_l1only.pth"))
print(f"L1-only model saved to weights/edsr_simulated_l1only.pth "
      f"(val loss: {best_val_loss_l1:.4f})")

# Save L1 training history
history_l1_path = os.path.join(weights_dir, "edsr_simulated_l1only_history.json")
with open(history_l1_path, "w") as f:
    json.dump(history_l1, f)
print(f"L1-only training history saved to {history_l1_path}")

# ── Evaluate L1-only model on test set with self-ensemble ─────────────────────
model_l1.eval()
results_l1_ensemble = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}

with torch.no_grad():
    for hr, lr in tqdm(test_loader, desc="L1-only EDSR+ evaluation"):
        sr_ens = self_ensemble_predict(model_l1, lr, device)

        hr_np = hr[0, 0].numpy()
        sr_ens_np = sr_ens[0, 0].numpy()

        m = compute_metrics(sr_ens_np, hr_np)
        for k in results_l1_ensemble:
            results_l1_ensemble[k].append(m[k])

print("\nL1-only EDSR+ Test Results")
print("-" * 50)
for k in results_l1_ensemble:
    print(format_metric_row(k, results_l1_ensemble[k]))

# %% [markdown]
# ## 16. Ablation Results Comparison

# %%
# ── Comparison table: Bicubic / L1-only EDSR+ / Composite EDSR+ ──────────────
print("=" * 90)
print("Ablation Study: Loss Function Comparison")
print("=" * 90)
print(f"{'Metric':<14} {'Bicubic':<24} {'L1-only EDSR+':<24} {'Composite EDSR+':<24}")
print("-" * 90)

for k in ["psnr", "ssim", "mse", "flux_error"]:
    bic = np.array(bicubic_metrics[k])
    l1 = np.array(results_l1_ensemble[k])
    comp = np.array(results_ensemble[k])

    print(f"{k.upper():<14} "
          f"{bic.mean():.4f} ± {bic.std():.4f}   "
          f"{l1.mean():.4f} ± {l1.std():.4f}   "
          f"{comp.mean():.4f} ± {comp.std():.4f}")

# ── Improvement deltas ────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("Improvement Deltas (Composite EDSR+ vs L1-only EDSR+)")
print("=" * 90)

for k in ["psnr", "ssim", "mse", "flux_error"]:
    l1_mean = np.mean(results_l1_ensemble[k])
    comp_mean = np.mean(results_ensemble[k])
    delta = comp_mean - l1_mean

    # For MSE and flux_error, lower is better
    if k in ("mse", "flux_error"):
        pct = (1 - comp_mean / l1_mean) * 100 if l1_mean != 0 else 0.0
        direction = "reduction" if delta < 0 else "increase"
        print(f"{k.upper():<14} {delta:+.6f}  ({abs(pct):.1f}% {direction})")
    else:
        pct = (comp_mean / l1_mean - 1) * 100 if l1_mean != 0 else 0.0
        direction = "gain" if delta > 0 else "drop"
        print(f"{k.upper():<14} {delta:+.4f}  ({abs(pct):.2f}% {direction})")

print("\n** Flux error is where the composite loss should shine — the flux-consistency")
print("   term directly penalises integrated-intensity mismatches during training. **")

# ── Ablation table figure ─────────────────────────────────────────────────────
ablation_results = {
    "Bicubic": bicubic_metrics,
    "L1-only EDSR+": results_l1_ensemble,
    "Composite EDSR+": results_ensemble,
}
plot_ablation_table(
    ablation_results,
    ["Bicubic", "L1-only EDSR+", "Composite EDSR+"],
    save_path=os.path.join("..", "figures", "ablation_6a.png"),
)

# %% [markdown]
# ## 17. Discussion

# %% [markdown]
# TODO
