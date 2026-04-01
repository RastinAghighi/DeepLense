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
# # Task VI.B: Supervised Super-Resolution on Real Strong Lensing Images
# **Author:** Rastin Aghighi
# **DEEPLENSE – GSoC 2026 (ML4SCI)**
# **Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images
#
# ## Strategy
#
# This notebook transfers the EDSR model trained on simulated data (Task VI.A) to
# real telescope observations via **transfer learning with gradual unfreezing**.
# Real lensing images differ from simulations in several important ways: they
# contain instrumental noise patterns, PSF variations, background subtraction
# artifacts, and complex foreground/background contamination that simulators
# do not model. This **domain gap** means that a model trained purely on
# simulated pairs will under-perform on real data — the high-frequency textures
# it learned to recover are specific to the simulation pipeline.
#
# To bridge this gap efficiently with only 300 real pairs, we adopt a
# **3-stage gradual unfreezing** schedule inspired by ULMFiT (Howard & Ruder,
# 2018). Rather than fine-tuning all parameters at once (which risks
# catastrophic forgetting of useful low-level features) or training only a
# linear head (which lacks capacity to adapt), we progressively unfreeze
# deeper layers:
#
# - **Stage 1 – Tail only:** Unfreeze only the pixel-shuffle upsampler and
#   final convolution. These layers are the most task-specific and need the
#   most adaptation to the new degradation model.
# - **Stage 2 – Blocks 10–15:** Unfreeze the later residual blocks, which
#   encode mid-level texture priors. These adapt to the real-data noise floor
#   and PSF while retaining general SR features.
# - **Stage 3 – Blocks 6–9:** Unfreeze the middle residual blocks. Earlier
#   blocks (0–5) are kept frozen because they capture generic edge and
#   gradient features that transfer well across domains.
#
# Each stage uses a reduced learning rate (1e-4 → 5e-5 → 2e-5) and
# **L2-SP regularization** (Li et al., 2018). Unlike standard L2 weight
# decay, which penalises deviation from zero, L2-SP penalises deviation from
# the *pretrained* weights:
#
# $$\mathcal{L}_{\text{L2-SP}} = \alpha \sum_i (\theta_i - \theta_i^{\text{pretrained}})^2$$
#
# This explicitly anchors newly unfrozen parameters near their pretrained
# values, preventing catastrophic forgetting while still allowing sufficient
# adaptation. The combination of gradual unfreezing + L2-SP + the composite
# SR loss from VI.A gives us a principled transfer learning pipeline that
# maximises data efficiency on the small real dataset.

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Allow imports from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("__file__"), "..")))

from src.edsr import EDSR
from src.losses import CompositeSRLoss, L2SPRegularizer
from src.dataset import load_sr_pairs, train_test_split, LensingSRDataset
from src.metrics import compute_metrics, bootstrap_ci, format_metric_row, self_ensemble_predict
from src.visualization import (
    plot_sample_pairs,
    plot_training_curves,
    plot_training_curves_staged,
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
PATIENCE = 10

# Stage-specific learning rates and epochs
STAGE_CONFIGS = {
    1: {"lr": 1e-4, "epochs": 60, "name": "Tail only"},
    2: {"lr": 5e-5, "epochs": 40, "name": "Blocks 10-15"},
    3: {"lr": 2e-5, "epochs": 30, "name": "Blocks 6-9"},
}
L2SP_ALPHA = 0.01

# %% [markdown]
# ## 2. Data Loading and Exploration

# %%
DATA_ROOT = os.path.join("..", "Dataset 3B", "Dataset")

hr_dir = os.path.join(DATA_ROOT, "HR")
lr_dir = os.path.join(DATA_ROOT, "LR")

keys, hr_dict, lr_dict = load_sr_pairs(hr_dir, lr_dir, hr_prefix="HR_", lr_prefix="LR_")
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
    save_path=os.path.join("..", "figures", "sample_pairs_6b.png"),
)

# %% [markdown]
# ## 3. Domain Gap Visualization
#
# The simulated data (VI.A) has clean noise characteristics and controlled PSFs.
# Real telescope data (VI.B) from HSC/HST has complex noise patterns, varying
# PSFs, background sources, and potential cosmic ray artifacts. This domain gap
# is the core challenge for transfer learning.

# %%
# ── Load simulated (VI.A) data for comparison ────────────────────────────────
SIM_DATA_ROOT = os.path.join("..", "Dataset", "Dataset")
sim_hr_dir = os.path.join(SIM_DATA_ROOT, "HR")
sim_lr_dir = os.path.join(SIM_DATA_ROOT, "LR")

sim_keys, sim_hr_dict, sim_lr_dict = load_sr_pairs(sim_hr_dir, sim_lr_dir)
print(f"Simulated pairs available: {len(sim_keys)}")

# %%
# ── Domain gap figure: HR and LR side by side ────────────────────────────────
n_compare = 3
fig, axes = plt.subplots(4, n_compare, figsize=(4 * n_compare, 14))

row_labels = ["Simulated HR", "Real HR", "Simulated LR", "Real LR"]

for i in range(n_compare):
    # Row 0: Simulated HR
    sim_hr = np.load(sim_hr_dict[sim_keys[i]])[0]
    axes[0, i].imshow(np.log1p(np.clip(sim_hr, 0, None)), cmap="inferno")
    axes[0, i].set_title(f"Sample {i+1}", fontsize=11)
    axes[0, i].axis("off")

    # Row 1: Real HR
    real_hr = np.load(hr_dict[keys[i]])[0]
    axes[1, i].imshow(np.log1p(np.clip(real_hr, 0, None)), cmap="inferno")
    axes[1, i].axis("off")

    # Row 2: Simulated LR
    sim_lr = np.load(sim_lr_dict[sim_keys[i]])[0]
    axes[2, i].imshow(np.log1p(np.clip(sim_lr, 0, None)), cmap="inferno")
    axes[2, i].axis("off")

    # Row 3: Real LR
    real_lr = np.load(lr_dict[keys[i]])[0]
    axes[3, i].imshow(np.log1p(np.clip(real_lr, 0, None)), cmap="inferno")
    axes[3, i].axis("off")

for row, label in enumerate(row_labels):
    axes[row, 0].set_ylabel(label, fontsize=12, rotation=90, labelpad=10)

fig.suptitle("Domain Gap: Simulated vs Real Telescope Data", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join("..", "figures", "domain_gap_comparison.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %%
# ── Intensity distribution comparison ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Gather pixel distributions (subsample for speed)
sim_pixels = np.concatenate([
    np.load(sim_hr_dict[sim_keys[i]]).ravel()
    for i in range(min(50, len(sim_keys)))
])
real_pixels = np.concatenate([
    np.load(hr_dict[keys[i]]).ravel()
    for i in range(min(50, len(keys)))
])

ax1.hist(sim_pixels, bins=200, alpha=0.7, label="Simulated", color="#1f77b4", density=True)
ax1.hist(real_pixels, bins=200, alpha=0.7, label="Real", color="#d62728", density=True)
ax1.set_xlabel("Pixel intensity")
ax1.set_ylabel("Density")
ax1.set_title("Intensity Distribution (linear)")
ax1.legend()
ax1.set_xlim(-0.1, 1.0)

ax2.hist(np.log1p(np.clip(sim_pixels, 0, None)), bins=200, alpha=0.7,
         label="Simulated", color="#1f77b4", density=True)
ax2.hist(np.log1p(np.clip(real_pixels, 0, None)), bins=200, alpha=0.7,
         label="Real", color="#d62728", density=True)
ax2.set_xlabel("log(1 + intensity)")
ax2.set_ylabel("Density")
ax2.set_title("Intensity Distribution (log scale)")
ax2.legend()

fig.suptitle("Domain Gap: Pixel Intensity Distributions", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join("..", "figures", "domain_gap_distributions_6b.png"),
            dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Dataset and DataLoader

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
# ## 5. Bicubic Interpolation Baseline
#
# Before any learned method, we establish the performance floor using bicubic
# interpolation on the real data. This also lets us compare baseline quality
# between simulated and real domains.

# %%
bicubic_metrics = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}

with torch.no_grad():
    for hr, lr in tqdm(test_loader, desc="Bicubic baseline"):
        sr = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
        sr_np = sr[0, 0].numpy()
        hr_np = hr[0, 0].numpy()
        m = compute_metrics(sr_np, hr_np)
        for k in bicubic_metrics:
            bicubic_metrics[k].append(m[k])

print("Bicubic Interpolation Baseline (Real Data)")
print("-" * 50)
for k in bicubic_metrics:
    print(format_metric_row(k, bicubic_metrics[k]))

# %% [markdown]
# ## 6. Load Pretrained EDSR from VI.A

# %%
model = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2)

pretrained_path = os.path.join("..", "weights", "edsr_simulated_best.pth")
state_dict = torch.load(pretrained_path, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(device)

print("Loaded pretrained weights from VI.A (simulated data)")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# %% [markdown]
# ## 7. 3-Stage Gradual Unfreezing Setup

# %%
def set_trainable(model, stage):
    """Configure which parameters are trainable for each stage.

    Stage 1: Only tail (upsampling + final conv)
    Stage 2: Tail + body blocks 10-15 + body final conv
    Stage 3: Tail + body blocks 6-15 + body final conv
    Permanently frozen: head + body blocks 0-5
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze based on stage
    if stage >= 1:
        for param in model.tail.parameters():
            param.requires_grad = True

    if stage >= 2:
        # model.body is Sequential: indices 0-15 are ResBlocks, index 16 is final conv
        for i in range(10, 17):  # blocks 10-15 + final conv (index 16)
            for param in model.body[i].parameters():
                param.requires_grad = True

    if stage >= 3:
        for i in range(6, 10):  # blocks 6-9
            for param in model.body[i].parameters():
                param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Stage {stage}: {trainable:,} / {total:,} parameters trainable ({100*trainable/total:.1f}%)")


def get_param_groups(model, stage):
    """Return parameter groups with differential learning rates per stage."""
    groups = []
    if stage == 1:
        groups.append({"params": [p for p in model.tail.parameters() if p.requires_grad], "lr": 1e-4})
    elif stage == 2:
        groups.append({"params": [p for p in model.tail.parameters() if p.requires_grad], "lr": 5e-5})
        groups.append({"params": [p for n, p in model.body[13:16].named_parameters() if p.requires_grad], "lr": 1e-5, "name": "blocks_13-15"})
        groups.append({"params": [p for n, p in model.body[10:13].named_parameters() if p.requires_grad], "lr": 5e-6, "name": "blocks_10-12"})
        groups.append({"params": [p for p in model.body[16].parameters() if p.requires_grad], "lr": 1e-5, "name": "body_final_conv"})
    elif stage == 3:
        groups.append({"params": [p for p in model.tail.parameters() if p.requires_grad], "lr": 1e-5})
        groups.append({"params": [p for n, p in model.body[13:16].named_parameters() if p.requires_grad], "lr": 5e-6})
        groups.append({"params": [p for n, p in model.body[10:13].named_parameters() if p.requires_grad], "lr": 5e-6})
        groups.append({"params": [p for n, p in model.body[6:10].named_parameters() if p.requires_grad], "lr": 1e-6})
        groups.append({"params": [p for p in model.body[16].parameters() if p.requires_grad], "lr": 5e-6})
    # Filter out empty groups
    return [g for g in groups if len(list(g["params"])) > 0]


# Test: apply Stage 1 unfreezing
set_trainable(model, 1)

# %% [markdown]
# ## 8. Stage 1 Training (Tail Only)

# %%
# TODO: Freeze all layers except the tail (upsampler + final conv).
# Train with lr=1e-4 for up to 60 epochs with early stopping.
# Log training history for later concatenation.

# %% [markdown]
# ## 9. Stage 2 Training (Blocks 10-15)

# %%
# TODO: Additionally unfreeze residual blocks 10-15.
# Train with lr=5e-5 for up to 40 epochs with early stopping.
# L2-SP regularization anchors newly unfrozen blocks to pretrained values.

# %% [markdown]
# ## 10. Stage 3 Training (Blocks 6-9)

# %%
# TODO: Additionally unfreeze residual blocks 6-9.
# Train with lr=2e-5 for up to 30 epochs with early stopping.
# Blocks 0-5 remain frozen throughout (generic low-level features).

# %% [markdown]
# ## 11. Combined Training Curves

# %%
# TODO: Use plot_training_curves_staged() to visualize all three stages
# with vertical boundary markers. Save to ../figures/training_curves_6b.png

# %% [markdown]
# ## 12. Self-Ensemble Evaluation

# %%
# TODO: Apply self_ensemble_predict (8 geometric transforms) to fine-tuned model.
# Compare EDSR vs EDSR+ metrics on real test set.

# %% [markdown]
# ## 13. Full Evaluation (Two-Track)

# %%
# TODO: Evaluate fine-tuned EDSR and EDSR+ on all test samples.
# Track 1 — Engineering: MSE, SSIM, PSNR (clipped [0,1])
# Track 2 — Physics: Flux Error (raw values)
# Report means with bootstrap 95% CIs.

# %% [markdown]
# ## 14. Metric Distributions

# %%
# TODO: Violin plots comparing Bicubic vs EDSR vs EDSR+ on real data.
# Save to ../figures/metric_distributions_6b.png

# %% [markdown]
# ## 15. Visual Comparison

# %%
# TODO: Side-by-side grid: LR | Bicubic | EDSR | EDSR+ | HR
# Select ~5 representative test samples.
# Save to ../figures/visual_comparison_6b.png

# %% [markdown]
# ## 16. Error Maps

# %%
# TODO: Absolute error maps |method - HR| for Bicubic, EDSR, EDSR+.
# Save to ../figures/error_maps_6b.png

# %% [markdown]
# ## 17. Failure Analysis

# %%
# TODO: Identify the 5 worst-PSNR test samples for the fine-tuned EDSR.
# Display LR, SR, HR, and error map for each.
# Save to ../figures/failure_analysis_6b.png

# %% [markdown]
# ## 18. Ablation: Fine-tuned vs From-Scratch

# %%
# TODO: Train an EDSR from scratch on the 270 real training samples
# (same composite loss, same total epoch budget) to demonstrate that
# transfer learning outperforms random initialization on limited data.
# Save model to ../weights/edsr_real_scratch.pth

# %% [markdown]
# ## 19. Ablation Results Comparison

# %%
# TODO: Tabulate and compare:
#   - Bicubic baseline
#   - EDSR from scratch
#   - EDSR fine-tuned (gradual unfreezing + L2-SP)
#   - EDSR+ fine-tuned (self-ensemble)
# Use plot_ablation_table(). Save to ../figures/ablation_6b.png

# %% [markdown]
# ## 20. Discussion
#
# TODO: Summarize key findings:
# - Domain gap quantification (pretrained-on-sim vs fine-tuned performance)
# - Benefit of gradual unfreezing + L2-SP vs from-scratch training
# - Self-ensemble improvement on real data
# - Remaining failure modes and their physical interpretation
# - Implications for the unsupervised approach (Task VII)
