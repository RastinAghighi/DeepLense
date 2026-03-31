# DEEPLENSE GSoC 2026 — Claude Code Step-by-Step Execution Guide

> **How to use this file:** Complete each step in order. Each step has:
> - **Prompt**: Paste this into Claude Code (adjust paths if needed)
> - **Verify**: Run these commands to confirm the step worked
> - **Git**: Commands to commit and push after verification
>
> **Do NOT skip steps. Do NOT combine steps. Each step is sized to succeed.**
>
> Project root: `C:\Users\moham\OneDrive\Documents\Intro to Greatness\DeepLense`

---

## PHASE 0: PROJECT SETUP

### Step 0.1 — Initialize Git repository and folder structure

**Prompt:**
```
Create the following directory structure in the current project root (do NOT touch the Dataset/ or Dataset 3B/ folders — those contain raw data):

notebooks/
src/__init__.py
weights/
figures/

Also create a .gitignore file that ignores:
- __pycache__/
- *.pyc
- .ipynb_checkpoints/
- weights/*.pth
- Dataset/
- Dataset 3B/
- *.zip
- .env

And create a requirements.txt with:
torch
torchvision
numpy
matplotlib
scikit-image
scikit-learn
jupytext

Initialize a git repo if one doesn't exist. Make an initial commit with the folder structure.
```

**Verify:**
```powershell
ls src, notebooks, weights, figures
cat .gitignore
cat requirements.txt
git log --oneline
```

**Git:**
```powershell
git add -A
git commit -m "chore: initialize project structure"
git remote add origin https://github.com/RastinAghighi/DeepLense-GSoC-2026.git
git push -u origin main
```

---

### Step 0.2 — Create README.md

**Prompt:**
```
Create a README.md for this project. It should contain:

# DEEPLENSE — GSoC 2026 Evaluation Tests

Author: Rastin Aghighi
Organization: ML4SCI (Machine Learning for Science)
Project: Unsupervised Super-Resolution and Analysis of Real Lensing Images

## Overview
This repository contains evaluation test solutions for the DEEPLENSE Google Summer of Code 2026 project under ML4SCI. It includes three tasks:

1. **Task I** — Multi-class classification of gravitational lensing images (no substructure / subhalo / vortex)
2. **Task VI.A** — Supervised super-resolution on simulated strong lensing images using EDSR-baseline
3. **Task VI.B** — Transfer learning-based super-resolution on real HSC/HST telescope image pairs

## Architecture
- **SR Model:** EDSR-baseline (Lim et al., 2017) — 16 residual blocks, 64 features, no batch normalization
- **Loss:** Physics-informed composite loss (L1 + flux consistency + back-projection)
- **Transfer Learning:** 3-stage gradual unfreezing with L2-SP regularization for VI.B

## Project Structure
```
├── notebooks/          # Jupyter notebooks for each task
├── src/                # Shared utility modules
│   ├── edsr.py         # EDSR model definition
│   ├── losses.py       # Composite loss function
│   ├── dataset.py      # Dataset classes
│   ├── metrics.py      # Evaluation metrics
│   └── visualization.py# Plotting functions
├── weights/            # Trained model weights
├── figures/            # Saved plots
└── requirements.txt
```

## Setup
```bash
pip install -r requirements.txt
```

## Results
(To be filled after training)

Keep it clean and professional. This is what GSoC mentors will see first.
```

**Verify:**
```powershell
cat README.md
```

**Git:**
```powershell
git add README.md
git commit -m "docs: add project README"
git push
```

---

## PHASE 1: SHARED UTILITIES (src/)

### Step 1.1 — EDSR model definition (src/edsr.py)

**Prompt:**
```
Create src/edsr.py containing the EDSR-baseline model for super-resolution. Follow this specification EXACTLY:

class ResBlock(nn.Module):
    """Residual block without batch normalization."""
    - Constructor args: n_feats (int), res_scale (float, default=0.1)
    - Layers: conv1 = Conv2d(n_feats, n_feats, 3, padding=1), relu = ReLU(inplace=True), conv2 = Conv2d(n_feats, n_feats, 3, padding=1)
    - Forward: return x + self.res_scale * self.conv2(self.relu(self.conv1(x)))

class EDSR(nn.Module):
    """EDSR-baseline: 16 residual blocks, 64 filters, 2x upscale, no BN."""
    - Constructor args: n_channels=1, n_feats=64, n_resblocks=16, scale=2
    - self.head = Conv2d(n_channels, n_feats, 3, padding=1)
    - self.body = Sequential(*[ResBlock(n_feats) for _ in range(n_resblocks)], Conv2d(n_feats, n_feats, 3, padding=1))
    - self.tail = Sequential(Conv2d(n_feats, n_feats * scale**2, 3, padding=1), PixelShuffle(scale), Conv2d(n_feats, n_channels, 3, padding=1))
    - Forward:
        bicubic = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        head = self.head(x)
        body = self.body(head)
        res = head + body  # long skip connection
        sr = self.tail(res)
        return sr + bicubic  # global residual

Include:
- Docstrings explaining WHY no batch normalization (BN limits range flexibility for pixel regression)
- Docstrings explaining residual scaling (prevents gradient explosion)
- Docstrings explaining global residual learning (model learns high-frequency residual only)
- Import only torch, torch.nn, torch.nn.functional
- Type hints on constructor
- A __main__ block that creates the model with default args and prints the parameter count
```

**Verify:**
```powershell
py -c "from src.edsr import EDSR; import torch; m = EDSR(); print(f'Params: {sum(p.numel() for p in m.parameters()):,}'); x = torch.randn(1,1,75,75); y = m(x); print(f'Input: {x.shape} -> Output: {y.shape}')"
```
Expected output: ~1.37M params, input (1,1,75,75) → output (1,1,150,150)

**Git:**
```powershell
git add src/edsr.py
git commit -m "feat: add EDSR-baseline model definition"
git push
```

---

### Step 1.2 — Composite loss function (src/losses.py)

**Prompt:**
```
Create src/losses.py containing the physics-informed composite loss function. Specification:

class CompositeSRLoss(nn.Module):
    """Physics-informed composite loss: L1 + flux consistency + back-projection.

    L_total = L_L1(SR, HR) + lambda_flux * L_flux(SR, HR) + lambda_bp * L_bp(SR, LR)

    Components:
    - L1: pixel-wise MAE. Sharper than L2 which over-smooths.
    - Flux consistency: |sum(SR) - sum(HR)| / N_pixels. Preserves total
      integrated intensity — gravitational lensing conserves photon flux.
    - Back-projection: L1(downsample(SR), LR). Ensures SR is consistent
      with the LR observation when downsampled. Only requires the degradation
      model, not paired data — key insight for unsupervised SR.
    """

    Constructor args: lambda_flux=0.05, lambda_bp=0.1
    Store: self.l1 = nn.L1Loss(), self.lambda_flux, self.lambda_bp

    Forward(self, sr, hr, lr):
        l1_loss = self.l1(sr, hr)

        sr_flux = sr.sum(dim=(1, 2, 3))  # per-image flux
        hr_flux = hr.sum(dim=(1, 2, 3))
        n_pixels = sr.shape[-1] * sr.shape[-2]
        flux_loss = torch.mean(torch.abs(sr_flux - hr_flux)) / n_pixels

        sr_down = F.interpolate(sr, size=lr.shape[-2:], mode="bicubic", align_corners=False)
        bp_loss = self.l1(sr_down, lr)

        total = l1_loss + self.lambda_flux * flux_loss + self.lambda_bp * bp_loss

        return total, {"l1": l1_loss.item(), "flux": flux_loss.item(), "bp": bp_loss.item(), "total": total.item()}

Also add a class L2SPRegularizer:
    """L2-SP regularization (Li et al., 2018).
    Penalizes deviation from pretrained weights, not magnitude.
    Prevents catastrophic forgetting during fine-tuning.

    L_L2SP = alpha * sum((theta_i - theta_i_pretrained)^2)
    """

    Constructor: __init__(self, model, alpha=0.01)
        Store a deep copy of all parameters: self.pretrained = {name: param.clone().detach() for name, param in model.named_parameters()}
        Store self.alpha

    Method: penalty(self, model) -> torch.Tensor
        l2sp = sum of (param - self.pretrained[name]).pow(2).sum() for all param where param.requires_grad and name in self.pretrained
        return self.alpha * l2sp

Import only torch, torch.nn, torch.nn.functional, copy.
Include docstrings with the mathematical formulas.
Add a __main__ block that creates dummy tensors and tests both classes work.
```

**Verify:**
```powershell
py -c "from src.losses import CompositeSRLoss, L2SPRegularizer; import torch; loss_fn = CompositeSRLoss(); sr=torch.randn(2,1,150,150); hr=torch.randn(2,1,150,150); lr=torch.randn(2,1,75,75); total, parts = loss_fn(sr, hr, lr); print(f'Total: {total.item():.4f}, L1: {parts[\"l1\"]:.4f}, Flux: {parts[\"flux\"]:.4f}, BP: {parts[\"bp\"]:.4f}')"
```
Expected: prints loss values without error.

**Git:**
```powershell
git add src/losses.py
git commit -m "feat: add composite SR loss and L2-SP regularizer"
git push
```

---

### Step 1.3 — Dataset classes (src/dataset.py)

**Prompt:**
```
Create src/dataset.py with data loading utilities. Specification:

import os, glob, random, numpy as np, torch
from pathlib import Path
from torch.utils.data import Dataset

def load_sr_pairs(hr_dir: str, lr_dir: str, hr_prefix: str = "", lr_prefix: str = "") -> tuple:
    """Load HR/LR file path dicts and matched keys.

    For VI.A: files named sample{N}.npy in both HR/ and LR/ dirs.
    For VI.B: files named HR_{N}.npy and LR_{N}.npy.

    Args:
        hr_dir: path to HR directory
        lr_dir: path to LR directory
        hr_prefix: prefix to strip from HR filenames to get the matching key (e.g., "" for VI.A, "HR_" for VI.B)
        lr_prefix: prefix to strip from LR filenames to get the matching key (e.g., "" for VI.A, "LR_" for VI.B)

    Returns:
        (matched_keys, hr_files_dict, lr_files_dict)
        where keys are the common identifiers and dicts map key -> full filepath

    For VI.A call as: load_sr_pairs(hr_dir, lr_dir)
        → stems are identical (sample1, sample2, ...) so intersection works directly
    For VI.B call as: load_sr_pairs(hr_dir, lr_dir, hr_prefix="HR_", lr_prefix="LR_")
        → strips prefixes so HR_1 and LR_1 both become key "1"
    """
    hr_files = {}
    for f in glob.glob(os.path.join(hr_dir, "*.npy")):
        stem = Path(f).stem
        key = stem[len(hr_prefix):] if hr_prefix and stem.startswith(hr_prefix) else stem
        hr_files[key] = f

    lr_files = {}
    for f in glob.glob(os.path.join(lr_dir, "*.npy")):
        stem = Path(f).stem
        key = stem[len(lr_prefix):] if lr_prefix and stem.startswith(lr_prefix) else stem
        lr_files[key] = f

    matched_keys = sorted(set(hr_files.keys()) & set(lr_files.keys()))
    return matched_keys, hr_files, lr_files


def train_test_split(keys: list, train_ratio: float = 0.9, seed: int = 42) -> tuple:
    """Reproducible train/test split."""
    keys = keys.copy()
    rng = random.Random(seed)
    rng.shuffle(keys)
    split_idx = int(train_ratio * len(keys))
    return keys[:split_idx], keys[split_idx:]


class LensingSRDataset(Dataset):
    """Dataset for HR/LR lensing image pairs with consistent augmentation.

    Augmentation (applied identically to HR and LR):
    - Random 90-degree rotation (k in {0,1,2,3}): lensing has no preferred orientation
    - Random horizontal flip (p=0.5): no preferred handedness
    - Random vertical flip (p=0.5): same reasoning
    - Effective multiplier: up to 8x

    Data is NOT clipped — raw values (including negatives and >1.0) are preserved
    because they are physically meaningful (sky subtraction noise, bright peaks).
    """

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
        lr = np.load(self.lr_files[key]).astype(np.float32)  # (1, h, h)

        if self.augment:
            k = random.randint(0, 3)
            if k > 0:
                hr = np.rot90(hr, k, axes=(1, 2)).copy()
                lr = np.rot90(lr, k, axes=(1, 2)).copy()
            if random.random() > 0.5:
                hr = np.flip(hr, axis=2).copy()
                lr = np.flip(lr, axis=2).copy()
            if random.random() > 0.5:
                hr = np.flip(hr, axis=1).copy()
                lr = np.flip(lr, axis=1).copy()

        return torch.from_numpy(hr), torch.from_numpy(lr)

Include docstrings on everything. Add a __main__ block that:
1. Tests load_sr_pairs with VI.A paths (.\Dataset\Dataset\HR and .\Dataset\Dataset\LR) and prints the count
2. Tests load_sr_pairs with VI.B paths (.\Dataset 3B\Dataset\HR and .\Dataset 3B\Dataset\LR, hr_prefix="HR_", lr_prefix="LR_") and prints the count
3. Tests train_test_split on the VI.A keys and prints train/test counts
4. Creates a LensingSRDataset with augment=True, loads one sample, prints shapes
```

**Verify:**
```powershell
py -m src.dataset
```
Expected output: 4492 matched pairs (VI.A), 300 matched pairs (VI.B), correct train/test split counts, correct tensor shapes.

**Git:**
```powershell
git add src/dataset.py
git commit -m "feat: add dataset classes and data loading utilities"
git push
```

---

### Step 1.4 — Evaluation metrics (src/metrics.py)

**Prompt:**
```
Create src/metrics.py with the two-track evaluation protocol and self-ensemble inference. Specification:

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim
from skimage.metrics import peak_signal_noise_ratio as _psnr


def compute_metrics(sr: np.ndarray, hr: np.ndarray) -> dict:
    """Two-track evaluation: engineering metrics (clipped) + physics metrics (raw).

    Track 1 — Engineering (what the evaluation asks for):
        Clip both SR and HR to [0, 1].
        MSE = mean((sr_clip - hr_clip)^2)
        SSIM = structural_similarity(hr_clip, sr_clip, data_range=1.0)
        PSNR = peak_signal_noise_ratio(hr_clip, sr_clip, data_range=1.0)

    Track 2 — Physics:
        On raw unclipped values:
        Flux Error = |sum(SR) - sum(HR)| / |sum(HR)|

    Args:
        sr: numpy array (H, W), raw unclipped super-resolved image
        hr: numpy array (H, W), raw unclipped ground truth

    Returns:
        dict with keys: mse, ssim, psnr, flux_error
    """
    sr = sr.astype(np.float64)
    hr = hr.astype(np.float64)

    sr_clip = np.clip(sr, 0.0, 1.0)
    hr_clip = np.clip(hr, 0.0, 1.0)

    mse_val = np.mean((sr_clip - hr_clip) ** 2)
    ssim_val = _ssim(hr_clip, sr_clip, data_range=1.0)
    psnr_val = _psnr(hr_clip, sr_clip, data_range=1.0)

    hr_flux = hr.sum()
    sr_flux = sr.sum()
    flux_error = abs(sr_flux - hr_flux) / (abs(hr_flux) + 1e-10)

    return {"mse": mse_val, "ssim": ssim_val, "psnr": psnr_val, "flux_error": flux_error}


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval.

    Args:
        values: array-like of metric values
        n_bootstrap: number of bootstrap resamples
        ci: confidence level (0.95 = 95%)
        seed: random seed for reproducibility

    Returns:
        (lower, upper) bounds of the confidence interval
    """
    rng = np.random.RandomState(seed)
    values = np.array(values)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    boot_means.sort()
    lower = boot_means[int((1 - ci) / 2 * n_bootstrap)]
    upper = boot_means[int((1 + ci) / 2 * n_bootstrap)]
    return lower, upper


def format_metric_row(name, values):
    """Format a metric for printing: mean ± std, median, 95% CI."""
    lo, hi = bootstrap_ci(values)
    return (f"  {name:<14} {np.mean(values):.4f} +/- {np.std(values):.4f}  "
            f"median: {np.median(values):.4f}  95% CI: [{lo:.4f}, {hi:.4f}]")


@torch.no_grad()
def self_ensemble_predict(model, lr_tensor, device=None):
    """Geometric self-ensemble (EDSR+): run model on 8 augmented inputs, average outputs.

    Augmentations: 4 rotations (0, 90, 180, 270 degrees) x 2 (original + horizontal flip) = 8.
    Each SR output is de-augmented (reverse flip, reverse rotation) before averaging.

    This exploits the rotational symmetry of lensing images to suppress directional
    artifacts, typically adding +0.1 to +0.3 dB PSNR for free.

    Args:
        model: trained SR model in eval mode
        lr_tensor: (1, 1, H, W) tensor
        device: torch device (if None, uses lr_tensor's device)

    Returns:
        averaged SR tensor, same spatial size as model output
    """
    if device is None:
        device = lr_tensor.device
    model.eval()
    outputs = []

    for rot_k in range(4):
        for flip in [False, True]:
            x = torch.rot90(lr_tensor, rot_k, dims=(2, 3))
            if flip:
                x = torch.flip(x, dims=(3,))

            sr = model(x.to(device))

            if flip:
                sr = torch.flip(sr, dims=(3,))
            sr = torch.rot90(sr, -rot_k, dims=(2, 3))

            outputs.append(sr.cpu())

    return torch.stack(outputs).mean(dim=0)


Add a __main__ block that:
1. Creates two random 150x150 arrays and calls compute_metrics, prints the result
2. Calls bootstrap_ci on a list of 50 random floats, prints the CI
3. Prints "All metrics tests passed."
```

**Verify:**
```powershell
py -m src.metrics
```
Expected: prints metric dict, CI bounds, and "All metrics tests passed."

**Git:**
```powershell
git add src/metrics.py
git commit -m "feat: add two-track evaluation metrics and self-ensemble"
git push
```

---

### Step 1.5 — Visualization functions (src/visualization.py)

**Prompt:**
```
Create src/visualization.py with ALL plotting functions used across the three notebooks. Every function should:
- Accept a save_path parameter (default None). If provided, save to that path at dpi=150.
- Use consistent style: matplotlib, tight_layout, axis labels, titles.
- Colormap: "inferno" for lensing images, "hot" for error maps.
- Call plt.show() at the end.

Functions to implement (with full signatures and docstrings):

1. plot_sample_pairs(hr_images: list[np.ndarray], lr_images: list[np.ndarray], n: int = 5, save_path=None):
    """Show HR/LR pairs in a 2×N grid. Log normalization for lensing dynamic range."""
    - Top row: HR images, bottom row: LR images
    - Use matplotlib.colors.LogNorm(vmin=max(img[img>0].min(), 1e-4), vmax=img.max())
    - Handle edge case where all values <= 0 by falling back to linear normalization

2. plot_training_curves(history: dict, save_path=None):
    """Dual-panel training curves.
    Left: train_loss, val_loss, train_l1, train_bp (dashed) vs epoch.
    Right: dual y-axis — val_psnr (left, green) and val_ssim (right, purple)."""
    - history dict has keys: train_loss, val_loss, train_l1, train_bp, val_psnr, val_ssim
    - Some keys may be missing (e.g., if training didn't track all of them) — handle gracefully

3. plot_training_curves_staged(histories: list[dict], stage_names: list[str], save_path=None):
    """Training curves for multi-stage training (VI.B).
    Concatenates histories from multiple stages. Draws vertical lines at stage transitions.
    Same dual-panel layout as plot_training_curves."""

4. plot_visual_comparison(samples: list[tuple], col_titles: list[str], metric_col_idx: int = None, save_path=None):
    """N-row × M-column image comparison grid.
    Each sample is a tuple of M numpy arrays (H, W).
    Shared vmin/vmax per row (from last column, assumed to be ground truth).
    If metric_col_idx is set, annotate that column with PSNR/SSIM from compute_metrics."""

5. plot_error_maps(samples: list[tuple], method_names: list[str], hr_images: list[np.ndarray], save_path=None):
    """Error maps: |method_output - HR| for each method.
    N_methods rows × N_samples columns. "hot" colormap, shared vmax per column."""

6. plot_metric_distributions(metric_dicts: list[dict], method_names: list[str], save_path=None):
    """Violin plots for PSNR, SSIM, Flux Error.
    1×3 figure. Each violin colored differently per method.
    Colors: gray for bicubic, blue for EDSR, green for EDSR+."""

7. plot_failure_analysis(worst_samples: list[tuple], save_path=None):
    """5 worst PSNR images: LR / SR / |SR-HR| error map.
    Each tuple is (lr, sr, hr, metrics_dict).
    Row label shows rank and PSNR."""

8. plot_ablation_table(results: dict, method_names: list[str], save_path=None):
    """Formatted comparison table as a matplotlib figure.
    results[method_name] = dict with keys mse, ssim, psnr, flux_error (each a list).
    Shows mean ± std for each metric per method."""

9. plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, class_names: list[str], save_path=None):
    """ROC curves: one per class (OvR) + macro-average.
    Uses sklearn.metrics.roc_curve and auc.
    Includes diagonal reference line. AUC values in legend."""

10. plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Normalized confusion matrix heatmap using matplotlib imshow.
    Annotate each cell with the normalized value."""

Import: numpy, matplotlib.pyplot, matplotlib.colors (LogNorm), matplotlib.patches, sklearn.metrics (for ROC/confusion only — import inside the function to avoid hard dependency).

Do NOT use seaborn — keep dependencies minimal.

Add a __main__ block that prints "visualization module loaded successfully" and tests that all functions exist (just check they're callable, don't run them with data).
```

**Verify:**
```powershell
py -m src.visualization
```
Expected: "visualization module loaded successfully"

**Git:**
```powershell
git add src/visualization.py
git commit -m "feat: add full visualization suite"
git push
```

---

## PHASE 2: TASK VI.A — SIMULATED SUPER-RESOLUTION

### Step 2.1 — Create VI.A notebook skeleton

**Prompt:**
```
Create notebooks/Task_VIA_SuperResolution_Simulated.py as a Jupytext percent-format notebook (cells separated by # %% and # %% [markdown]).

This is ONLY the skeleton — markdown cells describing each section, imports, and configuration. NO training code yet. Include:

# %% [markdown]
# # Task VI.A: Supervised Super-Resolution on Simulated Strong Lensing Images
# Author: Rastin Aghighi
# DEEPLENSE – GSoC 2026 (ML4SCI)
# Project: Unsupervised Super-Resolution and Analysis of Real Lensing Images
#
# ## Strategy
# (Write a detailed 3-paragraph strategy section explaining:
#  1. EDSR-baseline architecture choice and why no BN
#  2. Composite loss function with physical motivation for each component
#  3. Self-ensemble inference and augmentation rationale)

Then section headers as markdown cells for:
1. Setup and Imports
2. Data Loading and Exploration
3. Dataset and DataLoader
4. Bicubic Interpolation Baseline
5. Model Definition
6. Composite Loss Function
7. Training (Composite Loss)
8. Training Curves
9. Self-Ensemble Inference (EDSR+)
10. Full Evaluation on Test Set
11. Metric Distribution Plots
12. Visual Comparison
13. Error Maps
14. Failure Analysis
15. Ablation Study: L1-only vs Composite Loss
16. Ablation Results Comparison
17. Discussion

For the "Setup and Imports" code cell, include:
- All imports (os, glob, random, numpy, matplotlib, torch, etc.)
- Import from src modules: from src.edsr import EDSR; from src.losses import CompositeSRLoss; from src.dataset import load_sr_pairs, train_test_split, LensingSRDataset; from src.metrics import compute_metrics, bootstrap_ci, format_metric_row, self_ensemble_predict; from src.visualization import (all functions)
- sys.path.insert so imports work from the notebooks/ directory
- Seed setting: random.seed(42), np.random.seed(42), torch.manual_seed(42), torch.cuda.manual_seed_all(42), torch.backends.cudnn.deterministic = True
- Device detection with GPU info printing
- BATCH_SIZE = 16, NUM_EPOCHS = 100, LEARNING_RATE = 1e-4, PATIENCE = 10

For the "Data Loading" code cell:
- DATA_ROOT = os.path.join("..", "Dataset", "Dataset")
- Call load_sr_pairs to get matched keys
- Print counts and shapes
- Call train_test_split
- Print train/test counts

For the "Dataset and DataLoader" code cell:
- Create train_dataset (augment=True), test_dataset (augment=False)
- Create train_loader (batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
- Create test_loader (batch_size=1, shuffle=False)

Leave all other code cells empty with just a "# TODO" comment — we will fill them in subsequent steps.
```

**Verify:**
```powershell
# Check the file exists and has the right structure
py -c "
with open('notebooks/Task_VIA_SuperResolution_Simulated.py') as f:
    content = f.read()
sections = [l for l in content.split('\n') if '# %%' in l]
print(f'Total cells: {len(sections)}')
for s in sections[:10]:
    print(s[:80])
"
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A notebook skeleton with imports and data loading"
git push
```

---

### Step 2.2 — VI.A: Data exploration and sample visualization

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in the "Data Loading and Exploration" section.

After the data loading cell (which already loads matched_keys, hr_files, lr_files), add a new code cell that:

1. Loads 200 sample images and computes dataset-wide statistics:
   - HR min/max range across all 200
   - LR min/max range across all 200
   - Print a note: "Values outside [0,1] are physically meaningful and will NOT be clipped during training."

2. A visualization cell that calls plot_sample_pairs:
   - Load 5 HR and 5 LR images (first 5 matched keys)
   - HR images: [np.load(hr_files[matched_keys[i]])[0] for i in range(5)]  # remove channel dim
   - LR images: [np.load(lr_files[matched_keys[i]])[0] for i in range(5)]
   - save_path=os.path.join("..", "figures", "sample_pairs_6a.png")
```

**Verify:**
```powershell
cd notebooks
py -c "exec(open('Task_VIA_SuperResolution_Simulated.py').read().split('# %% [markdown]')[0])" 2>&1 | Select-String "Matched|HR shape|LR shape"
cd ..
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A data exploration and sample visualization"
git push
```

---

### Step 2.3 — VI.A: Bicubic baseline evaluation

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in the "Bicubic Interpolation Baseline" section.

Code cell:
1. Import F from torch.nn.functional if not already imported
2. Iterate over test_loader:
   - For each (hr, lr) pair, compute bicubic upscale: F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
   - Call compute_metrics(sr_np, hr_np) where sr_np = bicubic[0,0].numpy(), hr_np = hr[0,0].numpy()
   - Append each metric to a dict: bicubic_metrics = {"mse": [], "ssim": [], "psnr": [], "flux_error": []}
3. Print results using format_metric_row for each metric
4. Store bicubic_metrics for later comparison

Add a markdown cell above explaining:
"Before training a learned model, we establish a performance floor using bicubic interpolation. This classical method uses a cubic polynomial kernel to upsample — it produces smooth results but cannot recover high-frequency details lost in the downsampling process."
```

**Verify:**
Run the notebook up to this cell and check that metrics print correctly (PSNR should be ~24-27 dB range).

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A bicubic baseline evaluation"
git push
```

---

### Step 2.4 — VI.A: Model instantiation and loss setup

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in the "Model Definition" and "Composite Loss Function" sections.

Model cell:
- model = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2).to(device)
- Print parameter count: total and trainable
- Add markdown explaining the architecture choice (reference the strategy section)

Loss cell:
- criterion = CompositeSRLoss(lambda_flux=0.05, lambda_bp=0.1)
- Print the loss formula and lambda values
- Add markdown explaining each component's physical motivation
```

**Verify:**
```powershell
py -c "from src.edsr import EDSR; m = EDSR(); print(sum(p.numel() for p in m.parameters()))"
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A model and loss setup"
git push
```

---

### Step 2.5 — VI.A: Training loop

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in the "Training" section. This is the main training loop.

Code cell:
- optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
- scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, verbose=True)
- history = {"train_loss": [], "train_l1": [], "train_flux": [], "train_bp": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
- best_val_loss = float("inf"), patience_counter = 0, best_model_state = None

Training loop (for epoch in range(NUM_EPOCHS)):
  Training phase:
    - model.train()
    - For each (hr, lr) in train_loader:
      - Move to device
      - optimizer.zero_grad()
      - sr = model(lr)
      - loss, loss_parts = criterion(sr, hr, lr)
      - loss.backward()
      - optimizer.step()
      - Accumulate loss components weighted by batch size
    - Average and append to history

  Validation phase:
    - model.eval(), torch.no_grad()
    - For each (hr, lr) in test_loader:
      - sr = model(lr)
      - loss, _ = criterion(sr, hr, lr)
      - Compute PSNR and SSIM on clipped [0,1] outputs using skimage directly (not compute_metrics, to keep it fast)
    - Average and append to history

  Logging:
    - Print every 5 epochs or when best model improves
    - Format: "Ep {epoch}/100 | Loss: {total} (L1:{l1} Flux:{flux} BP:{bp}) | Val PSNR: {psnr} SSIM: {ssim} | LR: {lr}"
    - Mark best epoch with " *"

  Early stopping:
    - If val_loss < best_val_loss: save state dict (cpu clone), reset counter
    - Else: increment counter
    - If counter >= PATIENCE: break, print message

  After training:
    - Restore best model: model.load_state_dict(best_model_state); model.to(device)
    - Save weights: torch.save(best_model_state, os.path.join("..", "weights", "edsr_simulated_best.pth"))
    - Print confirmation

Add markdown cell above explaining: optimizer choice (Adam), loss function, scheduler strategy, early stopping rationale.
```

**Verify:**
Run this cell — it will take time. Check that:
- Loss decreases over epochs
- PSNR increases
- Model weights file is saved
```powershell
ls ..\weights\edsr_simulated_best.pth
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A training loop with composite loss"
git push
```

---

### Step 2.6 — VI.A: Training curve visualization

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in the "Training Curves" section.

Code cell:
- Call plot_training_curves(history, save_path=os.path.join("..", "figures", "training_curves_6a.png"))
```

**Verify:**
Check that `figures/training_curves_6a.png` exists and looks correct.

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py figures/training_curves_6a.png
git commit -m "feat: VI.A training curve visualization"
git push
```

---

### Step 2.7 — VI.A: Full evaluation with self-ensemble

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in "Self-Ensemble Inference" and "Full Evaluation on Test Set" sections.

Self-ensemble markdown cell:
"We run the model on 8 augmented versions of each LR input (4 rotations × 2 flip states) and average the de-augmented outputs. This geometric self-ensemble suppresses directional artifacts and typically adds +0.1 to +0.3 dB PSNR without retraining."

Evaluation code cell:
- model.eval()
- Initialize: results_standard = {"mse":[], "ssim":[], "psnr":[], "flux_error":[]}, results_ensemble = same, test_samples = []
- For each (hr, lr) in test_loader:
    - Standard inference: sr_std = model(lr.to(device)).cpu()
    - Ensemble inference: sr_ens = self_ensemble_predict(model, lr, device)
    - Convert to numpy: hr_np = hr[0,0].numpy(), lr_np = lr[0,0].numpy(), etc.
    - Compute metrics for both: compute_metrics(sr_std_np, hr_np), compute_metrics(sr_ens_np, hr_np)
    - Append to results dicts
    - Store first 10 samples: test_samples.append((lr_np, sr_std_np, sr_ens_np, hr_np))
- Print comprehensive results table:
    For each metric: print Bicubic / EDSR / EDSR+ with mean±std, median, 95% CI
    Print improvement summary: PSNR gain, SSIM gain, MSE reduction %
```

**Verify:**
Run this cell. Check:
- EDSR+ metrics are slightly better than EDSR
- Both are significantly better than bicubic
- No NaN or inf values

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A self-ensemble evaluation"
git push
```

---

### Step 2.8 — VI.A: All visualizations (distributions, comparisons, error maps, failure analysis)

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in sections 11-14 (Metric Distributions, Visual Comparison, Error Maps, Failure Analysis).

Section 11 — Metric Distributions:
- Call plot_metric_distributions(
    [bicubic_metrics, results_standard, results_ensemble],
    ["Bicubic", "EDSR", "EDSR+"],
    save_path=os.path.join("..", "figures", "metric_distributions_6a.png"))

Section 12 — Visual Comparison:
- Build samples list: [(lr, sr_std, sr_ens, hr) for first 5 test_samples]
- Call plot_visual_comparison(
    samples=[(s[0], s[1], s[2], s[3]) for s in test_samples[:5]],
    col_titles=["LR Input (75×75)", "EDSR Output", "EDSR+ (Ensemble)", "HR Ground Truth (150×150)"],
    save_path=os.path.join("..", "figures", "visual_comparison_6a.png"))

Section 13 — Error Maps:
- For each of first 5 test_samples:
    - Compute bicubic upsample of LR
    - Compute |bicubic - HR|, |EDSR - HR|, |EDSR+ - HR|
- Call plot_error_maps(
    samples=[(bic_err, std_err, ens_err) for ...],
    method_names=["Bicubic", "EDSR", "EDSR+"],
    hr_images=[s[3] for s in test_samples[:5]],
    save_path=os.path.join("..", "figures", "error_maps_6a.png"))

Section 14 — Failure Analysis:
- Find 5 worst PSNR indices from results_ensemble["psnr"]
- For each, re-run self_ensemble_predict on the corresponding test_loader item
- Collect: (lr_np, sr_np, hr_np, metrics_dict) for each
- Call plot_failure_analysis(worst_samples, save_path=os.path.join("..", "figures", "failure_analysis_6a.png"))
- Print analysis notes:
  "Worst PSNR images tend to have bright, compact central sources where small spatial errors produce large intensity mismatches."
  "EDSR's L1 loss treats all errors equally — perceptual or adversarial losses could preserve sharp features better."
```

**Verify:**
```powershell
ls ..\figures\*6a*
```
Should show: metric_distributions_6a.png, visual_comparison_6a.png, error_maps_6a.png, failure_analysis_6a.png

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py figures/
git commit -m "feat: VI.A visualization suite (distributions, comparisons, error maps, failures)"
git push
```

---

### Step 2.9 — VI.A: Ablation study (L1-only model)

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in section 15 (Ablation Study).

Markdown cell explaining:
"To validate the composite loss, we train an identical EDSR model using only L1 loss (no flux consistency, no back-projection). Same architecture, same seed (42), same train/test split, same augmentation, same optimizer, same scheduler. The only difference is the loss function."

Code cell:
1. Create a new EDSR model: model_l1 = EDSR(...).to(device)
2. criterion_l1 = nn.L1Loss()
3. optimizer_l1 = Adam(model_l1.parameters(), lr=LEARNING_RATE)
4. scheduler_l1 = ReduceLROnPlateau(...)
5. Run the same training loop as the primary model but with L1 loss only
   - Track: history_l1 = {"train_loss": [], "val_loss": [], "val_psnr": [], "val_ssim": [], "lr": []}
   - Same early stopping logic
   - Print progress every 5 epochs
6. Restore best model, save to weights/edsr_simulated_l1only.pth
7. Evaluate on test set with self_ensemble_predict
   - Store results_l1_ensemble = {"mse":[], "ssim":[], "psnr":[], "flux_error":[]}

Section 16 — Ablation Results Comparison:
- Print comparison table: Bicubic / L1-only EDSR+ / Composite EDSR+
- For each metric: mean ± std, with improvement deltas
- Highlight flux_error specifically — this is where composite should shine
- Call plot_ablation_table with results from all three methods
  save_path=os.path.join("..", "figures", "ablation_6a.png")
```

**Verify:**
```powershell
ls ..\weights\edsr_simulated_l1only.pth
ls ..\figures\ablation_6a.png
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py weights/ figures/
git commit -m "feat: VI.A ablation study (L1-only vs composite loss)"
git push
```

---

### Step 2.10 — VI.A: Discussion section

**Prompt:**
```
In notebooks/Task_VIA_SuperResolution_Simulated.py, fill in section 17 (Discussion) as a LONG markdown cell.

Write a thoughtful, detailed discussion covering these exact subsections:

### Where the Model Succeeds
- Smooth extended arc structures (dominant lensing morphology)
- Global residual connection ensures stable baseline
- Self-ensemble exploits rotational symmetry

### Where the Model Fails
- Bright compact central galaxies (small spatial errors → large intensity mismatch)
- Faint extended features near the noise floor
- Reference the failure analysis figures

### Why It Fails
- L1 loss predicts conditional median, not full posterior → regression-to-the-mean
- No mechanism to prioritize astrophysically significant features over noise
- Limited receptive field of 3×3 convolutions

### Physics-Informed Loss
- Flux consistency enforces photon conservation (reference ablation results)
- Back-projection constrains solution to be observation-consistent
- Reference the ablation table showing flux error improvement

### Self-Ensemble (EDSR+)
- Exploits rotational/reflective symmetry of lensing images
- Free quality boost — quantify the PSNR gain from the results

### What Would Improve Results
- Perceptual loss (VGG features)
- Adversarial training (ESRGAN) for sharper textures
- Diffusion models (DiffLense by the mentors) for posterior sampling
- Attention mechanisms (RCAN, SwinIR)
- Physics-informed regularization beyond flux (PSF modeling, power spectrum matching)

### Connection to GSoC Proposal
- This supervised model is the performance ceiling for the proposed unsupervised approach
- Back-projection loss only requires degradation model, not paired data
- Directly applicable to unsupervised/self-supervised SR
- The gap between supervised performance and unsupervised (proposed) will quantify the cost of removing paired data

Make it sound like a thoughtful researcher, not a template. Reference specific numbers from the evaluation results where possible (use placeholder values like [PSNR_VALUE] that can be filled in after training).
```

**Verify:**
Read the discussion section — it should be substantive, specific, and demonstrate domain understanding.

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.py
git commit -m "feat: VI.A discussion section"
git push
```

---

### Step 2.11 — Convert VI.A to .ipynb

**Prompt:**
```
Convert notebooks/Task_VIA_SuperResolution_Simulated.py to .ipynb format using jupytext:

py -m jupytext --to notebook notebooks/Task_VIA_SuperResolution_Simulated.py

Verify the .ipynb file was created. Keep both the .py and .ipynb files.
```

**Verify:**
```powershell
ls notebooks\*.ipynb
```

**Git:**
```powershell
git add notebooks/Task_VIA_SuperResolution_Simulated.ipynb
git commit -m "feat: VI.A notebook converted to ipynb"
git push
```

---

## PHASE 3: TASK VI.B — REAL TELESCOPE SUPER-RESOLUTION

### Step 3.1 — Create VI.B notebook skeleton

**Prompt:**
```
Create notebooks/Task_VIB_SuperResolution_Real.py as a Jupytext percent-format notebook.

Same overall structure as VI.A but with these differences:

Strategy markdown: explain transfer learning from VI.A, 3-stage gradual unfreezing, L2-SP regularization, and the domain gap challenge (simulated vs real telescope data).

Section headers:
1. Setup and Imports (same as VI.A plus import L2SPRegularizer)
2. Data Loading and Exploration (300 real pairs)
3. Domain Gap Visualization (show simulated vs real side by side)
4. Dataset and DataLoader
5. Bicubic Interpolation Baseline
6. Load Pretrained EDSR from VI.A
7. 3-Stage Gradual Unfreezing Setup
8. Stage 1 Training (Tail Only)
9. Stage 2 Training (Blocks 10-15)
10. Stage 3 Training (Blocks 6-9)
11. Combined Training Curves
12. Self-Ensemble Evaluation
13. Full Evaluation (Two-Track)
14. Metric Distributions
15. Visual Comparison
16. Error Maps
17. Failure Analysis
18. Ablation: Fine-tuned vs From-Scratch
19. Ablation Results Comparison
20. Discussion

Data loading:
- DATA_ROOT = os.path.join("..", "Dataset 3B", "Dataset")
- load_sr_pairs(hr_dir, lr_dir, hr_prefix="HR_", lr_prefix="LR_")
- Should find 300 matched pairs
- train_test_split → 270 train / 30 test

Fill in sections 1-5 (through bicubic baseline) with actual code.
Leave sections 6-20 as TODO.
```

**Verify:**
```powershell
py -c "
with open('notebooks/Task_VIB_SuperResolution_Real.py') as f:
    content = f.read()
print(f'Lines: {len(content.splitlines())}')
print('300' if '300' in content else 'WARNING: 300 not found')
print('HR_' if 'HR_' in content else 'WARNING: HR_ prefix not found')
"
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B notebook skeleton with data loading and bicubic baseline"
git push
```

---

### Step 3.2 — VI.B: Domain gap visualization

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 3 (Domain Gap Visualization).

Create a figure showing simulated (VI.A) and real (VI.B) images side by side to visually illustrate the domain gap.

Code:
1. Load 3 HR images from VI.A (Dataset/Dataset/HR/): simulated, clean, no noise
2. Load 3 HR images from VI.B (Dataset 3B/Dataset/HR/): real HST, noisy, artifacts
3. Load 3 LR images from VI.A and VI.B similarly
4. Create a 2×6 figure (or 4×3): top row simulated HR, bottom row real HR, etc.
   Or better: a 2×3 grid where each column is one pair and rows are "Simulated" vs "Real"
5. Use log normalization, inferno colormap
6. Title: "Domain Gap: Simulated vs Real Telescope Data"
7. Save to figures/domain_gap_comparison.png

Add a markdown cell explaining:
"The simulated data (VI.A) has clean noise characteristics and controlled PSFs. Real telescope data (VI.B) from HSC/HST has complex noise patterns, varying PSFs, background sources, and potential cosmic ray artifacts. This domain gap is the core challenge for transfer learning."
```

**Verify:**
```powershell
ls ..\figures\domain_gap_comparison.png
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py figures/
git commit -m "feat: VI.B domain gap visualization"
git push
```

---

### Step 3.3 — VI.B: Load pretrained weights and setup unfreezing

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in sections 6 and 7.

Section 6 — Load Pretrained EDSR:
- model = EDSR(n_channels=1, n_feats=64, n_resblocks=16, scale=2)
- pretrained_path = os.path.join("..", "weights", "edsr_simulated_best.pth")
- state_dict = torch.load(pretrained_path, map_location="cpu")
- model.load_state_dict(state_dict)
- model = model.to(device)
- Print: "Loaded pretrained weights from VI.A (simulated data)"
- Print parameter count

Section 7 — Unfreezing Setup:
Create a helper function set_trainable(model, stage) that freezes/unfreezes parameters:

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

Also create a helper function get_param_groups(model, stage) that returns optimizer parameter groups with differential learning rates:

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

Test: call set_trainable(model, 1) and print the result.
```

**Verify:**
Run the cell — should print parameter counts for stage 1 (only tail trainable).

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B pretrained weights loading and unfreezing setup"
git push
```

---

### Step 3.4 — VI.B: Stage 1 training (tail only)

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 8 (Stage 1 Training).

Markdown explaining: "Stage 1 adapts only the upsampling module (tail) to the real data's pixel scale and noise characteristics, while keeping all feature extractors frozen. This is the safest first step — we change as little as possible."

Code:
- set_trainable(model, stage=1)
- criterion = CompositeSRLoss(lambda_flux=0.05, lambda_bp=0.1)
- param_groups = get_param_groups(model, stage=1)
- optimizer = Adam(param_groups, weight_decay=1e-4)  # Standard L2 for stage 1
- scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
- MAX_EPOCHS_S1 = 20, PATIENCE_S1 = 5
- Run training loop (same structure as VI.A but shorter)
- Track history_s1 = {"train_loss":[], "val_loss":[], "val_psnr":[], "val_ssim":[], "lr":[]}
- Early stopping on val_loss
- After training, save best state dict as stage1_state
- Print best val PSNR/SSIM achieved in stage 1
```

**Verify:**
Check that training runs and PSNR improves over epochs.

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B stage 1 training (tail only)"
git push
```

---

### Step 3.5 — VI.B: Stage 2 training (deep blocks)

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 9 (Stage 2 Training).

Markdown: "Stage 2 unfreezes the deeper residual blocks (10-15) with differential learning rates. We introduce L2-SP regularization to prevent catastrophic forgetting of useful features learned on simulated data."

Code:
- set_trainable(model, stage=2)
- l2sp = L2SPRegularizer(model, alpha=0.01)  # Create BEFORE changing requires_grad
  Actually — create L2SP regularizer right after loading pretrained weights (section 6), storing the pretrained state. Then use it here.
  Move the L2SP creation: l2sp_reg = L2SPRegularizer(model, alpha=0.01) right after model.load_state_dict() in section 6.
- param_groups = get_param_groups(model, stage=2)
- optimizer = Adam(param_groups)  # No weight_decay, using L2-SP instead
- scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
- MAX_EPOCHS_S2 = 40, PATIENCE_S2 = 8
- Training loop:
    loss, loss_parts = criterion(sr, hr, lr)
    l2sp_penalty = l2sp_reg.penalty(model)
    total_loss = loss + l2sp_penalty
    total_loss.backward()
- Track history_s2
- Early stopping
- Print best val PSNR/SSIM
```

**Verify:**
Check that L2-SP penalty is being computed (print it during first epoch).

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B stage 2 training (deep blocks + L2-SP)"
git push
```

---

### Step 3.6 — VI.B: Stage 3 training (mid blocks)

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 10 (Stage 3 Training).

Markdown: "Stage 3 unfreezes mid-level blocks (6-9) with the lowest learning rate. Blocks 0-5 remain permanently frozen — low-level feature detectors (edges, textures) generalize well across domains. We switch to cosine annealing for smooth convergence."

Code:
- set_trainable(model, stage=3)
- param_groups = get_param_groups(model, stage=3)
- optimizer = Adam(param_groups)
- scheduler = CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-7)
- MAX_EPOCHS_S3 = 40, PATIENCE_S3 = 10
- Same training loop with L2-SP regularization
- Track history_s3
- Early stopping
- Save final best model: torch.save(best_state, os.path.join("..", "weights", "edsr_real_finetuned.pth"))
```

**Verify:**
```powershell
ls ..\weights\edsr_real_finetuned.pth
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B stage 3 training (mid blocks + cosine annealing)"
git push
```

---

### Step 3.7 — VI.B: Combined training curves

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 11 (Combined Training Curves).

Call plot_training_curves_staged(
    histories=[history_s1, history_s2, history_s3],
    stage_names=["Stage 1 (Tail)", "Stage 2 (Blocks 10-15)", "Stage 3 (Blocks 6-9)"],
    save_path=os.path.join("..", "figures", "training_curves_6b.png"))
```

**Verify:**
```powershell
ls ..\figures\training_curves_6b.png
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py figures/
git commit -m "feat: VI.B combined training curves"
git push
```

---

### Step 3.8 — VI.B: Full evaluation and all visualizations

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in sections 12-17 (evaluation, distributions, comparison, error maps, failure analysis).

This is the same pattern as VI.A steps 2.7 and 2.8 but adapted for VI.B:
- Use test_loader from VI.B (30 test images)
- Compare: bicubic / EDSR (standard) / EDSR+ (ensemble)
- All the same visualization calls with "_6b" suffix on save paths
- All saved to figures/ directory

Print the full evaluation table with two-track metrics.
Print improvement over bicubic.
```

**Verify:**
```powershell
ls ..\figures\*6b*
```
Should show all VI.B figures.

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py figures/
git commit -m "feat: VI.B full evaluation and visualization suite"
git push
```

---

### Step 3.9 — VI.B: Ablation (fine-tuned vs from-scratch)

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in sections 18-19 (ablation).

Markdown: "To validate that transfer learning helps, we train an identical EDSR from random initialization on the same 270 real training images. Same composite loss, same augmentation, same seed."

Code:
1. model_scratch = EDSR(...).to(device)  # Fresh random weights
2. criterion_scratch = CompositeSRLoss(lambda_flux=0.05, lambda_bp=0.1)
3. optimizer_scratch = Adam(model_scratch.parameters(), lr=1e-4)
4. scheduler_scratch = ReduceLROnPlateau(...)
5. Train for up to 100 epochs with early stopping (patience=10)
6. Save to weights/edsr_real_scratch.pth
7. Evaluate with self-ensemble
8. Print comparison table: Bicubic / From-scratch EDSR+ / Fine-tuned EDSR+
9. Call plot_ablation_table, save to figures/ablation_6b.png
```

**Verify:**
```powershell
ls ..\weights\edsr_real_scratch.pth
ls ..\figures\ablation_6b.png
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py weights/ figures/
git commit -m "feat: VI.B ablation study (fine-tuned vs from-scratch)"
git push
```

---

### Step 3.10 — VI.B: Discussion section

**Prompt:**
```
In notebooks/Task_VIB_SuperResolution_Real.py, fill in section 20 (Discussion) as a detailed markdown cell.

Cover these subsections:

### Domain Gap
- Simulated data is clean; real HSC/HST has complex noise, varying PSFs, background sources, cosmic rays
- Different pixel scales between HSC and HST
- Transfer learning bridges this gap by preserving useful feature extractors

### Small Dataset Challenges
- 270 training images is extremely limited
- Overfitting is the primary risk
- Mitigated by: 3-stage unfreezing, L2-SP regularization, 8× augmentation, early stopping per stage

### Transfer Learning Analysis
- Reference ablation results (fine-tuned vs from-scratch PSNR difference)
- Low-level features (edges, textures in blocks 0-5) transfer well — that's why they're permanently frozen
- High-level features (noise-specific patterns in blocks 10-15) needed adaptation

### Why Gradual Unfreezing
- Prevents catastrophic forgetting
- Differential learning rates respect feature hierarchy
- L2-SP anchors parameters near pretrained values

### Connection to GSoC Project
- Real telescope data is the actual target
- Domain gap challenge motivates unsupervised approaches
- Even with paired supervision, reconstruction is imperfect → underscores difficulty
- The back-projection loss's independence from paired data is the bridge to unsupervised SR

Reference specific numbers from the results.
```

**Verify:**
Read the discussion — should be substantive and reference actual results.

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.py
git commit -m "feat: VI.B discussion section"
git push
```

---

### Step 3.11 — Convert VI.B to .ipynb

**Prompt:**
```
Convert notebooks/Task_VIB_SuperResolution_Real.py to .ipynb format:
py -m jupytext --to notebook notebooks/Task_VIB_SuperResolution_Real.py
```

**Verify:**
```powershell
ls notebooks\Task_VIB*.ipynb
```

**Git:**
```powershell
git add notebooks/Task_VIB_SuperResolution_Real.ipynb
git commit -m "feat: VI.B notebook converted to ipynb"
git push
```

---

## PHASE 4: TASK I — CLASSIFICATION

> **Note:** Complete this phase after downloading the classification dataset.

### Step 4.1 — Download and inspect classification dataset

**Prompt:**
```
I've downloaded the classification dataset to [PATH]. Inspect it:
1. Show directory structure
2. Count files per class subfolder
3. Load one sample from each class — print shape, dtype, value range
4. Print class names and sample counts
```

**Verify:**
Paste the output — confirm 3 classes, image dimensions, file counts.

**Git:**
No commit needed for data inspection.

---

### Step 4.2 — Create Task I notebook with full implementation

**Prompt:**
```
Create notebooks/Task_I_Classification.py as a complete Jupytext notebook for multi-class gravitational lensing classification.

Dataset is at: [FILL IN PATH AFTER DOWNLOAD]
Classes: no substructure (0), subhalo (1), vortex (2)
Image shape: [FILL IN AFTER INSPECTION]

Sections:
1. Setup and Imports
2. Data Loading and Exploration (class distribution bar chart, 3×5 sample grid)
3. Dataset and DataLoader (with augmentation: continuous rotation, h-flip, v-flip)
4. Model (ResNet-18 pretrained, repeat grayscale 3x for input, fc → 3 classes)
5. Training (Adam lr=1e-4, CrossEntropyLoss, ReduceLROnPlateau, early stopping pat=7, 50 epochs max)
6. Training Curves (loss + accuracy, train and val)
7. Confusion Matrix (normalized heatmap)
8. ROC Curves and AUC (per-class OvR + macro-average)
9. Sample Predictions (correct and incorrect examples)
10. Discussion

For the ROC computation:
- After training, run model on test set collecting softmax probabilities and true labels
- Use sklearn.metrics.roc_curve with one-vs-rest for each class
- Use sklearn.metrics.auc for area under each curve
- Plot all curves on one figure with AUC in legend

For confusion matrix:
- sklearn.metrics.confusion_matrix
- Normalize by true class (row normalization)
- Plot as heatmap with annotated cells

Save all figures to figures/ directory.
Save model to weights/resnet18_classifier.pth.

Discussion:
- Architecture choice (ResNet-18, established for lensing)
- Which class is hardest to classify?
- Which class pairs are most confused?
- Augmentation rationale (rotational symmetry)
- Connection to GSoC (SR improves classification accuracy)
```

**Verify:**
Run the notebook top to bottom. Check:
- AUC values are all > 0.95
- ROC curves plot correctly
- Confusion matrix looks reasonable

**Git:**
```powershell
git add notebooks/Task_I_Classification.py
py -m jupytext --to notebook notebooks/Task_I_Classification.py
git add notebooks/Task_I_Classification.ipynb weights/resnet18_classifier.pth figures/*task1*
git commit -m "feat: Task I classification notebook (complete)"
git push
```

---

## PHASE 5: PACKAGING AND SUBMISSION

### Step 5.1 — Update README with results

**Prompt:**
```
Update README.md with actual results from all three tasks. Add a "Results" section with:

1. Task VI.A results table (Bicubic / EDSR / EDSR+ with PSNR, SSIM, MSE, Flux Error)
2. Task VI.A ablation table (L1-only vs composite)
3. Task VI.B results table
4. Task VI.B ablation table (fine-tuned vs from-scratch)
5. Task I AUC scores (per class + macro)

Use the actual numbers from the notebook outputs. Format as markdown tables.
```

**Verify:**
```powershell
cat README.md
```

**Git:**
```powershell
git add README.md
git commit -m "docs: add final results to README"
git push
```

---

### Step 5.2 — Final quality check

**Prompt:**
```
Run these checks across all notebooks and src files:

1. Verify no hardcoded absolute paths (search for "C:\Users" or "moham" in all .py files)
2. Verify all imports resolve (run "py -c 'from src.edsr import EDSR; from src.losses import CompositeSRLoss; from src.dataset import LensingSRDataset; from src.metrics import compute_metrics'" from project root)
3. Verify all weight files exist in weights/
4. Verify all figure files exist in figures/
5. Verify all notebooks have a Discussion section at the end
6. List any TODO comments remaining in any file
7. Check .gitignore covers all data files

Print a summary of all checks.
```

**Verify:**
All checks should pass with no issues.

**Git:**
```powershell
git add -A
git commit -m "chore: final quality check passed"
git push
```

---

### Step 5.3 — Final push and submission

**Manual steps (not Claude Code):**

1. Go to your GitHub repo — verify all files are visible
2. Download the 3 .ipynb files from notebooks/
3. Download the .pth files from weights/ (or provide download links in the form)
4. Fill out the Google Form with:
   - Notebook files
   - GitHub repo link: https://github.com/RastinAghighi/DeepLense-GSoC-2026
   - Model weight files
5. Submit before April 1, 2026, 18:00 UTC

---

## QUICK REFERENCE: EXECUTION DEPENDENCY GRAPH

```
Step 0.1 → 0.2 → 1.1 → 1.2 → 1.3 → 1.4 → 1.5
                                                  ↓
                                    2.1 → 2.2 → 2.3 → 2.4 → 2.5 → 2.6 → 2.7 → 2.8 → 2.9 → 2.10 → 2.11
                                                                                                         ↓
                                    3.1 → 3.2 → 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9 → 3.10 → 3.11
                                    
                                    4.1 → 4.2  (can start after Phase 1, independent of VI.A/VI.B)
                                    
                                    5.1 → 5.2 → 5.3  (after all tasks complete)
```

**Total steps: 28**
**Estimated time with GPU: 10-15 hours (most is training time)**
