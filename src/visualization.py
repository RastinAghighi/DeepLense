"""Plotting utilities for DeepLense super-resolution and classification experiments.

All functions accept an optional *save_path*; when provided the figure is saved
at 150 dpi before ``plt.show()`` is called.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_METHOD_COLORS = {
    "bicubic": "#888888",
    "edsr": "#1f77b4",
    "edsr+": "#2ca02c",
}


def _save_and_show(fig, save_path):
    """Tight-layout, optionally save, then show."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def _lognorm(img):
    """Return a LogNorm suitable for a lensing image, with linear fallback."""
    positive = img[img > 0]
    if positive.size == 0:
        return None  # fall back to linear
    vmin = max(positive.min(), 1e-4)
    vmax = img.max()
    if vmax <= vmin:
        return None
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def _color_for(name):
    return _METHOD_COLORS.get(name.lower(), None)


# ---------------------------------------------------------------------------
# 1. Sample pairs
# ---------------------------------------------------------------------------

def plot_sample_pairs(hr_images: list, lr_images: list, n: int = 5,
                      save_path=None):
    """Show HR / LR pairs in a 2xN grid with log normalization."""
    n = min(n, len(hr_images), len(lr_images))
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j in range(n):
        for row, (img, label) in enumerate(
                [(hr_images[j], "HR"), (lr_images[j], "LR")]):
            ax = axes[row, j]
            norm = _lognorm(img)
            ax.imshow(img, cmap="inferno", norm=norm)
            ax.set_title(f"{label} #{j}")
            ax.axis("off")

    axes[0, 0].set_ylabel("HR", fontsize=12)
    axes[1, 0].set_ylabel("LR", fontsize=12)
    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 2. Training curves (single stage)
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path=None):
    """Dual-panel training curves (loss left, quality-metrics right)."""
    fig, (ax_loss, ax_qual) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left panel: losses --
    _keys_styles = [
        ("train_loss", "Train loss", "-", "C0"),
        ("val_loss", "Val loss", "-", "C1"),
        ("train_l1", "Train L1", "--", "C2"),
        ("train_bp", "Train BP", "--", "C3"),
    ]
    for key, label, ls, color in _keys_styles:
        if key in history and history[key]:
            ax_loss.plot(history[key], label=label, linestyle=ls, color=color)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss")
    ax_loss.legend()

    # -- Right panel: PSNR & SSIM --
    if "val_psnr" in history and history["val_psnr"]:
        ax_qual.plot(history["val_psnr"], color="green", label="Val PSNR")
        ax_qual.set_ylabel("PSNR (dB)", color="green")
        ax_qual.tick_params(axis="y", labelcolor="green")

    ax_qual.set_xlabel("Epoch")
    ax_qual.set_title("Validation Quality Metrics")

    if "val_ssim" in history and history["val_ssim"]:
        ax2 = ax_qual.twinx()
        ax2.plot(history["val_ssim"], color="purple", label="Val SSIM")
        ax2.set_ylabel("SSIM", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

    # Combine legends
    lines, labels = ax_qual.get_legend_handles_labels()
    if "val_ssim" in history and history["val_ssim"]:
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    if lines:
        ax_qual.legend(lines, labels, loc="lower right")

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 3. Multi-stage training curves
# ---------------------------------------------------------------------------

def plot_training_curves_staged(histories: list, stage_names: list,
                                save_path=None):
    """Concatenated multi-stage training curves with vertical stage markers."""
    # Concatenate histories
    combined: dict = {}
    boundaries: list = []  # epoch indices where a new stage starts
    offset = 0

    for hist in histories:
        length = 0
        for key, vals in hist.items():
            if vals:
                combined.setdefault(key, []).extend(vals)
                length = max(length, len(vals))
        boundaries.append(offset)
        offset += length

    fig, (ax_loss, ax_qual) = plt.subplots(1, 2, figsize=(14, 5))

    # -- Left panel --
    _keys_styles = [
        ("train_loss", "Train loss", "-", "C0"),
        ("val_loss", "Val loss", "-", "C1"),
        ("train_l1", "Train L1", "--", "C2"),
        ("train_bp", "Train BP", "--", "C3"),
    ]
    for key, label, ls, color in _keys_styles:
        if key in combined and combined[key]:
            ax_loss.plot(combined[key], label=label, linestyle=ls, color=color)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training & Validation Loss (staged)")
    ax_loss.legend()

    # -- Right panel --
    ax2 = None
    if "val_psnr" in combined and combined["val_psnr"]:
        ax_qual.plot(combined["val_psnr"], color="green", label="Val PSNR")
        ax_qual.set_ylabel("PSNR (dB)", color="green")
        ax_qual.tick_params(axis="y", labelcolor="green")

    if "val_ssim" in combined and combined["val_ssim"]:
        ax2 = ax_qual.twinx()
        ax2.plot(combined["val_ssim"], color="purple", label="Val SSIM")
        ax2.set_ylabel("SSIM", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

    ax_qual.set_xlabel("Epoch")
    ax_qual.set_title("Validation Quality Metrics (staged)")

    lines, labels = ax_qual.get_legend_handles_labels()
    if ax2 is not None:
        l2, la2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += la2
    if lines:
        ax_qual.legend(lines, labels, loc="lower right")

    # Stage boundary lines
    for i, b in enumerate(boundaries):
        for ax in (ax_loss, ax_qual):
            ax.axvline(b, color="gray", linestyle=":", linewidth=1)
        label = stage_names[i] if i < len(stage_names) else f"stage {i}"
        ax_loss.text(b + 0.5, ax_loss.get_ylim()[1] * 0.95, label,
                     fontsize=8, color="gray", va="top")

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 4. Visual comparison grid
# ---------------------------------------------------------------------------

def plot_visual_comparison(samples: list, col_titles: list,
                           metric_col_idx: int = None, save_path=None):
    """N-row x M-column image comparison grid.

    Parameters
    ----------
    samples : list of tuples
        Each tuple contains M numpy arrays (H, W).
    col_titles : list of str
        Column headers.
    metric_col_idx : int, optional
        If set, annotate that column with PSNR/SSIM (requires last column
        to be ground truth).
    """
    n_rows = len(samples)
    n_cols = len(col_titles)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for i, sample in enumerate(samples):
        gt = sample[-1]  # last column = ground truth
        vmin, vmax = gt.min(), gt.max()
        for j, img in enumerate(sample):
            ax = axes[i, j]
            norm = _lognorm(img)
            ax.imshow(img, cmap="inferno", norm=norm)
            if i == 0:
                ax.set_title(col_titles[j], fontsize=10)
            ax.axis("off")

            if metric_col_idx is not None and j == metric_col_idx:
                from src.metrics import compute_metrics
                m = compute_metrics(img, gt)
                ax.text(0.02, 0.02,
                        f"PSNR {m['psnr']:.1f}\nSSIM {m['ssim']:.3f}",
                        transform=ax.transAxes, fontsize=7,
                        color="white", va="bottom",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 5. Error maps
# ---------------------------------------------------------------------------

def plot_error_maps(samples: list, method_names: list,
                    hr_images: list, save_path=None):
    """Absolute-error maps: |method_output - HR| per method.

    Parameters
    ----------
    samples : list of tuples
        samples[i] is a tuple of N_methods arrays for the i-th test image.
    method_names : list of str
    hr_images : list of np.ndarray
    """
    n_samples = len(samples)
    n_methods = len(method_names)
    fig, axes = plt.subplots(n_methods, n_samples,
                             figsize=(3 * n_samples, 3 * n_methods))
    if n_methods == 1:
        axes = axes[np.newaxis, :]
    if n_samples == 1:
        axes = axes[:, np.newaxis]

    for col in range(n_samples):
        hr = hr_images[col].astype(np.float64)
        errors = [np.abs(samples[col][m].astype(np.float64) - hr)
                  for m in range(n_methods)]
        vmax = max(e.max() for e in errors) or 1.0

        for row in range(n_methods):
            ax = axes[row, col]
            ax.imshow(errors[row], cmap="hot", vmin=0, vmax=vmax)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(method_names[row], fontsize=10)
            if row == 0:
                ax.set_title(f"Sample #{col}", fontsize=10)

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 6. Metric distributions (violin plots)
# ---------------------------------------------------------------------------

def plot_metric_distributions(metric_dicts: list, method_names: list,
                              save_path=None):
    """Violin plots for PSNR, SSIM, and Flux Error.

    Parameters
    ----------
    metric_dicts : list of dict
        One dict per method. Keys: psnr, ssim, flux_error (each a list).
    method_names : list of str
    """
    metrics = ["psnr", "ssim", "flux_error"]
    titles = ["PSNR (dB)", "SSIM", "Flux Error"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, key, title in zip(axes, metrics, titles):
        data = []
        colors = []
        for md, name in zip(metric_dicts, method_names):
            vals = md.get(key, [])
            data.append(vals if len(vals) else [0])
            colors.append(_color_for(name) or "C" + str(len(colors)))

        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for idx, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(colors[idx])
            pc.set_alpha(0.7)

        ax.set_xticks(range(1, len(method_names) + 1))
        ax.set_xticklabels(method_names)
        ax.set_title(title)
        ax.set_ylabel(title)

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 7. Failure analysis
# ---------------------------------------------------------------------------

def plot_failure_analysis(worst_samples: list, save_path=None):
    """5 worst-PSNR images: LR | SR | |SR-HR| error map.

    Parameters
    ----------
    worst_samples : list of tuples (lr, sr, hr, metrics_dict)
        Sorted worst-first.
    """
    n = min(len(worst_samples), 5)
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_labels = ["LR", "SR", "|SR - HR|"]

    for i in range(n):
        lr, sr, hr, m = worst_samples[i]
        error = np.abs(sr.astype(np.float64) - hr.astype(np.float64))
        imgs = [lr, sr, error]
        cmaps = ["inferno", "inferno", "hot"]

        for j, (img, cmap) in enumerate(zip(imgs, cmaps)):
            ax = axes[i, j]
            if cmap == "inferno":
                norm = _lognorm(img)
            else:
                norm = None
            ax.imshow(img, cmap=cmap, norm=norm)
            ax.axis("off")
            if i == 0:
                ax.set_title(col_labels[j], fontsize=11)

        psnr_val = m.get("psnr", float("nan"))
        axes[i, 0].set_ylabel(f"Rank {i + 1}\nPSNR {psnr_val:.2f}",
                               fontsize=9, rotation=0, labelpad=50, va="center")

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 8. Ablation table
# ---------------------------------------------------------------------------

def plot_ablation_table(results: dict, method_names: list, save_path=None):
    """Formatted comparison table rendered as a matplotlib figure.

    Parameters
    ----------
    results : dict
        results[method_name] = dict with keys mse, ssim, psnr, flux_error,
        each a list of per-sample values.
    method_names : list of str
    """
    metric_keys = ["mse", "ssim", "psnr", "flux_error"]
    col_labels = ["Method", "MSE", "SSIM", "PSNR (dB)", "Flux Error"]

    cell_text = []
    for name in method_names:
        row = [name]
        rd = results.get(name, {})
        for key in metric_keys:
            vals = np.array(rd.get(key, [0.0]))
            row.append(f"{vals.mean():.4f} \u00b1 {vals.std():.4f}")
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(12, 0.6 * len(method_names) + 1.5))
    ax.axis("off")

    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Bold header row
    for j in range(len(col_labels)):
        table[0, j].set_text_props(fontweight="bold")
        table[0, j].set_facecolor("#d9e2f3")

    ax.set_title("Method Comparison", fontsize=13, pad=20)
    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 9. ROC curves
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray,
                    class_names: list, save_path=None):
    """One-vs-Rest ROC curves + macro-average.

    Parameters
    ----------
    y_true : (N,) integer class labels
    y_probs : (N, C) predicted probabilities
    class_names : list of str, length C
    """
    from sklearn.metrics import roc_curve, auc

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 7))

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 200)

    for i in range(n_classes):
        binary = (np.asarray(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= n_classes
    macro_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, "k--", linewidth=2,
            label=f"Macro-avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], ":", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# 10. Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Normalized confusion matrix heatmap."""
    from sklearn.metrics import confusion_matrix as _cm

    cm = _cm(y_true, y_pred)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.0)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalized Confusion Matrix")

    # Annotate cells
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    color=color, fontsize=9)

    _save_and_show(fig, save_path)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _funcs = [
        plot_sample_pairs,
        plot_training_curves,
        plot_training_curves_staged,
        plot_visual_comparison,
        plot_error_maps,
        plot_metric_distributions,
        plot_failure_analysis,
        plot_ablation_table,
        plot_roc_curves,
        plot_confusion_matrix,
    ]
    for fn in _funcs:
        assert callable(fn), f"{fn.__name__} is not callable"
    print("visualization module loaded successfully")
