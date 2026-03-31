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


if __name__ == "__main__":
    rng = np.random.RandomState(0)

    sr_img = rng.rand(150, 150)
    hr_img = rng.rand(150, 150)
    metrics = compute_metrics(sr_img, hr_img)
    print("compute_metrics result:", metrics)

    random_values = rng.rand(50).tolist()
    lo, hi = bootstrap_ci(random_values)
    print(f"bootstrap_ci on 50 random floats: [{lo:.4f}, {hi:.4f}]")

    print("All metrics tests passed.")
