"""Data loading utilities for gravitational lensing super-resolution datasets."""

import os
import glob
import random

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def load_sr_pairs(hr_dir: str, lr_dir: str, hr_prefix: str = "", lr_prefix: str = "") -> tuple:
    """Load HR/LR file path dicts and matched keys.

    For VI.A: files named sample{N}.npy in both HR/ and LR/ dirs.
    For VI.B: files named HR_{N}.npy and LR_{N}.npy.

    Args:
        hr_dir: path to HR directory
        lr_dir: path to LR directory
        hr_prefix: prefix to strip from HR filenames to get the matching key
            (e.g., "" for VI.A, "HR_" for VI.B)
        lr_prefix: prefix to strip from LR filenames to get the matching key
            (e.g., "" for VI.A, "LR_" for VI.B)

    Returns:
        (matched_keys, hr_files_dict, lr_files_dict)
        where keys are the common identifiers and dicts map key -> full filepath

    For VI.A call as: load_sr_pairs(hr_dir, lr_dir)
        -> stems are identical (sample1, sample2, ...) so intersection works directly
    For VI.B call as: load_sr_pairs(hr_dir, lr_dir, hr_prefix="HR_", lr_prefix="LR_")
        -> strips prefixes so HR_1 and LR_1 both become key "1"
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
    """Reproducible train/test split.

    Args:
        keys: list of sample keys to split.
        train_ratio: fraction of keys to assign to the training set.
        seed: random seed for reproducibility.

    Returns:
        (train_keys, test_keys) tuple of lists.
    """
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

    Data is NOT clipped -- raw values (including negatives and >1.0) are preserved
    because they are physically meaningful (sky subtraction noise, bright peaks).
    """

    def __init__(self, keys, hr_files, lr_files, augment=False):
        """Initialise the dataset.

        Args:
            keys: list of matched sample keys.
            hr_files: dict mapping key -> HR .npy filepath.
            lr_files: dict mapping key -> LR .npy filepath.
            augment: whether to apply random augmentation.
        """
        self.keys = keys
        self.hr_files = hr_files
        self.lr_files = lr_files
        self.augment = augment

    def __len__(self):
        """Return the number of samples."""
        return len(self.keys)

    def __getitem__(self, idx):
        """Load and return an (HR, LR) tensor pair.

        Args:
            idx: sample index.

        Returns:
            (hr, lr) tuple of float32 tensors with shape (1, H, H) and (1, h, h).
        """
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


if __name__ == "__main__":
    # 1. Test load_sr_pairs with VI.A paths
    via_hr = os.path.join(".", "Dataset", "Dataset", "HR")
    via_lr = os.path.join(".", "Dataset", "Dataset", "LR")
    via_keys, via_hr_files, via_lr_files = load_sr_pairs(via_hr, via_lr)
    print(f"VI.A matched pairs: {len(via_keys)}")

    # 2. Test load_sr_pairs with VI.B paths
    vib_hr = os.path.join(".", "Dataset 3B", "Dataset", "HR")
    vib_lr = os.path.join(".", "Dataset 3B", "Dataset", "LR")
    vib_keys, vib_hr_files, vib_lr_files = load_sr_pairs(
        vib_hr, vib_lr, hr_prefix="HR_", lr_prefix="LR_"
    )
    print(f"VI.B matched pairs: {len(vib_keys)}")

    # 3. Test train_test_split on VI.A keys
    train_keys, test_keys = train_test_split(via_keys)
    print(f"VI.A train: {len(train_keys)}, test: {len(test_keys)}")

    # 4. Create a LensingSRDataset with augment=True, load one sample, print shapes
    ds = LensingSRDataset(via_keys, via_hr_files, via_lr_files, augment=True)
    hr_sample, lr_sample = ds[0]
    print(f"HR shape: {hr_sample.shape}, LR shape: {lr_sample.shape}")
