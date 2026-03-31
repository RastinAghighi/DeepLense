# DEEPLENSE — GSoC 2026 Evaluation Tests

**Author:** Rastin Aghighi
**Organization:** ML4SCI (Machine Learning for Science)
**Project:** Unsupervised Super-Resolution and Analysis of Real Lensing Images

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
├── notebooks/           # Jupyter notebooks for each task
├── src/                 # Shared utility modules
│   ├── edsr.py          # EDSR model definition
│   ├── losses.py        # Composite loss function
│   ├── dataset.py       # Dataset classes
│   ├── metrics.py       # Evaluation metrics
│   └── visualization.py # Plotting functions
├── weights/             # Trained model weights
├── figures/             # Saved plots
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## Results

*(To be filled after training)*
