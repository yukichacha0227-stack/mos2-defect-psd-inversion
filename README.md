# MoS2 PSD Defect Inversion

This repository is a cleaned, portfolio-ready implementation of a research
prototype for estimating atom-vacancy defect activation energies from
low-frequency noise power spectral density (PSD) data in few-layer MoS2.

The target variables are:

- `Ea_alpha`: activation energy of the shallower defect level
- `Ea_beta`: activation energy of the deeper defect level

The core idea is to treat PSD-based parameter estimation as a physics-guided
inverse problem. A 3000-point PSD sequence is split into three frequency bands,
each band is encoded by a 1D CNN with learnable GeM pooling, and the three
feature maps are recombined as an RGB-like tensor for a ResNet-style image
backbone.

## Why This Project Is Interesting

- Uses domain knowledge to impose an inductive bias before deep learning.
- Converts a 1D physical signal into a 3-channel representation that can reuse
  image-model building blocks.
- Predicts two physically meaningful quantities, `Ea_alpha` and `Ea_beta`, from
  noisy PSD curves.
- Includes a synthetic PSD generator so the repository can run without private
  lab data.

## Scientific Context

The architecture is based on the workflow described in:

Y. Nonaka, K. Takaki, Y. Kobayashi, and J. Haruyama,
"Machine learning for predicting physical parameters of atom-vacancy defects
from low-frequency noise in few-atom layer MoS2,"
AIP Advances 15, 045202 (2025). DOI: 10.1063/5.0254351

The paper models PSD spectra using two Lorentzian components and relates
frequency behavior to defect activation energies through SRH statistics. This
repository does not include the paper PDF, private experimental data, or trained
weights.

## Architecture

```text
PSD: (batch, 1, 3000)
        |
        | log scaling + Z-score normalization
        |
        +-- low band  [0:80]     -> 1D CNN -> (64, 64)
        +-- mid band  [30:300]   -> 1D CNN + GeM -> (64, 64)
        +-- high band [300:3000] -> 1D CNN + GeM -> (64, 64)
        |
        v
RGB-like tensor: (batch, 3, 64, 64)
        |
        v
SE-ResNet regressor
        |
        v
(Ea_alpha, Ea_beta)
```

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python scripts/train_demo.py --epochs 2 --samples 256 --batch-size 16
```

The demo trains on generated Lorentzian PSD curves and writes checkpoints and a
loss curve under `artifacts/demo/`.

## Using Your Own Data

Prepare tensors with these shapes:

```python
inputs.shape  == (num_samples, 1, 3000)
targets.shape == (num_samples, 2)  # Ea_alpha, Ea_beta
```

Then use `mos2_psd_inversion.data.make_dataloaders` and
`mos2_psd_inversion.train.fit`.

## What Was Cleaned From The Research Notebook

- Notebook cells were converted into importable modules.
- The two outputs were named explicitly as `Ea_alpha` and `Ea_beta`.
- Training, evaluation, checkpointing, and synthetic data generation were split.
- The model clamps PSD values before the logarithm to avoid invalid values after
  noisy augmentation.
- Private data files and model weights are excluded from version control.

## Resume-Friendly Summary

Built a PyTorch system for inverse estimation of MoS2 defect activation energies
from low-frequency PSD signals. Designed a physics-guided neural architecture
that segments the spectrum into low, mid, and high-frequency regions, extracts
band-specific features with 1D CNN and GeM pooling, recombines them as an
RGB-like tensor, and predicts shallow/deep defect levels with an SE-ResNet
regression head.
