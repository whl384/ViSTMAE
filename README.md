# ViST-MAE: Decoupled Spatio-Temporal Masked Autoencoder with Vision Prior

Official PyTorch implementation of **"ViST-MAE: Decoupled Spatio-Temporal Masked Autoencoders with Vision Prior for Multivariate Time-Series Anomaly Detection"**.

---

## 🌟 Overview

**ViST-MAE** is a novel multivariate time-series anomaly detection framework designed to overcome the over-smoothing and interpolation shortcuts inherent in traditional autoencoders. Key components include:

- **Temporal Stream (`TimeMAE`)**: Suspicion-guided patch masking driven by discrete first- and second-order derivatives to suppress trivial temporal interpolation shortcuts.
- **Spatial Stream (`SpatialMAE`)**: Channel-wise topological consensus masking to capture inter-sensor physical correlations.
- **Vision Prior Flow (`DINOv2`)**: Transforms 1D series into high-dimensional 2D geometric manifolds via bilinear interpolation and channel replication, processed through a frozen `dinov2_vits14` extractor.
- **Minimax Adversarial Optimization**: A zero-sum game aligned via symmetric Kullback-Leibler Divergence (KLD) to penalize over-smoothed temporal reconstructions.
- **Continuous Profiler**: Closed-form autonomous convex routing $(\gamma, \beta, \alpha)$ derived from uncorrupted normal sequences without requiring anomaly labels.

---

## 🛠️ Environment Setup

```bash
# Clone the repository
git clone <ANONYMOUS_REPO_URL>
cd ViSTMAE

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.9
- PyTorch >= 2.0.0
- torchvision
- timm >= 0.9.0
- numpy >= 1.22.0, < 2.0.0
- scikit-learn
- scipy
- pandas

---

## 📂 Data Preparation

Due to file size limits and standard benchmark licensing, datasets are not bundled directly within this repository. Please obtain the public benchmarks from their official open repositories and organize them inside the `dataset/` directory following this layout:

```text
ViSTMAE/
├── dataset/
│   ├── SMD/
│   │   ├── SMD_train.npy
│   │   ├── SMD_test.npy
│   │   └── SMD_test_label.npy
│   ├── SMAP/
│   │   ├── SMAP_train.npy
│   │   ├── SMAP_test.npy
│   │   └── SMAP_test_label.npy
│   ├── MSL/
│   │   ├── MSL_train.npy
│   │   ├── MSL_test.npy
│   │   └── MSL_test_label.npy
│   ├── SWaT/
│   │   ├── SWaT_train.npy
│   │   ├── SWaT_test.npy
│   │   └── SWaT_test_label.npy
│   └── PSM/
│       ├── PSM_train.npy
│       ├── PSM_test.npy
│       └── PSM_test_label.npy
```

> **Note on Data Sources**:
> - **SMD** (Server Machine Dataset): OmniAnomaly Benchmark Repository
> - **SMAP / MSL**: NASA Telemetry Data Repository
> - **SWaT**: iTrust Center for Research in Cyber Security
> - **PSM**: eBay Public Data Repository

---

## 🚀 Reproducibility & Benchmark Scripts

Run end-to-end training and evaluation across 5 random seeds (`42, 43, 44, 45, 46`). The evaluation pipeline outputs Mean ± Std for **Standard F1**, **VUS-ROC**, and **VUS-PR**:

```bash
# 1. Server Machine Dataset (SMD)
python run.py --dataset SMD --device cuda:0 --seeds 42 43 44 45 46

# 2. Soil Moisture Active Passive satellite (SMAP)
python run.py --dataset SMAP --device cuda:0 --seeds 42 43 44 45 46

# 3. Secure Water Treatment testbed (SWaT)
python run.py --dataset SWaT --device cuda:0 --seeds 42 43 44 45 46

# 4. Mars Science Laboratory rover (MSL)
python run.py --dataset MSL --device cuda:0 --seeds 42 43 44 45 46

# 5. Pooled Server Metrics (PSM)
python run.py --dataset PSM --device cuda:0 --seeds 42 43 44 45 46
```

---

## 🔒 License & Double-Blind Policy

This repository is organized strictly for double-blind peer review. Author credentials, institution details, and licenses are temporarily withheld.
