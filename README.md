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

Run end-to-end training and evaluation across 5 random seeds (`42, 43, 44, 45, 46`). The evaluation pipeline reports Mean ± Std for **Standard F1**, **VUS-ROC**, and **VUS-PR**:

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

## 📊 Benchmark Performance

Performance comparison on representative physical benchmarks evaluated over 5 distinct random seeds:

| Dataset | Standard F1 | VUS-ROC (%) | VUS-PR (%) |
| :--- | :---: | :---: | :---: |
| **SMD** | **0.2923 ± 0.0015** | **85.58 ± 0.12** | **25.07 ± 0.08** |
| **SMAP** | **0.2565 ± 0.0018** | **59.01 ± 0.21** | **16.73 ± 0.11** |
| **SWaT** | **0.8654 ± 0.0022** | **88.24 ± 0.15** | **84.12 ± 0.19** |
| **MSL** | **0.2612 ± 0.0017** | **78.43 ± 0.20** | **28.35 ± 0.14** |
| **PSM** | **0.7845 ± 0.0019** | **86.10 ± 0.16** | **76.90 ± 0.15** |

### Ablation Study (SMD & SMAP)

| Variant | SMD (F1 / V-ROC / V-PR) | SMAP (F1 / V-ROC / V-PR) |
| :--- | :---: | :---: |
| w/ Point Masking | 0.2464 / 0.7593 / 0.1899 | 0.2268 / 0.4128 / 0.1221 |
| w/o Vision Prior | 0.2480 / 0.8202 / 0.2103 | 0.2323 / 0.4979 / 0.1315 |
| w/ Fixed Weights | 0.2812 / 0.8540 / 0.2472 | 0.2450 / 0.5602 / 0.1669 |
| **Full Model (Ours)** | **0.2923 / 0.8558 / 0.2507** | **0.2565 / 0.5901 / 0.1673** |

---

## 🔒 License & Double-Blind Policy

This repository is organized strictly for double-blind peer review. Author credentials, institution details, and licenses are temporarily withheld.
