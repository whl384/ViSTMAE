# ViST-MAE: Decoupled Spatio-Temporal Masked Autoencoder with Vision Prior

Official PyTorch implementation of ViST-MAE for multivariate time-series anomaly detection.

## ⚙️ Environment Setup

```bash
pip install -r requirements.txt
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
