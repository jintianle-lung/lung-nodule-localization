# Intraoperative Lung Nodule Localization System

> **An intraoperative lung nodule localization system based on a flexible array tactile sensor and deep learning fusion.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Paper: TBME](https://img.shields.io/badge/Paper-TBME%20(Draft)-green.svg)](docs/paper/)

## Overview

This repository contains the complete codebase, model definitions, training pipelines, and experimental results for a **surgical lung nodule localization system** that combines a **flexible array tactile sensor** with **deep learning fusion**.

The system is designed to assist thoracic surgeons during minimally invasive lung resection by:
1. **Detecting** whether a pulmonary nodule is present under the sensor
2. **Estimating** the nodule size
3. **Predicting** the nodule depth (exploratory)

## Key Results

| Stage | Task | Metric | Value |
|-------|------|--------|-------|
| Stage 1 | Nodule Detection | F1 Score | 0.6647 |
| Stage 1 | Nodule Detection | AUC | 0.8311 |
| Stage 2 | Size Estimation | Top-2 Accuracy | 0.8215 |
| Stage 2 | Size Estimation | MAE | 0.1338 cm |
| Stage 3 | Depth Estimation (auxiliary) | Balanced Accuracy | 0.6251 |

## Repository Structure

```
github_reviewer_release/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── requirements.txt                   # Python dependencies
│
├── data/                              # Dataset documentation and samples
│   ├── README.md                      # Data format and usage guide
│   └── sample_data/                   # Example sensor data files
│
├── models/                            # Model definitions
│   ├── __init__.py
│   ├── dual_stream_mstcn_detection.py   # Stage 1: Detection model
│   ├── raw_positive_size_model_v2.py    # Stage 2: Size estimation model
│   ├── hierarchical_shared_window_mtl.py # Stage 3: Depth estimation model
│   ├── input_normalization_v1.py        # Input normalization utilities
│   └── task_protocol_v1.py              # Task protocol definitions
│
├── training/                          # Training scripts
│   ├── train_stage1_detection.py        # Stage 1 training
│   ├── train_stage1_dualstream_mstcn.py # Stage 1 dual-stream training
│   ├── train_stage2_raw_positive_size_v2.py  # Stage 2 training
│   └── train_stage3_hierarchical_shared_depth.py # Stage 3 training
│
├── evaluation/                        # Evaluation and analysis scripts
│   ├── evaluate_file3_active_model.py   # File 3 evaluation
│   ├── compare_ablation_models.py       # Ablation study comparison
│   └── calculate_detection_rate.py      # Detection rate statistics
│
├── visualization/                     # Figure generation scripts
│   ├── generate_stage1_lite_detection_cam_attention.py
│   ├── generate_stage2_lite_cam_attention.py
│   ├── generate_xgboost_latest_interpretability_20260404.py
│   └── generate_posneg_bestdet_compact_20260407.py
│
├── experiments/                       # Experimental results
│   ├── mainline_release_lock_20260405_final.json  # Locked mainline results
│   ├── outputs_stage1_dualstream_mstcn_detection_raw_only_lite_20260402_run1/
│   ├── outputs_stage2_raw_positive_size_v2_lite_nodelta_noimplicit_nophase_20260402_run1/
│   ├── outputs_stage3_hierarchical_shared_depth_ctx2_v3legacy_cls_20260404_run2_lowbs/
│   ├── outputs_xgboost_baselines_v1/
│   ├── outputs_xgboost_explainability_v1/
│   └── outputs_unified_stage123_cam_20260404/
│
├── docs/                              # Documentation
│   ├── MODEL_AND_ALGO.md              # Model and algorithm description
│   ├── DATAFLOW_REPORT.md             # Data flow report
│   ├── ORIGINAL_MODEL_MATH.md         # Mathematical details
│   └── paper/                         # Paper draft (TBME)
│       ├── main.tex
│       ├── sections/
│       └── references.bib
│
└── baselines/                         # Baseline methods
    └── train_xgboost_baselines_lite.py  # XGBoost baseline
```

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 1.12+
- CUDA 11.6+ (for GPU training)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/lung-nodule-localization.git
cd lung-nodule-localization

# Install dependencies
pip install -r requirements.txt
```

### Inference Example

```python
from models.dual_stream_mstcn_detection import DualStreamMSTCN
import torch

# Load model
model = DualStreamMSTCN()
model.load_state_dict(torch.load('checkpoints/stage1_detection.pth'))
model.eval()

# Prepare input: 10-frame sequence of 12x8 tactile arrays
# Shape: (batch, 10, 1, 12, 8)
sensor_data = torch.randn(1, 10, 1, 12, 8)

# Run inference
with torch.no_grad():
    prob, size, depth = model(sensor_data)
    detection_prob = torch.sigmoid(prob)
    print(f"Detection probability: {detection_prob.item():.4f}")
```

### Training

```bash
# Stage 1: Detection
python training/train_stage1_detection.py \
    --data_path data/training_data \
    --output_dir experiments/outputs_stage1

# Stage 2: Size Estimation
python training/train_stage2_raw_positive_size_v2.py \
    --data_path data/training_data \
    --output_dir experiments/outputs_stage2
```

## System Architecture

### Sensor Input

The system uses a **flexible array tactile sensor** that captures 96-dimensional pressure arrays (12×8 grid) at each time step. A sliding window of 10 consecutive frames forms the input sequence.

### Model Architecture

The core model is a **Dual-Stream 3D CNN + LSTM** architecture:

```
Input: 10 frames × 12 × 8 tactile array
       │
       ├─── Shape Stream ──────────────────────┐
       │   3D CNN → Spatiotemporal features    │
       │   (10, 1, 12, 8) → (32,)              │
       │                                       ├──→ Fusion → Multi-task Output
       │   Statistical Stream ─────────────────┤
       │   Mean/Max/Std per frame → BiLSTM     │
       │   (10, 3) → (128,)                    │
       └───────────────────────────────────────┘
                                               │
                                    ┌──────────┼──────────┐
                                    ▼          ▼          ▼
                              Detection   Size (cm)  Depth (cm)
                              (logits)   (regression) (classification)
```

### Three-Stage Pipeline

1. **Stage 1 - Detection**: Binary classification (nodule present/absent)
   - Model: Dual-Stream MSTCN
   - Output: Detection probability

2. **Stage 2 - Size Estimation**: Regression for positive windows
   - Model: Raw Positive Size Model v2
   - Output: Nodule size in cm

3. **Stage 3 - Depth Estimation**: Coarse depth prediction (exploratory)
   - Model: Hierarchical Shared Window MTL
   - Output: Depth category

## Mathematical Details

For complete mathematical formulation including normalization, feature extraction, and loss functions, see [docs/ORIGINAL_MODEL_MATH.md](docs/ORIGINAL_MODEL_MATH.md).

### Key Equations

**Per-frame normalization:**
```
x̃_t = (x_t - min(x_t)) / (max(x_t) - min(x_t))
```

**Statistical features per frame:**
```
μ_t = mean(x_t),  m_t = max(x_t),  σ_t = std(x_t)
```

**Total loss:**
```
L = L_cls + L_size + L_depth
```

Where regression losses are only computed for positive predictions.

## Dataset

The dataset consists of tactile sensor recordings from ex vivo lung tissue experiments. Each recording contains:
- 96-channel tactile sensor data (12×8 grid)
- Manual keyframe labels for nodule presence
- Nodule size and depth annotations

See [data/README.md](data/README.md) for detailed format specifications.

## Reproducibility

All mainline results are locked in [experiments/mainline_release_lock_20260405_final.json](experiments/mainline_release_lock_20260405_final.json).

To reproduce the main results:

```bash
# Reproduce Stage 1 detection results
python training/train_stage1_dualstream_mstcn.py \
    --config experiments/mainline_release_lock_20260405_final.json \
    --stage 1

# Reproduce Stage 2 size results
python training/train_stage2_raw_positive_size_v2.py \
    --config experiments/mainline_release_lock_20260405_final.json \
    --stage 2
```

## Ablation Studies

We conducted comprehensive ablation studies to validate each component:

| Variant | F1 | AUC | Notes |
|---------|-----|-----|-------|
| Full model | 0.6647 | 0.8311 | Dual-stream + LSTM |
| Shape only | - | - | 3D CNN only |
| Intensity only | - | - | Statistical stream only |
| No LSTM | - | - | Mean pooling instead of LSTM |

See [docs/MODEL_AND_ALGO.md](docs/MODEL_AND_ALGO.md) for details.

## Paper Draft

A draft manuscript targeting **IEEE Transactions on Biomedical Engineering (TBME)** is available in [docs/paper/](docs/paper/).

Key sections:
- [Abstract](docs/paper/sections/abstract.tex)
- [Introduction](docs/paper/sections/introduction.tex)
- [Methods](docs/paper/sections/methods.tex)
- [Results](docs/paper/sections/results.tex)
- [Discussion](docs/paper/sections/discussion.tex)

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{lung_nodule_localization_2026,
  title={An Intraoperative Lung Nodule Localization System Based on a Flexible Array Tactile Sensor and Deep Learning Fusion},
  author={Your Name et al.},
  journal={IEEE Transactions on Biomedical Engineering},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions about this repository or the associated paper, please open an issue or contact the authors.
