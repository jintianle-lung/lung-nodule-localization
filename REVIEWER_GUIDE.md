# Reviewer Guide

Welcome, reviewer! This guide helps you navigate the codebase efficiently.

## Quick Overview

This repository accompanies the paper:

> **An Intraoperative Lung Nodule Localization System Based on a Flexible Array Tactile Sensor and Deep Learning Fusion**

### Key Claims

| Stage | Task | Metric | Value | Status |
|-------|------|--------|-------|--------|
| 1 | Detection | F1 | 0.6647 | **Main result** |
| 1 | Detection | AUC | 0.8311 | **Main result** |
| 2 | Size | Top-2 Acc | 0.8215 | **Main result** |
| 2 | Size | MAE | 0.1338 cm | **Main result** |
| 3 | Depth | Balanced Acc | 0.6251 | *Exploratory* |

## Where to Start

### 1. Understand the System (5 min)
- Read `README.md` for project overview
- Check `experiments/mainline_release_lock.json` for official metrics

### 2. Review Model Architecture (10 min)
- `docs/MODEL_AND_ALGO.md` - Architecture description
- `docs/ORIGINAL_MODEL_MATH.md` - Mathematical formulation
- `models/dual_stream_mstcn_detection.py` - Core detection model

### 3. Check Data Protocol (5 min)
- `data/README.md` - Data format specification
- `models/task_protocol_v1.py` - Size/depth class definitions

### 4. Verify Reproducibility (Optional)
- `docs/REPRODUCIBILITY.md` - Step-by-step reproduction guide

## Code Quality Checklist

- [x] Clear module structure
- [x] Documented model architecture
- [x] Mathematical formulation provided
- [x] Locked experimental results
- [x] Reproducibility instructions
- [x] Baseline comparisons (XGBoost)

## Key Technical Decisions

### Why Dual-Stream?
- **Shape stream**: Captures morphological features via 3D CNN
- **Statistical stream**: Models intensity dynamics via LSTM
- **Fusion**: Combines complementary information

### Why Multi-Scale Temporal Convolution?
- Dilated convolutions capture patterns at multiple time scales
- More efficient than pure LSTM for fixed-length sequences

### Why Ordinal Regression for Size?
- Size classes have natural ordering (0.25cm < 0.5cm < ...)
- Ordinal loss encourages predictions close to ground truth

### Why Detection-Gated Output?
- Prevents false size/depth predictions when no nodule present
- Clinically meaningful: only show properties if nodule detected

## Common Reviewer Questions

### Q: How does the model handle different nodule sizes?
A: The size estimation head uses ordinal regression with 7 classes (0.25 to 1.75 cm). See `models/task_protocol_v1.py` for class definitions.

### Q: What is the input data format?
A: 10-frame sliding windows of 12x8 tactile arrays. See `data/README.md`.

### Q: How is depth estimation handled?
A: Depth is treated as exploratory. The model uses size-conditioned expert routing for coarse depth (shallow/middle/deep) prediction.

### Q: What baselines were compared?
A: XGBoost with handcrafted features. See `baselines/train_xgboost_baseline.py`.

## File Quick Reference

| Purpose | File |
|---------|------|
| Main detection model | `models/dual_stream_mstcn_detection.py` |
| Task definitions | `models/task_protocol_v1.py` |
| Training script | `training/train_stage1_detection.py` |
| Evaluation script | `evaluation/evaluate.py` |
| Official metrics | `experiments/mainline_release_lock.json` |
| Math formulation | `docs/ORIGINAL_MODEL_MATH.md` |

## Contact

For questions about this submission, please contact the corresponding author.
