# Neuromorphic Device ML Workflow

Python workflow for analyzing conductance switching data from a neuromorphic device and using the fitted device response in a device-inspired MNIST classification experiment.

The project combines experimental conductance-cycle analysis, LTP/LTD curve fitting, CDF heatmap generation, and a custom NumPy-based neural-network simulation inspired by memristive/neuromorphic hardware.

## Overview

This repository contains two main analysis pipelines:

1. **Cycle analysis**
   - Loads repeated conductance-cycle data from CSV.
   - Computes conductance from measured current and voltage.
   - Separates potentiation and depression branches.
   - Builds CDF heatmaps of conductance updates.
   - Generates interpolated heatmaps for visualization.

2. **Device-fit MNIST experiment**
   - Loads LTP/LTD conductance data from Excel.
   - Normalizes and fits potentiation/depression curves.
   - Generates experimental, fitted, smooth, and noisy conductance-update heatmaps.
   - Uses the fitted device response in a device-inspired neural-network workflow.
   - Trains a custom NumPy implementation of a `784 -> 128 -> 10` ReLU neural network on MNIST.
   - Saves accuracy curves, confusion matrices, and processed data files.

The goal is to connect measured device-level switching behavior with machine-learning performance in a reproducible Python workflow.

## Repository Structure

```text
neuromorphic-device-ml/
├── data/
│   └── raw/
│       ├── S_13_P_D_C_final_50_cycle.csv
│       └── LTP_LTD_CsPbI3_perovskite.xlsx
├── outputs/
│   ├── cycle_analysis/
│   └── device_fit_mnist/
├── scripts/
│   ├── run_all.py
│   ├── run_cycle_analysis.py
│   └── run_device_fit_mnist.py
├── README.md
└── requirements.txt
