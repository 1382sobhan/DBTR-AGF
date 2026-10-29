# DBTR-AGF: Dual-Branch Transformer-Recurrent Network with Adaptive Gating Fusion

Official implementation of the DBTR-AGF framework for epileptic seizure prediction.

## Overview
DBTR-AGF is a dual-branch architecture:
- **Temporal Branch:** Transformer + BiLSTM on CWT scalograms.
- **Spectral Branch:** Transformer + TCN on STFT features.
- **Adaptive Gating Fusion (AGF):** Sample-specific weighting of the two branches.
- **Causal HMM Post-processing:** Forward filtering to reduce false alarms.

## Datasets
- CHB-MIT Scalp EEG Database (PhysioNet)
- Bonn University EEG Dataset

## Files
- `preprocessing.py`: Extracts CWT/STFT features and builds the HDF5 dataset.
- `main_pipeline.py`: Model definition, training, HMM, and evaluation protocols.

## Requirements
```bash
pip install -r requirements.txt