# Hybrid Classical–Quantum Deep Learning (CQDL) for Lung Cancer Detection

## Overview
This project proposes a Hybrid Classical–Quantum Deep Learning framework that integrates:

- Classical Convolutional Neural Network (CNN) for spatial feature extraction
- Quantum Convolutional Neural Network (QCNN) using an 8-qubit variational circuit
- Hybrid integration for enhanced classification performance

The objective is to explore how quantum-enhanced feature representations can improve medical image classification.

---

## Architecture

1. Input: 128x128 grayscale CT slice
2. CNN Feature Extractor
3. 8-dimensional feature embedding
4. AngleEmbedding into 8-qubit quantum circuit
5. Variational entanglement layers (BasicEntanglerLayers)
6. Expectation measurement (Pauli-Z)
7. Final classification layer (Sigmoid)

---

## Tech Stack

- Python
- PyTorch
- PennyLane (Quantum simulation)
- NumPy
- Matplotlib

---

## Repository Structure


---

## Current Status

- Hybrid CNN–QCNN architecture implemented in PyTorch + PennyLane
- 8-qubit variational quantum circuit integrated using AngleEmbedding and entanglement layers
- End-to-end hybrid forward pass verified
- Structural training loop validated using synthetic data

Full dataset training, hyperparameter tuning, and metric benchmarking are currently in progress.

---

## Research Motivation

While classical CNNs demonstrate strong performance in medical image classification, they often require large labeled datasets and may struggle with capturing complex global feature correlations.

Variational quantum circuits operate in high-dimensional Hilbert spaces using superposition and entanglement, potentially enabling richer feature mappings with fewer parameters.

This project explores whether integrating quantum feature transformations with classical convolutional representations can improve generalization and reduce overfitting in medical imaging tasks.

---

## Future Work

- Full-scale training on LIDC-IDRI dataset
- Accuracy, Precision, Recall, F1-score, and AUC benchmarking
- Noise robustness evaluation under simulated quantum noise
- Deployment testing on cloud-based quantum hardware
- Parameter efficiency comparison with classical CNN baseline
