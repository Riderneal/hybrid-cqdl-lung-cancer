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

