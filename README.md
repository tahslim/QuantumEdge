# QuantumEdge: Quantum-Inspired On-Device AI for Raspberry Pi

> 🏆 Arm AI Developer Challenge Submission  
> Runs **entirely offline** on Raspberry Pi 4/5 using Arm Cortex-A72/A76 CPUs  
> Combines **quantum tensor network compression** + **lightweight neural inference** for real-time edge intelligence

---

## 🌟 Overview
**QuantumEdge** is a novel on-device AI framework that uses **quantum-inspired tensor decomposition** to compress neural networks, enabling faster, leaner, and more energy-efficient inference on Arm-based edge devices—without cloud dependency.

- **Problem Solved**: Standard AI models are too large/slow for edge use.
- **Innovation**: Replaces dense layers with **Matrix Product State (MPS)** representations from quantum many-body physics.
- **Platform**: Fully optimized for **Raspberry Pi OS (64-bit)** using **Arm Neon SIMD** and **ONNX Runtime**.

---

## 📦 Requirements
- Raspberry Pi 4 or 5 (2GB+ RAM recommended)
- Raspberry Pi OS (64-bit, Bookworm)
- Pi Camera (optional, for demo)
- Python 3.9+
- `pip3 install -r requirements.txt`

---

## ⚙️ Installation

```bash
git clone https://github.com/yourname/QuantumEdge
cd QuantumEdge
./setup_arm.sh          # Installs ONNX Runtime for Arm64 + dependencies
python3 demo.py --mode camera      # or --mode sensor
