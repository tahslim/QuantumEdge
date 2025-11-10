# 🚀 QuantumEdge — Quick-Start Cheat Sheet  
*Quantum-Inspired On-Device AI for Raspberry Pi*

> ✅ Runs **offline** • ✅ Uses **Arm Neon** • ✅ Open Source (MIT)

---

## 📦 Requirements
- **Hardware**: Raspberry Pi 4 or 5 (2GB+ RAM)  
- **OS**: Raspberry Pi OS (64-bit, Bookworm)  
- **Optional**: Pi Camera (for live demo)

---

## ⏱️ 5-Minute Setup

```bash
# 1. Clone repo
git clone https://github.com/yourname/QuantumEdge && cd QuantumEdge

# 2. Install Arm-optimized deps
chmod +x setup_arm.sh
./setup_arm.sh

# 3. (If needed) Fetch a base model
mkdir -p models
wget -O models/mobilenet_cifar10.onnx \
  https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-7.onnx

# 4. Run!
python3 demo.py --mode camera    # or --mode sensor
