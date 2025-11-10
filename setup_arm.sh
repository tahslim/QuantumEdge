#!/bin/bash
# setup_arm.sh - Install Arm-optimized dependencies for Raspberry Pi

echo "Updating system..."
sudo apt update -y

echo "Installing system dependencies..."
sudo apt install -y python3-pip python3-opencv libatlas-base-dev

echo "Installing Python packages (excluding onnxruntime for now)..."
pip3 install -r requirements.txt --no-binary=onnxruntime

echo "Downloading ONNX Runtime for Arm64 (Raspberry Pi OS 64-bit)..."
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-aarch64-1.18.0.tgz
tar -xzf onnxruntime-linux-aarch64-1.18.0.tgz

echo "Installing ONNX Runtime..."
pip3 install --upgrade onnxruntime-linux-aarch64-1.18.0/onnxruntime-1.18.0-cp39-cp39-linux_aarch64.whl

echo "Setup complete! Run: python3 demo.py --mode camera"
