# demo.py – Main demo runner

import os
import time
import argparse
import numpy as np
import cv2
from PIL import Image
from onnxruntime import InferenceSession
from src.quantum_mps import pad_to_power_of_two, tensor_to_mps, mps_contract
from src.mps_onnx_converter import MPSClassifier

# Dummy baseline model weights (replace with real ONNX export in practice)
# Inside demo.py, add new function:


def run_camera_demo():
    print("🚀 Starting QuantumEdge with REAL MobileNet + MPS...")
    import onnxruntime as ort

    backbone_path = "models/mobilenet_cifar10.onnx"
    if not os.path.exists(backbone_path):
        print("❌ ONNX model missing. Run export_mobilenet_onnx.py first.")
        return

    # Load CNN backbone (without final classifier)
    backbone = ort.InferenceSession(backbone_path)
    mps_classifier = MPSClassifier(backbone_path, bond_dim=16)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break

        # Preprocess as CIFAR-10
        img = cv2.resize(frame, (32, 32))
        img = img.astype(np.float32) / 255.0
        img = (img - [0.4914, 0.4822, 0.4465]) / [0.2023, 0.1994, 0.2010]
        img = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,32,32)

        # Run backbone
        feat = backbone.run(None, {'input': img})[0]  # (1, 1024)
        feat = feat.reshape(1, -1)

        # MPS classification
        start = time.perf_counter()
        logits = mps_classifier.predict(feat)
        latency = (time.perf_counter() - start) * 1000

        label = f"Class {np.argmax(logits)} | {latency:.1f}ms"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("QuantumEdge – Arm AI Challenge", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
