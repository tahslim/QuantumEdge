# src/mps_onnx_converter.py
import numpy as np
import onnx
from onnx import numpy_helper
from .quantum_mps import pad_to_power_of_two, tensor_to_mps, mps_contract

class MPSClassifier:
    def __init__(self, onnx_model_path, bond_dim=16):
        # Load ONNX model and extract final linear layer weights
        model = onnx.load(onnx_model_path)
        weights = None
        for init in model.graph.initializer:
            if 'classifier.weight' in init.name or 'Gemm' in str(model.graph.node[-1]):
                weights = numpy_helper.to_array(init)
                break
        if weights is None:
            raise ValueError("Could not find final linear layer in ONNX model")

        # Transpose if needed (ONNX stores [out, in])
        if weights.shape[0] < weights.shape[1]:
            weights = weights.T  # Now [in, out]

        # Compress via MPS
        W_pad, n = pad_to_power_of_two(weights)
        self.mps_cores = tensor_to_mps(W_pad, bond_dim=bond_dim)
        self.input_size = weights.shape[0]

    def predict(self, features):
        """features: (batch, D) → return (batch, num_classes)"""
        if features.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {features.shape[1]}")
        
        preds = []
        for x in features:
            pred = mps_contract(x, self.mps_cores)
            preds.append(pred)
        return np.stack(preds)
