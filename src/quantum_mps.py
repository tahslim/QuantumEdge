# src/quantum_mps.py
import numpy as np
from scipy.linalg import svd

def pad_to_power_of_two(W):
    """Zero-pad matrix to nearest power-of-two dimensions."""
    D_in, D_out = W.shape
    max_dim = max(D_in, D_out)
    n = int(np.ceil(np.log2(max_dim)))
    size = 2 ** n
    W_padded = np.zeros((size, size), dtype=W.dtype)
    W_padded[:D_in, :D_out] = W
    return W_padded, n

def tensor_to_mps(tensor, bond_dim=16):
    """Convert a 2^n x 2^n matrix into MPS cores."""
    n = int(np.log2(tensor.shape[0]))
    tensor = tensor.reshape([2] * (2 * n))
    cores = []
    current = tensor

    for site in range(2 * n - 1):
        # Reshape into matrix: left vs right
        left_shape = current.shape[:site+1]
        right_shape = current.shape[site+1:]
        left_dim = np.prod(left_shape)
        right_dim = np.prod(right_shape)
        mat = current.reshape(left_dim, right_dim)

        # Truncated SVD
        U, S, Vh = svd(mat, full_matrices=False)
        chi = min(bond_dim, len(S))
        U = U[:, :chi] * np.sqrt(S[:chi])
        Vh = np.sqrt(S[:chi])[:, None] * Vh[:chi, :]

        # Reshape U back to core
        core_shape = list(left_shape) + [chi]
        core = U.reshape(core_shape)
        cores.append(core)

        # Update current for next iteration
        Vh_shape = [chi] + list(right_shape)
        current = Vh.reshape(Vh_shape)

    cores.append(current)  # last core
    return cores

def mps_contract(input_vec, mps_cores):
    """Contract input with MPS cores using einsum."""
    # Reshape input to [2,2,...,2]
    n = len(mps_cores) // 2
    x = input_vec.reshape([2] * n)
    
    # Contract left-to-right
    result = x
    for core in mps_cores:
        if result.ndim == 1:
            result = np.einsum('i,ijk->jk', result, core)
        else:
            result = np.einsum('...i,ijk->...jk', result, core)
        result = result.reshape(-1, result.shape[-1])
    
    return result.flatten()
