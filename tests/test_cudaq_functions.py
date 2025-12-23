import os
import sys

# Define the relative path to the parent directory
relative_path = '..'

# Append the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)

import unittest
import torch
import random
import numpy as np
from functions.functions import *
from functions.functions_time_model import *
from functions.functions_cuda import *

def test_hamiltonian_consistency(n_spins, Jx, Jy, Jz, h):
    """
    Test if the CUDA-Q Hamiltonian and normal Hamiltonian produce the same matrix.

    Parameters:
    - n_spins: Number of qubits (spins).
    - Jx, Jy, Jz: Coupling strengths (scalars or lists).
    - h: Local field strengths.

    Returns:
    - bool: True if the matrices are element-wise close, False otherwise.
    """

    # Construct Hamiltonians using both methods
    hamiltonian_cudaq = construct_hamiltonian_cudaq(n_spins, Jx, Jy, Jz, h)
    hamiltonian_normal = construct_hamiltonian(n_spins, Jx, Jy, Jz, h)

    # Convert CUDA-Q Hamiltonian to matrix format
    matrix_cudaq = hamiltonian_cudaq.to_matrix()

    # Compare the matrices using element-wise tolerance
    are_close = np.allclose(matrix_cudaq, hamiltonian_normal, atol=1e-6)

    # Print result
    print(f"Are the matrices element-wise close? {are_close}")

    return are_close

# Example usage
n_spins = 4
Jx, Jy, Jz = -1, -1, -1  # Coupling strengths
h = 0.5  # Local field strength

# Run the test
test_result = test_hamiltonian_consistency(n_spins, Jx, Jy, Jz, h)
assert test_result, "Hamiltonians do not match!"