import os
import sys

# Define the relative path to the parent directory
relative_path = '..'

# Append the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)

import unittest
import torch
import numpy as np
from functions.functions import *
from functions.functions_time_model import *
from functions.functions_different_inputs import *
from functions.functions_pauli_strings import *


class TestFunctions(unittest.TestCase):
    def test_generate_pauli_strings(self):
        #for nearest next neighbour particle interaction
        test_cases = [
            {
                'n_particles': 2,
                'Jx': [1, 1],
                'Jy': [1, 1],
                'Jz': [1, 1],
                'h': [0, 0],
                'expected': [
                    ('X', 'X'),
                    ('Y', 'Y'),
                    ('Z', 'Z')
                ]
            },
            {
                'n_particles': 3,
                'Jx': [1, 0, 1],
                'Jy': [0, 1, 0],
                'Jz': [1, 1, 0],
                'h': [0, 1, 0],
                'expected': [
                    ('X', 'X', 'I'), 
                    ('Z', 'Z', 'I'), 
                    ('I', 'Y', 'Y'), 
                    ('I', 'Z', 'Z'), 
                    ('X', 'I', 'X'), 
                    ('I', 'Z', 'I')
                ]
            },
        ]

        for i, case in enumerate(test_cases):
            n_particles = case['n_particles']
            Jx = case['Jx']
            Jy = case['Jy']
            Jz = case['Jz']
            h = case['h']
            expected = case['expected']
            
            result = generate_pauli_strings(n_particles, Jx=Jx, Jy=Jy, Jz=Jz, h=h)
            
            result_set = set(result)
            expected_set = set(expected)
            
            self.assertTrue(result_set == expected_set, f"Test case {i} failed: {result_set} != {expected_set}")




    def test_generate_pauli_matrix(self):
        test_cases = [
            {
                'pauli_string': 'IX',
                'expected': torch.kron(identity, sigma_x)
            },
            {
                'pauli_string': 'YZ',
                'expected': torch.kron(sigma_y, sigma_z)
            },
            {
                'pauli_string': 'XIZ',
                'expected': torch.kron(torch.kron(sigma_x, identity), sigma_z)
            },
            {
                'pauli_string': 'II',
                'expected': torch.kron(identity, identity)
            },
            {
                'pauli_string': 'ZXY',
                'expected': torch.kron(torch.kron(sigma_z, sigma_x), sigma_y)
            },
        ]

        for i, case in enumerate(test_cases):
            pauli_string = case['pauli_string']
            expected = case['expected']
            
            result = generate_pauli_matrix(pauli_string)
            
            self.assertTrue(torch.allclose(result, expected), f"Test case {i} failed: {result} != {expected}")
            
    
    def test_pauli_string_exp_val_1(self):
        n_particles = 2
        num_states = 1  # Using one state for simplicity
        Jx = [1]
        Jy = [1]
        Jz = [1]
        h = [1, 1]
        interactions = [(0, 1)]

        # Construct the Hamiltonian
        hamiltonian = construct_hamiltonian(n_particles, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)

        # Generate a simple wavefunction
        wavefunctions = torch.tensor([[1, 1, 0, 0]], dtype=torch.complex64) / torch.sqrt(torch.tensor(2.0, dtype=torch.complex64))

        # Calculate ordered indices and sorted energy values
        sorted_indices, sorted_energy_values = ordered_indices(n_particles, hamiltonian)

        # Reorder the wavefunctions according to sorted indices
        ordered_wavefunction = wavefunctions[:, sorted_indices]

        # Define test cases
        test_cases = [
            {
                'pauli_strings': [('X', 'I'), ('Z', 'Z')],
                'ordered_wavefunction': ordered_wavefunction,
                'sorted_indices': sorted_indices,
                'expected': torch.tensor([[0.0, 0.0]])
            },
        ]

        for i, case in enumerate(test_cases):
            pauli_strings = case['pauli_strings']
            ordered_wavefunction = case['ordered_wavefunction']
            sorted_indices = case['sorted_indices']
            expected = case['expected']
            
            result = pauli_string_exp_val_1(pauli_strings, ordered_wavefunction, sorted_indices)
            self.assertTrue(torch.allclose(result, expected), f"Test case {i} failed: {result} != {expected}")
            
        


if __name__ == '__main__':
    unittest.main()
