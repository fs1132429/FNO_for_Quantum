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



class TestFunctions(unittest.TestCase):
    def test_low_energy_states_normalized(self):
        N = 3  # Example number of particles
        num_states = 5  # Example number of states
        populated_states= 2
        input_states = generate_low_energy_states(N, num_states,populated_states)
        for state in input_states:
            self.assertTrue(is_normalized(state), "Input state is not normalized.")
            
    def test_states_sorted_by_population_normalized(self):
        N = 3  # Example number of particles
        num_states = 5  # Example number of states
        input_states = generate_states_sorted_by_population(N, num_states)
        for state in input_states:
            self.assertTrue(is_normalized(state), "Input state is not normalized.")
            
    def test_create_dataset_low_energy_states(self):
        #architecture 1
        n = 4
        num_states = 100
        train_ratio = 0.8
        batch_size = 10
        Jx = -1
        Jy = -1
        Jz = -1
        h = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.complex64)
        populated_states=2
        train_loader, test_loader = create_dataset_low_energy_states(n, num_states=num_states, train_ratio=train_ratio, batch_size=batch_size, populated_states=populated_states,Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=None)
    
        train_size = int(num_states * train_ratio)
        test_size = num_states - train_size

        for data in train_loader:
            x_train, y_train = data['x'], data['y']
            self.assertTrue(x_train.shape == torch.Size([batch_size, 2, 2**n]), f"Expected x_train shape {[batch_size, 2, 2**n]}, got {x_train.shape}")
            self.assertTrue(y_train.shape == torch.Size([batch_size, 1, 2**n]), f"Expected y_train shape {[batch_size, 1, 2**n]}, got {y_train.shape}")

        for data in test_loader:
            x_test, y_test = data['x'], data['y']
            self.assertTrue(x_test.shape == torch.Size([batch_size, 2, 2**n]), f"Expected x_test shape {[batch_size, 2, 2**n]}, got {x_test.shape}")
            self.assertTrue(y_test.shape == torch.Size([batch_size, 1, 2**n]), f"Expected y_test shape {[batch_size, 1, 2**n]}, got {y_test.shape}")

        self.assertTrue(len(train_loader.dataset) == train_size, f"Expected train dataset size {int(num_states * train_ratio)}, got {len(train_loader.dataset)}")
        self.assertTrue(len(test_loader.dataset) == test_size, f"Expected test dataset size {num_states - int(num_states * train_ratio)}, got {len(test_loader.dataset)}")
    
    
    def test_create_dataset_sorted_by_population(self):
        #architecture 1
        N = 4
        num_states = 100
        train_ratio = 0.8
        batch_size = 10
        Jx = -1
        Jy = -1
        Jz = -1
        h = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.complex64)
        
        train_loader, test_loader = create_dataset_sorted_by_population(N, num_states=num_states, train_ratio=train_ratio, batch_size=batch_size, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=None)
    
        train_size = int(num_states * train_ratio)
        test_size = num_states - train_size

        for data in train_loader:
            x_train, y_train = data['x'], data['y']
            self.assertTrue(x_train.shape == torch.Size([batch_size, 2, 2**N]), f"Expected x_train shape {[batch_size, 2, 2**N]}, got {x_train.shape}")
            self.assertTrue(y_train.shape == torch.Size([batch_size, 1, 2**N]), f"Expected y_train shape {[batch_size, 1, 2**N]}, got {y_train.shape}")

        for data in test_loader:
            x_test, y_test = data['x'], data['y']
            self.assertTrue(x_test.shape == torch.Size([batch_size, 2, 2**N]), f"Expected x_test shape {[batch_size, 2, 2**N]}, got {x_test.shape}")
            self.assertTrue(y_test.shape == torch.Size([batch_size, 1, 2**N]), f"Expected y_test shape {[batch_size, 1, 2**N]}, got {y_test.shape}")

        self.assertTrue(len(train_loader.dataset) == train_size, f"Expected train dataset size {int(num_states * train_ratio)}, got {len(train_loader.dataset)}")
        self.assertTrue(len(test_loader.dataset) == test_size, f"Expected test dataset size {num_states - int(num_states * train_ratio)}, got {len(test_loader.dataset)}")
    
    def test_time_dataset_low_energy_states_normalized(self):
        #architecture 2
        n = 4  # Example number of particles
        num_states = 1  # Example number of states
        hamiltonian= construct_hamiltonian(n)
        time=0.314
        steps=100
        populated_states=2
        dataset=create_time_data_set_low_energy_states(n,num_states,populated_states,hamiltonian,time,steps)
        dataset=dataset.squeeze(0)
        for i in range(steps+1):
            state=dataset[:,i]
            self.assertTrue(is_normalized(state), f"State at step {i} is not normalized.")
            
    def test_time_dataset_sorted_by_population_normalized(self):
        #architecture 2
        n = 4  # Example number of particles
        num_states = 1  # Example number of states
        hamiltonian= construct_hamiltonian(n)
        time=0.314
        steps=100
        dataset=create_time_data_set_sorted_by_population(n,num_states,hamiltonian,time,steps)
        dataset=dataset.squeeze(0)
        for i in range(steps+1):
            state=dataset[:,i]
            self.assertTrue(is_normalized(state), f"State at step {i} is not normalized.")
            

    def test_generate_low_energy_states(self):
        n = 4
        num_states = 1
        populated_states = 3
        # Define a Hamiltonian 
        hamiltonian = construct_hamiltonian(n)
        eigenvalues, _ = np.linalg.eig(hamiltonian)
        sorted_eigenvalues= np.sort(eigenvalues)
        unique_eigenvalues = np.unique(sorted_eigenvalues)
        second_lowest_eigenvalue = unique_eigenvalues[1]
        #median_eigenvalue = np.median(sorted_eigenvalues)
        # Generate low energy states
        states = generate_low_energy_states(n, num_states, populated_states)
        def calculate_energy(state, hamiltonian):
            return torch.real(torch.vdot(state, torch.matmul(hamiltonian, state)))
        # Calculate the energy of each state
        energies = [calculate_energy(states[i], hamiltonian) for i in range(num_states)]
        # Verify if the energies are low 
        for energy in energies:
            self.assertTrue(energy < second_lowest_eigenvalue, f"Energy {energy} is not low")
        # Plot the energy spectrum and the energies of the generated states



    def test_population_sorted_energy_states(self):
        n = 4
        num_states = 1
        # Define a Hamiltonian 
        hamiltonian = construct_hamiltonian(n)
        eigenvalues, _ = np.linalg.eig(hamiltonian)
        sorted_eigenvalues= np.sort(eigenvalues)
        unique_eigenvalues = np.unique(sorted_eigenvalues)
        third_lowest_eigenvalue = unique_eigenvalues[-1]
        #median_eigenvalue = np.median(sorted_eigenvalues)
        # Generate low energy states
        states = generate_states_sorted_by_population(n, num_states)
        def calculate_energy(state, hamiltonian):
            return torch.real(torch.vdot(state, torch.matmul(hamiltonian, state)))
        # Calculate the energy of each state
        energies = [calculate_energy(states[i], hamiltonian) for i in range(num_states)]
        # Verify if the energies are low
        for energy in energies:
            self.assertTrue(energy < third_lowest_eigenvalue, f"Energy {energy} is not low")

if __name__ == '__main__':
    unittest.main()
