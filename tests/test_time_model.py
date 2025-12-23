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

class TestFunctions(unittest.TestCase):
    def test_time_dataset_normalized(self):
        n = 4  # Example number of particles
        num_states = 1  # Example number of states
        hamiltonian= construct_hamiltonian(n)
        time=0.314
        steps=100
        dataset=create_time_data_set(n,num_states,hamiltonian,time,steps)
        dataset=dataset.squeeze(0)
        for i in range(steps+1):
            state=dataset[:,i]
            self.assertTrue(is_normalized(state), f"State at step {i} is not normalized.")
    
    
    def test_data_preprocess(self):
        n = 4  # Example number of particles
        num_states = 100  # Example number of states
        hamiltonian= construct_hamiltonian(n)
        time=0.314
        steps=100
        dataset=create_time_data_set(n,num_states,hamiltonian,time,steps)
        input_T= 20
        output_T= 30
        train_ratio= 0.8
        batch_size= 10
        train_loader, test_loader = data_preprocess(dataset,input_T,output_T,num_states,train_ratio,batch_size)
    
        train_size = int(num_states * train_ratio)
        test_size = num_states - train_size

        for data in train_loader:
            x_train, y_train = data['x'], data['y']
            self.assertTrue(x_train.shape == torch.Size([batch_size,2**n+2,input_T]), f"Expected x_train shape {[batch_size,2**n+2,input_T]}, got {x_train.shape}")
            self.assertTrue(y_train.shape == torch.Size([batch_size,2**n,input_T]), f"Expected y_train shape {[batch_size,2**n, input_T]}, got {y_train.shape}")

        for data in test_loader:
            x_test, y_test = data['x'], data['y']
            self.assertTrue(x_test.shape == torch.Size([batch_size,2**n+2,input_T]), f"Expected x_test shape {[batch_size,2**n+2,input_T]}, got {x_test.shape}")
            self.assertTrue(y_test.shape == torch.Size([batch_size,2**n,input_T]), f"Expected y_test shape {[batch_size,2**n,input_T]}, got {y_test.shape}")

        self.assertTrue(len(train_loader.dataset) == train_size, f"Expected train dataset size {int(num_states * train_ratio)}, got {len(train_loader.dataset)}")
        self.assertTrue(len(test_loader.dataset) == test_size, f"Expected test dataset size {num_states - int(num_states * train_ratio)}, got {len(test_loader.dataset)}")


    def test_fidelity_func(self):
        def isclose(a, b, tol=1e-7):
            return torch.all(torch.abs(a - b) < tol)
        
        # Test case: Orthogonal states
        predictions = torch.tensor([[[1+0j, 0+0j], [0+0j, 1+0j], [0+0j, 0+0j]]], dtype=torch.complex64)
        ground_truth = torch.tensor([[[0+0j, 1+0j], [1+0j, 0+0j], [0+0j, 0+0j]]], dtype=torch.complex64)
        fidelity, avg_fidelity, super_avg = fidelity_func(predictions, ground_truth)
        expected_fidelity = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
        expected_avg_fidelity = torch.tensor([0.0, 0.0], dtype=torch.float32)
        expected_super_avg = 0.0
        self.assertTrue(isclose(fidelity, expected_fidelity), "Fidelity calculation is incorrect.")
        self.assertTrue(isclose(avg_fidelity, expected_avg_fidelity), "Average fidelity calculation is incorrect.")
        self.assertTrue(isclose(super_avg, expected_super_avg), "Super average fidelity calculation is incorrect.")

        # Test case: Identical states
        predictions = torch.tensor([[[1+0j, 0+0j], [0+0j, 1+0j]]], dtype=torch.complex64)
        ground_truth = torch.tensor([[[1+0j, 0+0j], [0+0j, 1+0j]]], dtype=torch.complex64)
        fidelity, avg_fidelity, super_avg = fidelity_func(predictions, ground_truth)
        expected_fidelity = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        expected_avg_fidelity = torch.tensor([1.0, 1.0], dtype=torch.float32)
        expected_super_avg = 1.0
        self.assertTrue(isclose(fidelity, expected_fidelity), "Fidelity of orthogonal states should be 0.")
        self.assertTrue(isclose(avg_fidelity, expected_avg_fidelity), "Average fidelity of orthogonal states should be 0.")
        self.assertTrue(isclose(super_avg, expected_super_avg), "Super average fidelity of orthogonal states should be 0.")

        # Test case: arbitrary complex states
        predictions = torch.tensor([[[3.1+0.6j, 1.4+5j], [3.2+1j, 1-9.8j]],[[1+0j, 9.4+0j], [0+7.4j, 1-4.3j]]], dtype=torch.complex64)
        ground_truth = torch.tensor([[[1+0j, 5+0j], [0+0j, 1+2j]],[[1+0j, 0+4.5j], [0+6.2j, 1+9j]]], dtype=torch.complex64)
        fidelity, avg_fidelity, super_avg = fidelity_func(predictions, ground_truth)
        num_states,N,T=predictions.shape
        expected_fidelity = torch.zeros((num_states, T), dtype=torch.float32)
        for i in range(num_states):
            for j in range(T):
                predictions_input=predictions[i,:,j]
                ground_truth_input= ground_truth[i,:,j]
                inner_product=torch.sum(torch.conj(predictions_input) * ground_truth_input)
                expected_fidelity[i,j]= torch.abs(inner_product) ** 2
        expected_avg_fidelity = torch.mean(fidelity, dim=0) 
        expected_super_avg= torch.mean(fidelity)
        self.assertTrue(isclose(fidelity, expected_fidelity), "Fidelity calculation is incorrect.")
        self.assertTrue(isclose(avg_fidelity, expected_avg_fidelity), "Average fidelity calculation is incorrect.")
        self.assertTrue(isclose(super_avg, expected_super_avg), "Super average fidelity calculation is incorrect.")
    
        
    def test_create_time_dataset(self):
        
        #Test 1: test the end state
        n = 2
        num_states = 4
        steps = 10
        time = 0.314
        # Correct 4x4 Hamiltonian for a 2-particle system
        hamiltonian = construct_hamiltonian(n)
        output_tensor = create_time_data_set(n, num_states, hamiltonian, time, steps)
        #extract input state from the output tensor
        input_states = output_tensor[:, :, 0]
        T= time*steps
        #compute the final state
        expected_final_state = evolve_states(input_states, hamiltonian, T)
        # Extract the final state from the output tensor
        final_output_state = output_tensor[:, :, -1]
        # Check if the final output state matches the expected final state
        self.assertTrue(torch.allclose(final_output_state, expected_final_state, atol=1e-3), "Final state does not match expected state")
        
        #Test 2: test any state at a random time in the dataset
        n = 2
        num_states = 4
        steps = 10
        time = 0.314
        # Correct 4x4 Hamiltonian for a 2-particle system
        hamiltonian = construct_hamiltonian(n)
        output_tensor = create_time_data_set(n, num_states, hamiltonian, time, steps)
        #extract input state from the output tensor
        input_states = output_tensor[:, :, 0]
        #choose any random time step
        random_step= random.randint(0,steps)
        T= time*random_step
        #compute the final state
        expected_final_state = evolve_states(input_states, hamiltonian, T)
        # Extract the final state from the output tensor
        final_output_state = output_tensor[:, :, random_step]
        # Check if the final output state matches the expected final state
        self.assertTrue(torch.allclose(final_output_state, expected_final_state, atol=1e-3), "Final state does not match expected state")
        
    def test_get_predictions2(self):
        # Define a mock model, test_loader, rollout_steps, and spatial_grid
        class MockModel(torch.nn.Module):
            def __init__(self):
                super(MockModel, self).__init__()
                self.fc = torch.nn.Linear(20, 10)  
                
            def forward(self, x):
                return self.fc(x)
        
        model = MockModel().cuda()  # Move model to GPU
        
        # Mock test_loader
        class MockDataLoader:
            def __iter__(self):
                # Mock batches of data with x and y tensors
                for i in range(5):  # 5 batches
                    x = torch.randn(2, 10).cuda()  
                    spatial_grid = torch.randn(2, 10).cuda() 
                    x = torch.cat([x, spatial_grid], dim=1) 
                    y = torch.randn(2, 1).cuda()   
                    yield {'x': x, 'y': y}
        
        test_loader = MockDataLoader()
        
        rollout_steps = 3
        spatial_grid = torch.randn(2, 10).cuda()  
        # Test get_predictions2 function
        predictions = get_predictions2(model, test_loader, rollout_steps, spatial_grid)
        
        # Assertions
        self.assertIsInstance(predictions, torch.Tensor)
        self.assertEqual(predictions.shape, (10, 30)) 
        
if __name__ == '__main__':
    unittest.main()
