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

class TestFunctions(unittest.TestCase):
    def test_input_states_normalized(self):
        N = 2  # Example number of particles
        num_states = 5  # Example number of states
        input_states = generate_random_input_states_wavefunction(N, num_states)
        for state in input_states:
            self.assertTrue(is_normalized(state), "Random input state is not normalized.")

    def test_hamiltonian_hermicity(self):
        n_particles = 5 # Example number of particles
        J = -1  # Example coupling constant
        hamiltonian = construct_hamiltonian(n_particles, J)
        self.assertTrue(is_hermitian(hamiltonian), "Hamiltonian is not Hermitian.")

    def test_check_hamiltonian_dimension(self):  
        n_particles = 3
        expected_dim = 2 ** n_particles
        hamiltonian = construct_hamiltonian(n_particles)
        self.assertTrue(hamiltonian.shape == (expected_dim, expected_dim), f"Hamiltonian is not of correct dimension {expected_dim}x{expected_dim}")
        
    def test_hamiltonian_with_vector_couplings_and_fields_dimension(self):
        n_particles = 4
        interactions = [(0, 1), (1, 2), (2, 3), (3, 0)]
        Jx = torch.tensor([-1, -1, -1, -1], dtype=torch.complex64)
        Jy = torch.tensor([-1, -1, -1, -1], dtype=torch.complex64)
        Jz = torch.tensor([-1, -1, -1, -1], dtype=torch.complex64)
        h = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.complex64)
        
        hamiltonian = construct_hamiltonian(n_particles, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)
        expected_size = (2**n_particles, 2**n_particles)
        self.assertTrue(hamiltonian.shape == expected_size, f"Expected shape {expected_size}, got {hamiltonian.shape}")

    def test_evolve_states_normalized(self):
        N = 3  # Example number of particles
        num_states = 5  # Example number of states
        input_states = generate_random_input_states_wavefunction(N, num_states)
        hamiltonian = construct_hamiltonian(N)
        times =  3.14 # Example time values
        output_states= torch.zeros_like(input_states)
        output_states = evolve_states(input_states, hamiltonian, times)
        for state in output_states:
            self.assertTrue(is_normalized(state), "Evolved state is not normalized.")
            
    def test_fidelity_state(self):
        # Define two orthogonal states (length 2)
        psi = np.array([1, 0], dtype=np.complex64)
        phi = np.array([0, 1], dtype=np.complex64)
        self.assertTrue(np.isclose(fidelity_state(psi, phi), 0), "Fidelity of orthogonal states should be 0")

        # Define two identical states (length 2)
        self.assertTrue(np.isclose(fidelity_state(psi, psi), 1), "Fidelity of identical states should be 1")

        # Define two arbitrary states (orthogonal, length 2)
        psi = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
        phi = np.array([1, -1], dtype=np.complex64) / np.sqrt(2)
        self.assertTrue(np.isclose(fidelity_state(psi, phi), 0), "Fidelity of orthogonal states should be 0")
        
        # Define two arbitrary states (identical, length 2)
        phi = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
        self.assertTrue(np.isclose(fidelity_state(psi, phi), 1), "Fidelity of identical states should be 1")

        # Define two general non-orthogonal, non-identical complex states (length 2)
        psi = np.array([1+1j, 0], dtype=np.complex64)
        phi = np.array([1, 1-1j], dtype=np.complex64) / np.sqrt(2)
        inner_product = np.sum(np.conj(psi) * phi)
        expected_fidelity = np.abs(inner_product) ** 2
        self.assertTrue(np.isclose(fidelity_state(psi, phi), expected_fidelity), "Fidelity calculation is incorrect for general states")

        # Define two arbitrary complex states (length 4)
        psi = np.array([1, 1j, -1, 0], dtype=np.complex64) / np.sqrt(3)
        phi = np.array([0.5, -0.5j, 0.5, -0.5], dtype=np.complex64) / np.sqrt(2)
        inner_product = np.sum(np.conj(psi) * phi)
        expected_fidelity = np.abs(inner_product) ** 2
        self.assertTrue(np.isclose(fidelity_state(psi, phi), expected_fidelity), "Fidelity calculation is incorrect for general states")

        # Define two arbitrary complex states (length 6)
        psi = np.array([1, 1j, -1, 0, 1+1j, -1-1j], dtype=np.complex64) / np.sqrt(6)
        phi = np.array([0.5, -0.5j, 0.5, -0.5, 0.5+0.5j, -0.5-0.5j], dtype=np.complex64) / np.sqrt(3)
        inner_product = np.sum(np.conj(psi) * phi)
        expected_fidelity = np.abs(inner_product) ** 2
        self.assertTrue(np.isclose(fidelity_state(psi, phi), expected_fidelity), "Fidelity calculation is incorrect for general states")

            
    def test_is_normalized(self):
        # Normalized real-valued vector of length 2
        state_normalized_real = torch.tensor([0.6, 0.8])
        self.assertTrue(is_normalized(state_normalized_real), "Test 1: Function failed, state is normalized")

        # Non-normalized real-valued vector of length 2
        state_non_normalized_real = torch.tensor([1.2, 0.8])
        self.assertFalse(is_normalized(state_non_normalized_real), "Test 2: Function failed, state is not normalized")

        # Normalized complex-valued vector of length 2
        state_normalized_complex = torch.tensor([0.38470837+0.8768447j, -0.07769712-0.27767968j])
        self.assertTrue(is_normalized(state_normalized_complex), "Test 3: Function failed, state is normalized")

        # Non-normalized complex-valued vector of length 2
        state_non_normalized_complex = torch.tensor([1.2 + 0.3j, 0.8 + 0.6j])
        self.assertFalse(is_normalized(state_non_normalized_complex), "Test 4: Function failed, state is not normalized")

        # Normalized complex-valued vector of length 3
        state_normalized_complex_3 = torch.tensor([-0.27515581+0.06650674j,0.33781785-0.47744086j,  0.75149509+0.11423922j])
        self.assertTrue(is_normalized(state_normalized_complex_3), "test 5: Function failed, state is normalized")

        # Non-normalized complex-valued vector of length 3
        state_non_normalized_complex_3 = torch.tensor([1.0 + 0.2j, 1.0 + 0.3j, 1.0 + 0.4j])
        self.assertFalse(is_normalized(state_non_normalized_complex_3), "Test 6: Function failed, state is not normalized")
        
    def test_is_hermitian(self):
        # Hermitian matrix (2x2)
        matrix_hermitian = torch.tensor([[1.0, 2.0 ], [2.0 , 3.0]])
        self.assertTrue(is_hermitian(matrix_hermitian), "Test 1: It is a hermitian matrix!")

        # Non-Hermitian matrix (2x2)
        matrix_non_hermitian = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertFalse(is_hermitian(matrix_non_hermitian), "test 2: It is not a hermitian matrix!")

        # Hermitian matrix with complex values (2x2)
        matrix_hermitian_complex = torch.tensor([[1.0, 1.0 - 2j], [1.0 + 2j, 3.0]])
        self.assertTrue(is_hermitian(matrix_hermitian_complex), "Test 3: It is a hermitian matrix!")

        # Non-Hermitian matrix with complex values (2x2)
        matrix_non_hermitian_complex = torch.tensor([[1.0, 1.0 - 2j], [1.0 - 2j, 3.0]])
        self.assertFalse(is_hermitian(matrix_non_hermitian_complex), "Test 4: It is not a hermitian matrix!")

        # Hermitian matrix with complex values (3x3)
        matrix_hermitian_complex_3x3 = torch.tensor([[1.0, 2.0 - 1j, 3.0 + 2j], [2.0 + 1j, 4.0, 5.0 - 1j], [3.0 - 2j, 5.0 + 1j, 6.0]])
        self.assertTrue(is_hermitian(matrix_hermitian_complex_3x3), "Test 5: It is a hermitian matrix!")

        # Non-Hermitian matrix with complex values (3x3)
        matrix_non_hermitian_complex_3x3 = torch.tensor([[1.0, 2.0 + 1j, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        self.assertFalse(is_hermitian(matrix_non_hermitian_complex_3x3), "Test 6: It is not a hermitian matrix!")
        
    def test_input_states_shape(self):
        n = 4  # Example number of particles
        num_states = 100  # Example number of states
        input_states = generate_random_input_states_wavefunction(n, num_states)
        expected_shape = (num_states, 2**n)  # Expected shape of the input states
        self.assertTrue(input_states.shape == expected_shape, "Shape is incorrect")
        
    def test_output_states_shape(self):
        n = 5  # Example number of particles
        num_states = 20  # Example number of states
        input_states = generate_random_input_states_wavefunction(n, num_states)
        hamiltonian= construct_hamiltonian(n)
        times=3.14
        output_states= evolve_states(input_states,hamiltonian,times)
        expected_shape = (num_states, 2**n)  # Expected shape of the input states
        self.assertTrue(output_states.shape == expected_shape, "Shape is incorrect")
    
    def test_extend_S(self):
        n = 4
        S = 1/2* torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        pos = [1, 3] 
        expected_output = torch.kron(torch.eye(2, dtype=torch.complex64),torch.kron(S,torch.kron(torch.eye(2, dtype=torch.complex64),S)))
        output = extend_S(n, S, pos)
        self.assertTrue(torch.allclose(expected_output, output), "extend_S function did not produce the expected output")

        
    def test_construct_hamiltonian(self):
        #define terms
        n_particles = 3
        Jx = 1
        Jy = -1
        Jz = 0.5
        h = torch.tensor([0.1, 0.2, 0.3], dtype=torch.complex64)
        
        sx = 1/2* torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sy = 1/2* torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sz = 1/2* torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        I = torch.eye(2, dtype=torch.complex64)
        
        #define n tensor product function
        def kron_n(*args):
            result = args[0]
            for mat in args[1:]:
                result = torch.kron(result, mat)
            return result
        
        hx_term = Jx * (
            kron_n(sx, sx, I) +
            kron_n(I, sx, sx) +
            kron_n(sx, I, sx)
        )
        
        hy_term = Jy * (
            kron_n(sy, sy, I) +
            kron_n(I, sy, sy) +
            kron_n(sy, I, sy)
        )
        
        hz_term = Jz * (
            kron_n(sz, sz, I) +
            kron_n(I, sz, sz) +
            kron_n(sz, I, sz)
        )
        
        hz_local = (
            h[0] * kron_n(sz, I, I) +
            h[1] * kron_n(I, sz, I) +
            h[2] * kron_n(I, I, sz)
        )
        #expected hamiltonian
        expected_hamiltonian = hx_term + hy_term + hz_term + hz_local
        #hamiltonian using the original function
        constructed_hamiltonian = construct_hamiltonian(n_particles, Jx, Jy, Jz, h)
        self.assertTrue(torch.allclose(constructed_hamiltonian, expected_hamiltonian, atol=1e-6), "Constructed Hamiltonian does not match the expected Hamiltonian")

    def test_construct_hamiltonian_with_vectors(self):
        # Define terms
        n_particles = 3
        interactions = [(0, 1), (0, 2)]
        Jx = torch.tensor([1, 0.7], dtype=torch.complex64)
        Jy = torch.tensor([-1, 0], dtype=torch.complex64)
        Jz = torch.tensor([0, -0.5], dtype=torch.complex64)
        h = torch.tensor([0.1, 0.2, 0.3], dtype=torch.complex64)

        sx = 1 / 2 * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sy = 1 / 2 * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sz = 1 / 2 * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        I = torch.eye(2, dtype=torch.complex64)
        
        def kron_n(*args):
            result = args[0]
            for mat in args[1:]:
                result = torch.kron(result, mat)
            return result

        # Expected Hamiltonian
        hx_term = Jx[0] * kron_n(sx, sx, I) + Jx[1] * kron_n(sx, I, sx) 
        hy_term = Jy[0] * kron_n(sy, sy, I) + Jy[1] * kron_n(sy, I, sy)
        hz_term = Jz[0] * kron_n(sz, sz, I) + Jz[1] * kron_n(sz, I, sz)

        hz_local = (
            h[0] * kron_n(sz, I, I) +
            h[1] * kron_n(I, sz, I) +
            h[2] * kron_n(I, I, sz)
        )

        expected_hamiltonian = hx_term + hy_term + hz_term + hz_local

        # Hamiltonian using the construct_hamiltonian function
        constructed_hamiltonian = construct_hamiltonian(n_particles, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)

        self.assertTrue(torch.allclose(constructed_hamiltonian, expected_hamiltonian, atol=1e-6),"Constructed Hamiltonian does not match the expected Hamiltonian")

    def test_create_dataset(self):
        N = 4
        num_states = 100
        train_ratio = 0.8
        batch_size = 10
        Jx = -1
        Jy = -1
        Jz = -1
        h = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.complex64)
        
        train_loader, test_loader = create_dataset(N, num_states=num_states, train_ratio=train_ratio, batch_size=batch_size, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=None)
    
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
    
    def test_input_states_complex(self):
        N = 4  # Example number of particles
        num_states = 100  # Example number of states
        input_states = generate_random_input_states_wavefunction(N, num_states)
        
        # Check if the input states are complex
        self.assertTrue(input_states.dtype.is_complex, "Input states are not complex")

    def test_output_states_complex(self):
        N = 4  # Example number of particles
        num_states = 100  # Example number of states
        input_states = generate_random_input_states_wavefunction(N, num_states)
        hamiltonian = construct_hamiltonian(N)
        times = 3.14
        output_states = evolve_states(input_states, hamiltonian, times)
        # Check if the output states are complex
        self.assertTrue(output_states.dtype.is_complex, "Output state is not complex")
        
    def test_evolve_states(self):
        #define terms
        n_particles = 3
        num_states=10
        #times=torch.linspace(0, 3.14/2, 700)
        times=3.14/2
        Jx = 1
        Jy = -1
        Jz = 0.5
        h = torch.tensor([0.1, 0.2, 0.3], dtype=torch.complex64)
        
        sx = 1/2* torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
        sy = 1/2* torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
        sz = 1/2* torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
        I = torch.eye(2, dtype=torch.complex64)
        
        #define n tensor product function
        def kron_n(*args):
            result = args[0]
            for mat in args[1:]:
                result = torch.kron(result, mat)
            return result
        
        hx_term = Jx * (
            kron_n(sx, sx, I) +
            kron_n(I, sx, sx) +
            kron_n(sx, I, sx)
        )
        
        hy_term = Jy * (
            kron_n(sy, sy, I) +
            kron_n(I, sy, sy) +
            kron_n(sy, I, sy)
        )
        
        hz_term = Jz * (
            kron_n(sz, sz, I) +
            kron_n(I, sz, sz) +
            kron_n(sz, I, sz)
        )
        
        hz_local = (
            h[0] * kron_n(sz, I, I) +
            h[1] * kron_n(I, sz, I) +
            h[2] * kron_n(I, I, sz)
        )
        #expected hamiltonian
        expected_hamiltonian = hx_term + hy_term + hz_term + hz_local
        
        #hamiltonian using the original function
        constructed_hamiltonian = construct_hamiltonian(n_particles, Jx, Jy, Jz, h)
        
        #input_states
        input_states= generate_random_input_states_wavefunction(n_particles,num_states)
        
        output_states= evolve_states(input_states,constructed_hamiltonian,times)
        expected_output_states = torch.zeros_like(input_states, dtype=torch.complex64)
        #Unitary evolution operator
        evolution_operator = torch.matrix_exp(-1j * expected_hamiltonian * times)
        for i in range(num_states):
            state = input_states[i]
            #Output wavefunction
            evolved_state = torch.matmul(evolution_operator, state)
            expected_output_states[i] = evolved_state     
        self.assertTrue(torch.allclose(output_states, expected_output_states, atol=1e-6), "Evolved states do not match the expected evolved states")

    def test_transform_ordered_input_states_back(self):
        n=5
        num_states=100
        input_states= generate_random_input_states_wavefunction(n,num_states)
        hamiltonian=construct_hamiltonian(n)
        times=3.14
        #ordered_input_states,_, indices,_ = ordered_data(input_states,hamiltonian,times)
        indices,_=ordered_indices(n,hamiltonian)
        ordered_input_states= order_input_states(input_states,indices)
    
        # Transform back ordered_input_states 
        back_transformed_states = transform_back(ordered_input_states, indices)
        
        # Check if we get back the original states and Hamiltonian
        self.assertTrue(torch.allclose(input_states, back_transformed_states, atol=1e-6), "Original input states and transformed back input states do not match")
        
    def test_transform_ordered_output_states_back(self):
        n=5
        num_states=100
        input_states= generate_random_input_states_wavefunction(n,num_states)
        hamiltonian=construct_hamiltonian(n)
        times=3.14
        output_states=evolve_states(input_states,hamiltonian)
            
        #_,ordered_output, indices,_ = ordered_data(input_states,hamiltonian)
        indices,_=ordered_indices(n,hamiltonian)
        ordered_input_states= order_input_states(input_states,indices)
        ordered_hamiltonian= order_hamiltonian(hamiltonian,indices)
        ordered_output= evolve_states(ordered_input_states,ordered_hamiltonian,times)
        # Transform back ordered_input_states 
        back_transformed_states = transform_back(ordered_output, indices)
        
        # Check if we get back the original states and Hamiltonian
        self.assertTrue(torch.allclose(output_states, back_transformed_states, atol=1e-6), "Original output states and transformed back output states do not match")
    
        
        
if __name__ == '__main__':
    unittest.main()

