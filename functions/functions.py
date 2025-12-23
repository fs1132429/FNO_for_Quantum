
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.embeddings import PositionalEmbedding


# Define spin operators for a single particle
sx = (1/2) * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sy = (1/2) * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sz = (1/2) * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
# Identity 
I = torch.eye(2, dtype=torch.complex64)

def generate_random_input_states_wavefunction(n, num_states):
    # Generate random complex amplitudes for each spin state
    real_part = torch.rand(num_states, 2**n) * 2 - 1  # Random real part between -1 and 1
    imag_part = torch.rand(num_states, 2**n) * 2 - 1  # Random imag part between -1 and 1
    amplitudes = real_part + 1j * imag_part
    # Calculate normalization factor
    norm = torch.sqrt(torch.sum(torch.square(real_part), dim=1) + torch.sum(torch.square(imag_part), dim=1))
    # Normalize the amplitudes 
    amplitudes /= norm.view(-1, 1)
    return amplitudes.to(torch.complex64)

def evolve_states(input_states, hamiltonian, time=3.14): #times=torch.linspace(0, 3.14, 700)
    num_states, _ = input_states.shape
    output_states = torch.zeros_like(input_states, dtype=torch.complex64)
    # Unitary evolution operator
    evolution_operator = torch.matrix_exp(-1j * hamiltonian * time)
    for i in range(num_states):
        state = input_states[i]
        # Output wavefunction
        evolved_state = torch.matmul(evolution_operator, state)
        output_states[i] = evolved_state 
    return output_states


def extend_S(n, S, pos):
    # Initialize the total S operator with identity
    S_total = torch.eye(1, dtype=torch.complex64)
    for i in range(n):
        # If i is in pos, use S; otherwise, use I
        S_total = torch.kron(S_total, S if i in pos else I)
    return S_total

def construct_hamiltonian(n_particles, Jx=None, Jy=None, Jz=None, h=None, interactions=None):
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
    
    # Ensure Jx, Jy, and Jz are tensors of the correct length
    if Jx is None:
        Jx = -1 * torch.ones(len(interactions), dtype=torch.complex64)
    elif isinstance(Jx, (int, float)):
        Jx = torch.tensor([Jx] * len(interactions), dtype=torch.complex64)

    if Jy is None:
        Jy = -1 * torch.ones(len(interactions), dtype=torch.complex64)
    elif isinstance(Jy, (int, float)):
        Jy = torch.tensor([Jy] * len(interactions), dtype=torch.complex64)

    if Jz is None:
        Jz = -1 * torch.ones(len(interactions), dtype=torch.complex64)
    elif isinstance(Jz, (int, float)):
        Jz = torch.tensor([Jz] * len(interactions), dtype=torch.complex64)

    if h is None:
        h = torch.zeros(n_particles, dtype=torch.complex64)
    elif isinstance(h, (int, float)):
        h = torch.tensor([h] * n_particles, dtype=torch.complex64)
    
    hamiltonian = torch.zeros((2**n_particles, 2**n_particles), dtype=torch.complex64)
    
    for idx, (i, j) in enumerate(interactions):
        # Interaction terms
        sx_i_sx_j = extend_S(n_particles, sx, [i]) @ extend_S(n_particles, sx, [j])
        sy_i_sy_j = extend_S(n_particles, sy, [i]) @ extend_S(n_particles, sy, [j])
        sz_i_sz_j = extend_S(n_particles, sz, [i]) @ extend_S(n_particles, sz, [j])
        
        hamiltonian += Jx[idx] * sx_i_sx_j + Jy[idx] * sy_i_sy_j + Jz[idx] * sz_i_sz_j
    
    for i in range(n_particles):
        h_i = h[i]
        sz_i = extend_S(n_particles, sz, [i])  # single qubit
        hamiltonian += h_i * sz_i

    return hamiltonian

def ordered_indices(n,hamiltonian):
    N=2**n
    energy_values = torch.zeros(N, dtype=torch.float32)
    
    # looping over amplitudes in psi_i
    for j in range(N):  # N = 2**n
        state = torch.zeros(N, dtype=torch.complex64)
        # converting amplitudes to basis states
        state[j] = 1+0j
        # expectation value to get energy
        x = state.conj().permute(*torch.arange(state.conj().ndim - 1, -1, -1))  # x = state.conj().T
        energy_component = torch.real(torch.matmul(x, torch.matmul(hamiltonian, state)))
        energy_values[j] = energy_component.item()
    
    # getting indices for sorted energy_values
    sorted_indices = torch.argsort(energy_values)
    # storing sorted energy values
    sorted_energy_values = energy_values[sorted_indices]

    return sorted_indices, sorted_energy_values

def order_hamiltonian(hamiltonian, sorted_indices):
    # ordering the hamiltonian
    reordered_hamiltonian = hamiltonian[:, sorted_indices][sorted_indices, :]
    return reordered_hamiltonian

def order_input_states(input_states,sorted_indices):
    # ordering the psi_i based on energy_values
    reordered_states = input_states[:,sorted_indices]
    return reordered_states
    
def transform_back(states, indices):
    num_states,_= states.shape
    original_states = torch.zeros_like(states,dtype=torch.complex64)
    for i, index in enumerate(indices):
        original_states[:, index] = states[:, i]
    return original_states

def is_normalized(state, tol=1e-6):
    norm = torch.norm(state)
    return torch.isclose(norm, torch.tensor(1.0), atol=tol)

def is_hermitian(matrix):
    return torch.allclose(matrix, matrix.conj().T)

def create_dataset(n, num_states, train_ratio, batch_size, times=3.14, Jx=None, Jy=None, Jz=None, h=None, interactions=None): #times=torch.linspace(0, 3.14, 700)
    # Generate input states
    input_states = generate_random_input_states_wavefunction(n, num_states)
    
    # Construct Hamiltonian
    hamiltonian = construct_hamiltonian(n, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)
    
    #ordered indices based on energy level
    sorted_indices,_= ordered_indices(n,hamiltonian)
    
    #order hamiltonian
    ordered_hamiltonian= order_hamiltonian(hamiltonian, sorted_indices)
    
    #order input_states
    ordered_input_states = order_input_states(input_states,sorted_indices)
    
    #ordered output
    ordered_output_states= evolve_states(ordered_input_states,ordered_hamiltonian,times)
    
    # Ordered input and output states
    #ordered_input_states, ordered_output_states, index, energy_values = ordered_data(input_states, hamiltonian,times)
    
    # Generate energy grid
    energy_grid = torch.linspace(0, 1, 2**n).unsqueeze(0).expand(num_states, -1)
    #energy_grid= energy_values.unsqueeze(0).expand(num_states, -1)
    
    # Concatenate spatial grid with input states
    train_input = torch.cat([ordered_input_states.unsqueeze(1), energy_grid.unsqueeze(1)], dim=1)  # train input for FNO needs to be of the form [num_states, in_channels, input_states_wavefunction]
    train_output = ordered_output_states.unsqueeze(1) # train output to FNO needs to be of the form [num_states, out_channels, input_states_wavefunction]
    
    # Split data into training and testing sets
    train_size = int(train_ratio * num_states)
    train_input_final, train_output_final = train_input[:train_size], train_output[:train_size]
    test_input, test_output = train_input[train_size:], train_output[train_size:]

    print(f'[Dataset] x_train: {train_input_final.shape}, y_train: {train_output_final.shape}')
    print(f'[Dataset] x_test: {test_input.shape}, y_test: {test_output.shape}')
    
    # Create dictionaries for train and test data
    train_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(train_input_final, train_output_final)]
    test_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(test_input, test_output)]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



# A function to plot comparison between predictions and ground truth, and plot error
def plot_comparison_with_error(predictions, ground_truth):
    # Calculate error
    error_real = np.abs(predictions.real - ground_truth.real)
    error_imag = np.abs(predictions.imag - ground_truth.imag)

    plt.figure(figsize=(20, 8))

    # Plot predictions and ground truth (real part)
    plt.subplot(1, 2, 1)
    plt.plot(predictions.real, label='Prediction (Real)')
    plt.plot(ground_truth.real, label='Ground Truth (Real)')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Amplitude', fontsize=15)  # Increase font size for y-axis label
    plt.title('Real Part - Prediction vs Ground Truth',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    # Plot predictions and ground truth (imaginary part)
    plt.subplot(1, 2, 2)
    plt.plot(predictions.imag, label='Prediction (Imaginary)')
    plt.plot(ground_truth.imag, label='Ground Truth (Imaginary)')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Amplitude', fontsize=15)  # Increase font size for y-axis label
    plt.title('Imaginary Part - Prediction vs Ground Truth',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    plt.tight_layout()
    plt.show()

    # Plot error (real and imaginary)
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.plot(error_real, label='Error (Real)', color='red')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Error', fontsize=15)  # Increase font size for y-axis label
    plt.title('Real Part - Error Comparison',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    plt.subplot(1, 2, 2)
    plt.plot(error_imag, label='Error (Imaginary)', color='blue')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Error', fontsize=15)  # Increase font size for y-axis label
    plt.title('Imaginary Part - Error Comparison',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    plt.tight_layout()
    plt.show()

def plot_comparison_with_error_pauli_strings(predictions, ground_truth):
    # Calculate error
    error_real = np.abs(predictions - ground_truth)

    plt.figure(figsize=(20, 8))

    # Plot predictions and ground truth (real part)
    plt.subplot(1, 2, 1)
    plt.plot(predictions, label='Prediction (Real)')
    plt.plot(ground_truth, label='Ground Truth (Real)')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Amplitude', fontsize=15)  # Increase font size for y-axis label
    plt.title('Real Part - Prediction vs Ground Truth',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    plt.tight_layout()
    plt.show()

    # Plot error (real and imaginary)
    plt.figure(figsize=(20, 8))

    plt.subplot(1, 2, 1)
    plt.plot(error_real, label='Error (Real)', color='red')
    plt.xlabel('Basis State', fontsize=15)  # Increase font size for x-axis label
    plt.ylabel('Error', fontsize=15)  # Increase font size for y-axis label
    plt.title('Real Part - Error Comparison',fontsize=15)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    plt.tight_layout()
    plt.show()


def fidelity_state(predictions, ground_truth):
    # Calculate inner product
    #inner_product=torch.sum(torch.conj(predictions) * ground_truth)
    inner_product = np.vdot(predictions, ground_truth)
    # Compute fidelity
    fidelity = np.abs(inner_product) ** 2
    return fidelity

def get_predictions(model, test_loader, rollout_steps, spatial_grid):
    all_predictions = []
    with torch.no_grad():
        for batch in test_loader:
            batch_predictions = []
            x, y = batch['x'].cuda(), batch['y'].cuda()  # Move data to the model's device
            # Initial prediction without autoregressive rollout
            predictions = model(x)
            batch_predictions.append(predictions.squeeze(1))
            # Perform auto-regressive rollout
            for _ in range(rollout_steps - 1):
                predictions = torch.cat([predictions, spatial_grid.unsqueeze(1)], dim=1) 
                predictions = model(predictions)
                batch_predictions.append(predictions.squeeze(1)) 
            # Append predictions and ground truth for this batch
            all_predictions.append(torch.stack(batch_predictions, dim=0)) 
    # Concatenate predictions and ground truth across batches
    all_predictions = torch.cat(all_predictions, dim=1) 
    return all_predictions


def get_ground_truth(test_loader, rollout_steps, hamiltonian, times):
    all_ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            batch_ground_truths = []
            x, y = batch['x'].cuda(), batch['y'].cuda()  # Move data to the model's device
            y = y.squeeze(1)  # Squeeze y if necessary
            batch_ground_truths.append(y)
            # Perform auto-regressive rollout
            for _ in range(rollout_steps - 1):
                # Ground truth for comparison
                ground_truth = evolve_states(y, hamiltonian.cuda(), times) 
                batch_ground_truths.append(ground_truth)
                y= ground_truth
            # Append predictions and ground truth for this batch
            all_ground_truths.append(torch.stack(batch_ground_truths, dim=0))  
    # Concatenate predictions and ground truth across batches
    all_ground_truths = torch.cat(all_ground_truths, dim=1)
    return all_ground_truths


def autoregressive_rollout(model, test_loader, rollout_steps, spatial_grid, hamiltonian, times):
    all_predictions=get_predictions(model,test_loader,rollout_steps,spatial_grid)
    all_ground_truths= get_ground_truth(test_loader,rollout_steps,hamiltonian,times)
    return all_predictions, all_ground_truths

"""
#delete after above test
def autoregressive_rollout2(model, test_loader, rollout_steps, spatial_grid, hamiltonian, times):
    all_predictions = []
    all_ground_truths = []

    with torch.no_grad():
        for batch in test_loader:
            batch_predictions = []
            batch_ground_truths = []

            x, y = batch['x'].cuda(), batch['y'].cuda()  # Move data to the model's device
            y = y.squeeze()  # Squeeze y if necessary

            # Initial prediction without autoregressive rollout
            predictions = model(x)
            batch_predictions.append(predictions.squeeze(1))
            batch_ground_truths.append(y)

            # Perform auto-regressive rollout
            for _ in range(rollout_steps - 1):
                predictions = torch.cat([predictions, spatial_grid.unsqueeze(-1)], dim=1)
                predictions = model(predictions)
                batch_predictions.append(predictions.squeeze(1))
                
                # Ground truth for comparison
                ground_truth = evolve_states(y, hamiltonian.cuda(), times) 
                batch_ground_truths.append(ground_truth)
                y= ground_truth

            # Append predictions and ground truth for this batch
            all_predictions.append(torch.stack(batch_predictions, dim=0)) 
            all_ground_truths.append(torch.stack(batch_ground_truths, dim=0))  
    # Concatenate predictions and ground truth across batches
    all_predictions = torch.cat(all_predictions, dim=1)
    all_ground_truths = torch.cat(all_ground_truths, dim=1)
    return all_predictions, all_ground_truths
"""

def calculate_mse(predictions, ground_truth):
    """Calculate Mean Squared Error separately for real and imaginary parts."""
    real_error = np.square(predictions.real - ground_truth.real)
    imag_error = np.square(predictions.imag - ground_truth.imag)
    mse_real = np.mean(real_error)
    mse_imag = np.mean(imag_error)
    return mse_real, mse_imag


import numpy as np
import matplotlib.pyplot as plt
import torch
import cudaq
from typing import List
from functions.functions_pauli_strings import *


def generate_random_input_states_wavefunction(n, num_states):
    # Generate random complex amplitudes for each spin state
    real_part = torch.rand(num_states, 2**n) * 2 - 1  # Random real part between -1 and 1
    imag_part = torch.rand(num_states, 2**n) * 2 - 1  # Random imag part between -1 and 1
    amplitudes = real_part + 1j * imag_part
    # Calculate normalization factor
    norm = torch.sqrt(torch.sum(torch.square(real_part), dim=1) + torch.sum(torch.square(imag_part), dim=1))
    # Normalize the amplitudes 
    amplitudes /= norm.view(-1, 1)
    return amplitudes.to(torch.complex64)

def generate_wavefunctions_shared_sparsity(n, num_states, sparsity=0.1, epsilon=1e-4):
    dim = 2**n
    num_significant = max(1, int(sparsity * dim))

    # Select same sparsity mask for all wavefunctions
    indices = torch.randperm(dim)
    significant_idx = indices[:num_significant]
    near_zero_idx = indices[num_significant:]

    # Create random complex amplitudes
    real_significant = torch.rand(num_states, num_significant) * 2 - 1
    imag_significant = torch.rand(num_states, num_significant) * 2 - 1
    values_significant = real_significant + 1j * imag_significant

    real_small = torch.rand(num_states, dim - num_significant) * 2 - 1
    imag_small = torch.rand(num_states, dim - num_significant) * 2 - 1
    values_small = epsilon * (real_small + 1j * imag_small)

    # Fill into the full amplitude tensor
    amplitudes = torch.zeros((num_states, dim), dtype=torch.complex64)
    amplitudes[:, significant_idx] = values_significant
    amplitudes[:, near_zero_idx] = values_small

    # Normalize
    norms = torch.linalg.norm(amplitudes, dim=1, keepdim=True)
    amplitudes /= norms

    return amplitudes


# Extract term coefficients from the Hamiltonian
def termCoefficients(op: cudaq.SpinOperator) -> List[complex]:
    result = []
    op.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

# Extract term words from the Hamiltonian
def termWords(op: cudaq.SpinOperator) -> List[str]:
    result = []
    op.for_each_term(lambda term: result.append(term.to_string(False)))
    return result

@cudaq.kernel
def trotter(state: cudaq.State, coefficients: List[complex],
            words: List[cudaq.pauli_word], dt: float):
    q = cudaq.qvector(state)
    for i in range(len(coefficients)):
        exp_pauli(coefficients[i].real * dt, q, words[i]) 

def create_time_data_set_cudaq2(n_spins, num_states, hamiltonian, dt, n_steps):
    input_states = generate_random_input_states_wavefunction(n_spins, num_states)
    
    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n_spins, n_steps + 1, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    for state_idx, random_state in enumerate(input_states):
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        # Perform the simulation
        state = initial_state
        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            sv = np.array(state)
            output_tensor[state_idx, :, i] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor

            
    
def create_time_data_set_cudaq(n, num_states, hamiltonian, time, steps):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)

    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n, steps + 1, dtype=torch.complex64
    )
    dimensions = {i: 2 for i in range(n)}
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        
        # Time steps for evolution
        time_steps = np.linspace(0, time, steps + 1)
        schedule = cudaq.Schedule(time_steps, ["time"])
        
        # Perform time evolution
        evolution_result = cudaq.evolve(
            hamiltonian=hamiltonian,
            dimensions=dimensions,
            schedule=schedule,
            initial_state=initial_state,
            store_intermediate_results=True,
        )

        # Extract and rearrange intermediate states
        for step_idx, state in enumerate(evolution_result.intermediate_states()):
            # Convert state vector to complex numpy array
            sv = np.array(state)
            # Assign to output tensor
            output_tensor[state_idx, :, step_idx] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor


#here added 1/2 as it is in our def of spin operators but not in cuda quantum's
def construct_hamiltonian_cudaq(
    n_particles, Jx=None, Jy=None, Jz=None, h=None, interactions=None):
    """
    Construct a generalized Hamiltonian using CUDA Quantum.

    Parameters:
    - n_particles: Number of particles (qubits) in the system.
    - Jx, Jy, Jz: Coupling strengths for X, Y, and Z interactions. Can be scalars or lists.
    - h: Local field strengths. Can be a scalar or a list.
    - interactions: Custom list of qubit pairs defining interactions.

    Returns:
    - cudaq.SpinOperator: The constructed Hamiltonian.
    """
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
    
    # Ensure Jx, Jy, and Jz are numpy arrays of the correct length
    if Jx is None:
        Jx = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jx, (int, float)):
        Jx = np.array([Jx] * len(interactions), dtype=complex)

    if Jy is None:
        Jy = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jy, (int, float)):
        Jy = np.array([Jy] * len(interactions), dtype=complex)

    if Jz is None:
        Jz = -1 * np.ones(len(interactions), dtype=complex)
    elif isinstance(Jz, (int, float)):
        Jz = np.array([Jz] * len(interactions), dtype=complex)

    if h is None:
        h = np.zeros(n_particles, dtype=complex)
    elif isinstance(h, (int, float)):
        h = np.array([h] * n_particles, dtype=complex)

    # Initialize the Hamiltonian
    hamiltonian = 0
    # Add interaction terms
    for idx, (i, j) in enumerate(interactions):
        hamiltonian += Jx[idx] * (1/2)*cudaq.spin.x(i) * (1/2)*cudaq.spin.x(j)
        hamiltonian += Jy[idx] * (1/2)*cudaq.spin.y(i) * (1/2)*cudaq.spin.y(j)
        hamiltonian += Jz[idx] * (1/2)*cudaq.spin.z(i) * (1/2)*cudaq.spin.z(j)

    # Add local field terms
    for i in range(n_particles):
        hamiltonian += h[i] * (1/2)*cudaq.spin.z(i)

    return hamiltonian



@cudaq.kernel
def trotter(state: cudaq.State, coefficients: List[complex],
            words: List[cudaq.pauli_word], dt: float):
    q = cudaq.qvector(state)
    for i in range(len(coefficients)):
        #flip the sign as exp_pauli implements e^{iHt}
        exp_pauli(-coefficients[i].real * dt, q, words[i]) 
        
# Extract term coefficients from the Hamiltonian
def termCoefficients(op: cudaq.SpinOperator) -> List[complex]:
    result = []
    op.for_each_term(lambda term: result.append(term.get_coefficient()))
    return result

# Extract term words from the Hamiltonian
def termWords(op: cudaq.SpinOperator) -> List[str]:
    result = []
    op.for_each_term(lambda term: result.append(term.to_string(False)))
    return result

def create_trotter_time_data_set_cudaq(n, num_states, hamiltonian, dt, n_steps):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)

    # Initialize the output tensor
    output_tensor = torch.zeros(
        num_states, 2**n, n_steps, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        print(f"Running for state {state_idx}")
        # Convert the current random state to a CUDA Quantum State
        initial_state = cudaq.State.from_data(random_state.numpy())
        # Perform the simulation
        state = initial_state
        for i in range(n_steps):
            # Trotter step to evolve the state
            state = cudaq.get_state(trotter, state, coefficients, words, dt)
            sv = np.array(state)
            output_tensor[state_idx, :, i] = torch.tensor(sv, dtype=torch.complex64)
    return output_tensor


def create_trotter_discretized_time_data_set_cudaq(n, num_states, hamiltonian, dt, n_steps,step_interval):
    # Generate all random input states as a list of PyTorch tensors
    input_states = generate_random_input_states_wavefunction(n, num_states)
    
    selected_steps = n_steps // step_interval
    
    # Initialize the output tensor
    output_tensor = torch.zeros(
    num_states, 2**n, selected_steps, dtype=torch.complex64
    )
    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)
    
    # Process each state individually
    for state_idx, random_state in enumerate(input_states):
        # Convert to CUDA Quantum state
        initial_state = cudaq.State.from_data(random_state.numpy())
        
        # Perform simulation
        state = initial_state
        store_idx = 0  # Index to store in output_tensor
        for i in range(n_steps):
            # Apply Trotter step
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

            # Store only if i is divisible by 100
            if i % step_interval == 0:
                sv = np.array(state)
                output_tensor[state_idx, :, store_idx] = torch.tensor(sv, dtype=torch.complex64)
                store_idx += 1  # Move to the next storage index
    return output_tensor


def create_trotter_discretized_time_data_set_with_expectations(n, num_states, hamiltonian, dt, n_steps, step_interval, pauli_strings, sparsity=0.1):
    """
    Generate Trotter-evolved states and compute expectation values for given Pauli strings.
    """

    selected_steps = n_steps // step_interval

    # Generate input states
    input_states = generate_wavefunctions_shared_sparsity(n, num_states,sparsity=sparsity)

    # Initialize output tensors
    output_tensor = torch.zeros((num_states, 2**n, selected_steps), dtype=torch.complex64)
    expectation_tensor = torch.zeros((num_states, len(pauli_strings), selected_steps), dtype=torch.float32)

    coefficients = termCoefficients(hamiltonian)
    words = termWords(hamiltonian)

    # Precompute Pauli operators
    pauli_ops = [generate_pauli_matrix(''.join(ps)) for ps in pauli_strings]

    for state_idx, random_state in enumerate(input_states):
        # Convert to CUDA Quantum state
        state = cudaq.State.from_data(random_state.numpy())
        store_idx = 0

        for i in range(n_steps):
            state = cudaq.get_state(trotter, state, coefficients, words, dt)

            if i % step_interval == 0:
                sv = np.array(state)
                psi = torch.tensor(sv, dtype=torch.complex64)
                output_tensor[state_idx, :, store_idx] = psi

                # Expectation values for each Pauli string
                for p_idx, op in enumerate(pauli_ops):
                    expectation_tensor[state_idx, p_idx, store_idx] = torch.vdot(psi, torch.matmul(op, psi)).real

                store_idx += 1

    return expectation_tensor

import os
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, data_path, input_T, output_T, num_states, device, train=True, train_ratio=0.8):
        self.data_path = data_path
        self.files = sorted(os.listdir(data_path))[:num_states]
        self.input_T = input_T
        self.output_T = output_T
        self.train_ratio = train_ratio
        self.train = train
        self.device = device

        self.train_size = int(train_ratio * len(self.files))
        self.file_subset = self.files[:self.train_size] if train else self.files[self.train_size:]

    def __len__(self):
        return len(self.file_subset)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_subset[idx])
        data = torch.load(file_path, map_location=self.device)

        input_data = data[:, :, 0:self.input_T].to(self.device)
        output_data = data[:, :, self.output_T:self.input_T + self.output_T].to(self.device)

        pos_embedding = PositionalEmbedding(2).to(self.device)
        timesteps = torch.linspace(0, self.input_T - 1, self.input_T, device=self.device)
        positional_embeddings = pos_embedding(timesteps)
        pos = positional_embeddings.T.unsqueeze(0).repeat(1, 1, 1)

        input_data = torch.cat([input_data, pos], dim=1)

        return {'x': input_data.squeeze(0), 'y': output_data.squeeze(0)}

def data_preprocess2(data_path, input_T, output_T, num_states, train_ratio, batch_size, device):
    train_dataset = MyDataset(data_path, input_T, output_T, num_states, device=device, train=True, train_ratio=train_ratio)
    test_dataset = MyDataset(data_path, input_T, output_T, num_states, device=device, train=False, train_ratio=train_ratio)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'[Dataset] x_train: {len(train_dataset)}, x_test: {len(test_dataset)}')
    return train_loader, test_loader


import os
import torch
from torch.utils.data import Dataset, DataLoader

def compute_welford_mean_std(data_path, num_states, input_T, output_T, device, train_ratio=0.8, eps=1e-7):
    """
    Compute mean and std per channel using Welford’s algorithm over all (x,y)
    in the training subset.
    """
    files = sorted(os.listdir(data_path))[:num_states]
    train_size = int(train_ratio * len(files))
    train_files = files[:train_size]

    mean = None
    m2 = None
    count = 0

    for fname in train_files:
        data = torch.load(os.path.join(data_path, fname), map_location=device)
        # Combine x and y along time dimension
        combined = data[:, :, :input_T + output_T]  # shape [1, channels, time]
        # Flatten over time (but keep channels)
        B, C, T = combined.shape
        combined = combined.view(C, T).T  # [T, C]
        for t in range(T):
            x = combined[t]
            count += 1
            if mean is None:
                mean = torch.zeros_like(x)
                m2 = torch.zeros_like(x)
            delta = x - mean
            mean += delta / count
            delta2 = x - mean
            m2 += delta * delta2

    variance = m2 / max(count - 1, 1)
    std = torch.sqrt(variance + eps)
    return mean, std

# -------------------------------
# 3. Normalized Dataset
# -------------------------------
class NormalizedDataset(Dataset):
    def __init__(self, data_path, input_T, output_T, num_states, device, mean, std, train=True, train_ratio=0.8):
        self.data_path = data_path
        self.files = sorted(os.listdir(data_path))[:num_states]
        self.input_T = input_T
        self.output_T = output_T
        self.train = train
        self.device = device
        self.mean = mean.to(device)
        self.std = std.to(device)

        train_size = int(train_ratio * len(self.files))
        self.file_subset = self.files[:train_size] if train else self.files[train_size:]

    def __len__(self):
        return len(self.file_subset)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_path, self.file_subset[idx])
        data = torch.load(file_path, map_location=self.device)

        input_data = data[:, :, 0:self.input_T]
        output_data = data[:, :, self.output_T:self.input_T + self.output_T]

        # Normalize along channel dimension (same mean/std across time)
        input_data = (input_data - self.mean[None, :, None]) / (self.std[None, :, None] + 1e-7)
        output_data = (output_data - self.mean[None, :, None]) / (self.std[None, :, None] + 1e-7)

        # Add positional embeddings
        pos_embedding = PositionalEmbedding(2).to(self.device)
        timesteps = torch.linspace(0, self.input_T - 1, self.input_T, device=self.device)
        positional_embeddings = pos_embedding(timesteps)
        pos = positional_embeddings.T.unsqueeze(0).repeat(1, 1, 1)

        input_data = torch.cat([input_data, pos], dim=1)

        return {'x': input_data.squeeze(0), 'y': output_data.squeeze(0)}

# -------------------------------
# 4. Preprocessing function
# -------------------------------
def data_preprocess_normalize(data_path, input_T, output_T, num_states, train_ratio, batch_size, device):
    print("[Normalization] Computing mean/std using Welford’s algorithm...")
    mean, std = compute_welford_mean_std(data_path, num_states, input_T, output_T, device, train_ratio)
    print("[Normalization] Done.")

    train_dataset = NormalizedDataset(data_path, input_T, output_T, num_states, device, mean, std, train=True, train_ratio=train_ratio)
    test_dataset = NormalizedDataset(data_path, input_T, output_T, num_states, device, mean, std, train=False, train_ratio=train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"[Dataset] x_train: {len(train_dataset)}, x_test: {len(test_dataset)}")
    return train_loader, test_loader, mean, std