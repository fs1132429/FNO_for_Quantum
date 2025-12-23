import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.data_transforms import DefaultDataProcessor
from loguru import logger

from neuralop.quantum.functions import *
from neuralop.quantum.functions_different_inputs import *
from neuralop.quantum.functions_time_model import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loguru import logger

# Define spin operators for a single particle
sigma_x = (1/2) * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = (1/2) * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = (1/2) * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
# Identity 
identity = torch.eye(2, dtype=torch.complex64)

def generate_pauli_strings(n_particles, interactions=None, Jx=None, Jy=None, Jz=None, h=None):
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
        
    #define pauli string set
    pauli_strings = set()
    
    for idx, (i, j) in enumerate(interactions):
        if idx < len(Jx) and Jx[idx] != 0:
            pauli_strings.add(tuple(('X' if k == i or k == j else 'I') for k in range(n_particles)))
        if idx < len(Jy) and Jy[idx] != 0:
            pauli_strings.add(tuple(('Y' if k == i or k == j else 'I') for k in range(n_particles)))
        if idx < len(Jz) and Jz[idx] != 0:
            pauli_strings.add(tuple(('Z' if k == i or k == j else 'I') for k in range(n_particles)))
    
    for i in range(n_particles):
        if i < len(h) and h[i] != 0:
            pauli_strings.add(tuple(('Z' if k == i else 'I') for k in range(n_particles)))
    
    return list(pauli_strings)

def generate_pauli_matrix(pauli_string):
    # Convert the Pauli string into a matrix
    operator = torch.eye(1, dtype=torch.cfloat)
    for pauli in pauli_string:
        if pauli == 'I':
            operator = torch.kron(operator, identity)
        elif pauli == 'X':
            operator = torch.kron(operator, sigma_x)
        elif pauli == 'Y':
            operator = torch.kron(operator, sigma_y)
        elif pauli == 'Z':
            operator = torch.kron(operator, sigma_z)
    return operator
    


#Architecture 1


# Function to compute the expectation value of a Pauli string
def pauli_string_exp_val_1(pauli_strings, ordered_wavefunction,sorted_indices):
    num_states,_= ordered_wavefunction.shape
    expectation_val = torch.zeros((num_states, len(pauli_strings)), dtype=torch.float32)
    for idx,ps in enumerate(pauli_strings):
        pauli_string=''.join(ps)
        operator=generate_pauli_matrix(pauli_string)
        for i in range(num_states):
            psi= ordered_wavefunction[i,:]
            ordered_operator= order_hamiltonian(operator,sorted_indices) #ordering the operator according to the sorted indices
            expectation_val[i,idx] = torch.vdot(psi, torch.matmul(ordered_operator, psi)) #.real
    return expectation_val

"""
class EnergyGridConcatenator():

    def __init__(self, energy_range=(0, 1)):
        self.energy_range = energy_range

    def generate_energy_grid(self, num_values, num_states, device='cpu', dtype=torch.float32):
        energy_grid = torch.linspace(self.energy_range[0], self.energy_range[1], num_values, device=device, dtype=dtype)
        return energy_grid.unsqueeze(0).expand(num_states, -1)

    def __call__(self, input_states,batched=True):
        num_states,pauli_strings= input_states.shape
        energy_grid = self.generate_energy_grid(pauli_strings,num_states, device=input_states.device, dtype=input_states.dtype)
        train_input = torch.cat([input_states.unsqueeze(1), energy_grid.unsqueeze(1)], dim=1)
        return train_input
"""

def create_dataset_pauli_strings(n, num_states, pauli_strings, train_ratio, batch_size, times=3.14, random=True, populated_states=3,Jx=None, Jy=None, Jz=None, h=None, interactions=None): 
    # Construct Hamiltonian
    hamiltonian = construct_hamiltonian(n, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)
    
    #ordered indices based on energy level
    sorted_indices,_= ordered_indices(n,hamiltonian)
    
    if random:
        # Generate input states
        input_states = generate_random_input_states_wavefunction(n, num_states)
        #order input_states
        ordered_input_states = order_input_states(input_states,sorted_indices)
    else:
        ordered_input_states= generate_low_energy_states(n,num_states,populated_states)
    
    #order hamiltonian
    ordered_hamiltonian= order_hamiltonian(hamiltonian, sorted_indices)
    
    #ordered output
    ordered_output_states= evolve_states(ordered_input_states,ordered_hamiltonian,times)
    
    input_pauli_strings= pauli_string_exp_val_1(pauli_strings,ordered_input_states,sorted_indices)
    
    output_pauli_strings= pauli_string_exp_val_1(pauli_strings,ordered_output_states,sorted_indices)
    
    # Generate energy grid
    energy_grid = torch.linspace(0, 1, len(pauli_strings)).unsqueeze(0).expand(num_states, -1)
    
    train_input= input_pauli_strings
    # Concatenate spatial grid with input states
    train_input = torch.cat([input_pauli_strings.unsqueeze(1), energy_grid.unsqueeze(1)], dim=1)  # train input for FNO needs to be of the form [num_states, in_channels, input_states_wavefunction]
    train_output = output_pauli_strings.unsqueeze(1) # train output to FNO needs to be of the form [num_states, out_channels, input_states_wavefunction]
    
    # Split data into training and testing sets
    train_size = int(train_ratio * num_states)
    train_input_final, train_output_final = train_input[:train_size], train_output[:train_size]
    test_input, test_output = train_input[train_size:], train_output[train_size:]

    print(f'[Dataset] x_train: {train_input_final.shape}, y_train: {train_output_final.shape}')
    print(f'[Dataset] x_test: {test_input.shape}, y_test: {test_output.shape}')
    
    input_encoder= UnitGaussianNormalizer(dim=[0,-1])
    input_encoder.fit(train_input_final,batch_size)
    
    output_encoder=UnitGaussianNormalizer(dim=[0,-1])
    output_encoder.fit(train_output_final,batch_size)
    
    #pos_encoding= EnergyGridConcatenator()
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
    )
    # Create dictionaries for train and test data
    train_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(train_input_final, train_output_final)]
    test_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(test_input, test_output)]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data_processor


def mse_pauli_string_1(predictions, ground_truth): #for l2 change mean to sum
    """Calculate Mean Squared Error separately for real and imaginary parts."""
    real_error = np.square(predictions - ground_truth)
    mse_real = np.mean(real_error)
    return mse_real

def mean_relative_error_1(predictions, ground_truth):
    """Calculate Mean Relative Error."""
    # Avoid division by zero by adding a small epsilon to the denominator
    epsilon = 1e-7
    relative_error = np.abs(predictions - ground_truth) / (np.abs(ground_truth) + epsilon)
    mean_relative_error = np.mean(relative_error)
    return mean_relative_error


#Architecture 2

def create_time_data_set_pauli_string(n,num_states,hamiltonian,time,steps):
    input_states= generate_random_input_states_wavefunction(n,num_states)
    #times=torch.linspace(0,final_T,time_step)
    index,_=ordered_indices(n,hamiltonian)
    ordered_input=order_input_states(input_states,index)
    ordered_hamiltonian= order_hamiltonian(hamiltonian,index)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        logger.info(f"step:{i}")
        output_states= evolve_states(ordered_input,ordered_hamiltonian,time)
        output_tensor[:,:,i] = output_states
        ordered_input=output_states
        logger.info("executed")
    return output_tensor


# Function to compute the expectation value of a Pauli string
def pauli_string_exp_val_2(pauli_strings, x,sorted_indices):
    num_states,_,input_t= x.shape
    expectation_val = torch.zeros((num_states, len(pauli_strings),input_t), dtype=torch.float32)
    for idx,ps in enumerate(pauli_strings):
        pauli_string=''.join(ps)
        operator= generate_pauli_matrix(pauli_string)
        for i in range(num_states):
            for j in range(input_t):
                psi= x[i,:,j]
                ordered_operator= order_hamiltonian(operator,sorted_indices)
                expectation_val[i,idx,j] = torch.vdot(psi, torch.matmul(ordered_operator, psi)).real
    return expectation_val

#function to extract input_t and output_t steps from dataset and compute expectation values
def dataset_pauli_string(x,n,input_T,output_T,start_index,interactions,Jx,Jy,Jz,h):
    start_index=0
    # get num_frames from the input
    input = x[:, :, start_index : start_index + input_T]
    output = x[:, :, start_index + output_T: start_index + input_T + output_T]
    hamiltonian= construct_hamiltonian(n,Jx,Jy,Jz,h,interactions)
    sorted_indices,_=ordered_indices(n,hamiltonian)
    pauli_stings= generate_pauli_strings(n,interactions,Jx,Jy,Jz,h)
    input_pauli_strings= pauli_string_exp_val_2(pauli_stings,input,sorted_indices)
    output_pauli_strings= pauli_string_exp_val_2(pauli_stings,output,sorted_indices)
    return input_pauli_strings,output_pauli_strings

def data_preprocess_pauli_string(x,n,input_T,output_T,num_states,train_ratio,batch_size,start_index=0,interactions=None,Jx=None,Jy=None,Jz=None,h=None):
    #start_index = np.random.randint(0, x.shape[-1] - T)
    if interactions is None:
        interactions = [(i, (i + 1) % n) for i in range(n)]
    
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
        h = torch.zeros(n, dtype=torch.complex64)
    elif isinstance(h, (int, float)):
        h = torch.tensor([h] * n, dtype=torch.complex64)
        
    input,train_output= dataset_pauli_string(x,n,input_T,output_T,start_index,interactions,Jx,Jy,Jz,h)
    pos_embedding = PositionalEmbedding(2) #higher atleat 8
    timesteps = torch.linspace(start_index, start_index+input_T,input_T)
    positional_embeddings = pos_embedding(timesteps)
    pos=positional_embeddings.T.repeat(num_states, 1, 1)
    train_input = torch.cat([input,pos],dim=1)
    train_size = int(train_ratio * num_states)
    train_input_final, train_output_final = train_input[:train_size], train_output[:train_size]
    test_input, test_output = train_input[train_size:], train_output[train_size:]

    print(f'[Dataset] x_train: {train_input_final.shape}, y_train: {train_output_final.shape}')
    print(f'[Dataset] x_test: {test_input.shape}, y_test: {test_output.shape}')
    
    input_encoder= UnitGaussianNormalizer(dim=[0,-1])
    input_encoder.partial_fit(train_input_final,batch_size)
    
    output_encoder=UnitGaussianNormalizer(dim=[0,-1])
    output_encoder.partial_fit(train_output_final,batch_size)
    
    #pos_encoding= EnergyGridConcatenator()
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
    )
    # Create dictionaries for train and test data
    train_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(train_input_final, train_output_final)]
    test_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(test_input, test_output)]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader,data_processor

def mse_pauli_string_2(predictions,ground_truth): 
    mse_loss = (predictions - ground_truth)**2
    mse= torch.mean(mse_loss,dim=0)
    average_mse = torch.mean(mse, dim=-1) 
    super_avg= torch.mean(mse)
    return mse, average_mse,super_avg

def mean_relative_error_2(predictions, ground_truth,epsilon=1):
    # Calculate the relative error
    relative_error =torch.abs(predictions - ground_truth) / (torch.abs(ground_truth)+epsilon) 
    mean_relative_error=torch.mean(relative_error,dim=0)
    average_mean_relative_error = torch.mean(mean_relative_error, dim=0) 
    super_avg_mean_relative_error = torch.mean(mean_relative_error)
    
    return mean_relative_error, average_mean_relative_error, super_avg_mean_relative_error

def mean_absolute_error_2(predictions, ground_truth):
    # Calculate the relative error
    relative_error =torch.abs(predictions - ground_truth) 
    mean_relative_error=torch.mean(relative_error,dim=0)
    average_mean_relative_error = torch.mean(mean_relative_error, dim=0) 
    super_avg_mean_relative_error = torch.mean(mean_relative_error)
    
    return mean_relative_error, average_mean_relative_error, super_avg_mean_relative_error

def load_dataset_pauli(dataset,input_t,output_t,start_index,num_states,train_ratio,batch_size):
    input,train_output= dataset[:num_states,:,start_index:input_t],dataset[:num_states,:,output_t:output_t+input_t]

    pos_embedding = PositionalEmbedding(2) #higher atleat 8
    timesteps = torch.linspace(start_index, start_index+input_t,input_t)
    positional_embeddings = pos_embedding(timesteps)
    pos=positional_embeddings.T.repeat(num_states, 1, 1)

    train_input = torch.cat([input,pos],dim=1)

    train_size = int(train_ratio * num_states)

    train_input_final, train_output_final = train_input[:train_size], train_output[:train_size]
    test_input, test_output = train_input[train_size:], train_output[train_size:]

    print(f'[Dataset] x_train: {train_input_final.shape}, y_train: {train_output_final.shape}')
    print(f'[Dataset] x_test: {test_input.shape}, y_test: {test_output.shape}')

    input_encoder= UnitGaussianNormalizer(dim=[0,-1])
    input_encoder.partial_fit(train_input_final,batch_size)

    output_encoder=UnitGaussianNormalizer(dim=[0,-1])
    output_encoder.partial_fit(train_output_final,batch_size)

    #pos_encoding= EnergyGridConcatenator()
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
    )
    # Create dictionaries for train and test data
    train_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(train_input_final, train_output_final)]
    test_data = [{'x': input_tensor, 'y': output_tensor} for input_tensor, output_tensor in zip(test_input, test_output)]

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader,data_processor



def filter_by_thresh(ground_truth, predictions, threshold):
    gt_values_matrix = []
    pred_values_matrix = []
    relative_errors_matrix = []
    mse_error_matrix = []

    # Iterate over the tensor and check values
    for k in range(ground_truth.shape[2]):
        gt_row_ = []
        pred_row_ = []
        rel_error_row_ = []
        mse_error_row_ = []
        for j in range(ground_truth.shape[1]):
            gt_row = []
            pred_row = []
            rel_error_row = []
            mse_error_row = []
            for i in range(ground_truth.shape[0]):
                if np.abs(ground_truth[i, j, k]) > threshold:
                    gt_val = ground_truth[i, j, k]
                    pred_val = predictions[i, j, k]
                    rel_error = np.abs(gt_val - pred_val) / np.abs(gt_val)
                    mse_error = (gt_val - pred_val) ** 2
                    gt_row.append(gt_val)
                    pred_row.append(pred_val)
                    rel_error_row.append(rel_error)
                    mse_error_row.append(mse_error)        
            
            # Prevent NaN by checking if lists are empty
            gt_row_.append(np.mean(gt_row) if gt_row else np.nan)
            pred_row_.append(np.mean(pred_row) if pred_row else np.nan)
            rel_error_row_.append(np.mean(rel_error_row) if rel_error_row else np.nan)
            mse_error_row_.append(np.mean(mse_error_row) if mse_error_row else np.nan)

        gt_values_matrix.append(gt_row_)
        pred_values_matrix.append(pred_row_)
        relative_errors_matrix.append(rel_error_row_)
        mse_error_matrix.append(mse_error_row_)

    return (
        np.array(gt_values_matrix), 
        np.array(pred_values_matrix), 
        np.array(relative_errors_matrix), 
        np.array(mse_error_matrix)
    )


def filter_by_thresh_2(ground_truth, predictions, threshold):
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()

    mask = np.abs(ground_truth) > threshold
    rel_error = np.zeros_like(ground_truth)
    mse_error = np.zeros_like(ground_truth)
    rel_error[mask] = np.abs(ground_truth[mask] - predictions[mask]) / np.abs(ground_truth[mask])
    mse_error[mask] = (ground_truth[mask] - predictions[mask]) ** 2

    ground_truth_masked = np.where(mask, ground_truth, np.nan)
    predictions_masked = np.where(mask, predictions, np.nan)
    rel_error_masked = np.where(mask, rel_error, np.nan)
    mse_error_masked = np.where(mask, mse_error, np.nan)

    # Average over axis=0 (same as manual i-loop)
    gt_values_matrix = np.nanmean(ground_truth_masked, axis=0).T
    pred_values_matrix = np.nanmean(predictions_masked, axis=0).T
    relative_errors_matrix = np.nanmean(rel_error_masked, axis=0).T
    mse_error_matrix = np.nanmean(mse_error_masked, axis=0).T

    return (
        gt_values_matrix, 
        pred_values_matrix, 
        relative_errors_matrix, 
        mse_error_matrix
    )

