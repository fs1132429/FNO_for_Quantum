import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from functions.functions import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def generate_low_energy_states(n,num_states,populated_states):
    # Random real part between -1 and 1
    real_part = torch.rand(num_states, populated_states) * 2 - 1
    # Random real part between -1 and 1
    imag_part = torch.rand(num_states, populated_states) * 2 - 1  
    amplitudes = real_part + 1j * imag_part
    # Calculate normalization factor
    norm = torch.sqrt(torch.sum(torch.square(real_part), dim=1) + torch.sum(torch.square(imag_part), dim=1))
    # Normalize the amplitudes 
    amplitudes /= norm.view(-1, 1)
    # Initialize coefficients tensor
    coefficients= torch.zeros(num_states,2**n,dtype=torch.complex64)
    # Append the amplitudes in the coefficients tensor
    coefficients[:,0:populated_states]=amplitudes
    return coefficients

def generate_states_sorted_by_population(n,num_states):
    # Random real part between -1 and 1
    real_part = torch.rand(num_states, 2**n) * 2 - 1  
    # Random real part between -1 and 1
    imag_part = torch.rand(num_states, 2**n) * 2 - 1  
    amplitudes = real_part + 1j * imag_part
    # Calculate normalization factor
    norm = torch.sqrt(torch.sum(torch.square(real_part), dim=1) + torch.sum(torch.square(imag_part), dim=1))
    # Normalize the amplitudes 
    amplitudes /= norm.view(-1, 1)
    # Compute the absolute square
    abs_square = torch.abs(amplitudes) ** 2
    # Sort the resulting tensor and get the indices
    sorted_abs_square, sorted_indices = torch.sort(abs_square, dim=1, descending=True)
    # Sort the coefficients based on indices
    sorted_amplitudes = torch.gather(amplitudes, 1, sorted_indices)
    return sorted_amplitudes

#Architecture - 1
def create_dataset_low_energy_states(n, num_states, train_ratio, batch_size,populated_states, times=3.14, Jx=None, Jy=None, Jz=None, h=None, interactions=None): #times=torch.linspace(0, 3.14, 700)
    # Generate input states
    ordered_input_states = generate_low_energy_states(n, num_states,populated_states)
    # Construct Hamiltonian
    hamiltonian = construct_hamiltonian(n, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)
    #ordered indices based on energy level
    sorted_indices,ordered_energy= ordered_indices(n,hamiltonian)
    #order hamiltonian
    ordered_hamiltonian= order_hamiltonian(hamiltonian, sorted_indices)
    #ordered output
    ordered_output_states= evolve_states(ordered_input_states,ordered_hamiltonian,times)
    # Generate energy grid
    energy_grid = torch.linspace(0, 1, 2**n).unsqueeze(0).expand(num_states, -1)
    #energy_grid= energy_values.unsqueeze(0).expand(num_states, -1)
    # Concatenate spatial grid with input states
    train_input = torch.cat([ordered_input_states.unsqueeze(-1), energy_grid.unsqueeze(-1)], dim=-1).transpose(1, 2)  # train input for FNO needs to be of the form [num_states, in_channels, input_states_wavefunction]
    train_output = ordered_output_states.unsqueeze(-1).transpose(1, 2)  # train output to FNO needs to be of the form [num_states, out_channels, input_states_wavefunction]
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

def create_dataset_sorted_by_population(n, num_states, train_ratio, batch_size, times=3.14, Jx=None, Jy=None, Jz=None, h=None, interactions=None): #times=torch.linspace(0, 3.14, 700)
    # Generate input states
    ordered_input_states = generate_states_sorted_by_population(n, num_states)
    # Construct Hamiltonian
    hamiltonian = construct_hamiltonian(n, Jx=Jx, Jy=Jy, Jz=Jz, h=h, interactions=interactions)
    #ordered indices based on energy level
    sorted_indices,ordered_energy= ordered_indices(n,hamiltonian)
    #order hamiltonian
    ordered_hamiltonian= order_hamiltonian(hamiltonian, sorted_indices)
    #ordered output
    ordered_output_states= evolve_states(ordered_input_states,ordered_hamiltonian,times)
    # Generate energy grid
    energy_grid = torch.linspace(0, 1, 2**n).unsqueeze(0).expand(num_states, -1)
    #energy_grid= energy_values.unsqueeze(0).expand(num_states, -1)
    # Concatenate spatial grid with input states
    train_input = torch.cat([ordered_input_states.unsqueeze(-1), energy_grid.unsqueeze(-1)], dim=-1).transpose(1, 2)  # train input for FNO needs to be of the form [num_states, in_channels, input_states_wavefunction]
    train_output = ordered_output_states.unsqueeze(-1).transpose(1, 2)  # train output to FNO needs to be of the form [num_states, out_channels, input_states_wavefunction]
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

#Architecture - 2
def create_time_data_set_low_energy_states(n,num_states,populated_states,hamiltonian,time,steps):
    input_states= generate_low_energy_states(n,num_states,populated_states)
    #times=torch.linspace(0,final_T,time_step)
    indices,_=ordered_indices(n,hamiltonian)
    ordered_hamiltonain= order_hamiltonian(hamiltonian,indices)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        output_states= evolve_states(input_states,ordered_hamiltonain,time)
        output_tensor[:,:,i] = output_states
        input_states=output_states
    return output_tensor

def create_time_data_set_sorted_by_population(n,num_states,hamiltonian,time,steps):
    input_states= generate_states_sorted_by_population(n,num_states)
    #times=torch.linspace(0,final_T,time_step)
    indices,_=ordered_indices(n,hamiltonian)
    ordered_hamiltonain= order_hamiltonian(hamiltonian,indices)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        output_states= evolve_states(input_states,ordered_hamiltonain,time)
        output_tensor[:,:,i] = output_states
        input_states=output_states
    return output_tensor
