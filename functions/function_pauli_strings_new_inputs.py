import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.data_transforms import DefaultDataProcessor

from neuralop.quantum.functions import *
from neuralop.quantum.functions_different_inputs import *
from neuralop.quantum.functions_time_model import *
from neuralop.quantum.functions_pauli_strings import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from loguru import logger

# Define spin operators for a single particle
sigma_x = (1/2) * torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
sigma_y = (1/2) * torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
sigma_z = (1/2) * torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
# Identity 
identity = torch.eye(2, dtype=torch.complex64)

def generate_specific_input_states_wavefunction(n, num_states,indices):
    # Create a tensor of zeros with the required shape
    amplitudes = torch.zeros(num_states, 2**n, dtype=torch.complex64)
    # Define arbitrary values between 0 and 1
    values = torch.rand(2, dtype=torch.float32)
    # Set the arbitrary values to the selected indices
    amplitudes[0, indices[0]] = values[0]
    amplitudes[0, indices[1]] = values[1]
    # Normalize the amplitudes
    norm = torch.sqrt(torch.sum(torch.square(amplitudes), dim=1))
    amplitudes /= norm.view(-1, 1)
    return amplitudes

def extend_S(n, S, pos):
    # Initialize the total S operator with identity
    S_total = torch.eye(1, dtype=torch.complex64)
    for i in range(n):
        # If i is in pos, use S; otherwise, use I
        S_total = torch.kron(S_total, S if i in pos else I)
    return S_total


def construct_hamiltonian_ising(n_particles, Jz=None, hx=None, interactions=None):    
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
    
    # Ensure Jz and hx are tensors of the correct length
    if Jz is None:
        Jz = -1 * torch.ones(len(interactions), dtype=torch.complex64)
    elif isinstance(Jz, (int, float)):
        Jz = torch.tensor([Jz] * len(interactions), dtype=torch.complex64)

    if hx is None:
        hx = torch.zeros(n_particles, dtype=torch.complex64)
    elif isinstance(hx, (int, float)):
        hx = torch.tensor([hx] * n_particles, dtype=torch.complex64)
    
    hamiltonian = torch.zeros((2**n_particles, 2**n_particles), dtype=torch.complex64)
    
    for idx, (i, j) in enumerate(interactions):
        # ZZ interaction term
        sz_i_sz_j = extend_S(n_particles, sz, [i]) @ extend_S(n_particles, sz, [j])
        hamiltonian += Jz[idx] * sz_i_sz_j
    
    for i in range(n_particles):
        h_i = hx[i]
        sx_i = extend_S(n_particles, sx, [i])  # single qubit X term
        hamiltonian += h_i * sx_i

    return hamiltonian

def generate_pauli_strings_ising(n_particles, interactions=None):
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
        
    # Define Pauli string set
    pauli_strings = set()
    
    for (i, j) in interactions:
        # Add ZZ interaction term
        pauli_strings.add(tuple(('Z' if k == i or k == j else 'I') for k in range(n_particles)))
    
    for i in range(n_particles):
        # Add X single-qubit term
        pauli_strings.add(tuple(('X' if k == i else 'I') for k in range(n_particles)))
    
    pauli_strings_list = sorted(pauli_strings)
    
    return pauli_strings_list


def generate_pauli_strings_ising_all(n_particles, interactions=None):
    # Define default interactions if None
    if interactions is None:
        interactions = [(i, (i + 1) % n_particles) for i in range(n_particles)]
        
    # Define Pauli string set
    pauli_strings = set()
    
    for (i, j) in interactions:
        # Add XX, YY, and ZZ interaction terms
        pauli_strings.add(tuple(('X' if k == i or k == j else 'I') for k in range(n_particles)))
        pauli_strings.add(tuple(('Y' if k == i or k == j else 'I') for k in range(n_particles)))
        pauli_strings.add(tuple(('Z' if k == i or k == j else 'I') for k in range(n_particles)))
    
    for i in range(n_particles):
        # Add single-qubit X, Y, and Z terms
        pauli_strings.add(tuple(('X' if k == i else 'I') for k in range(n_particles)))
        pauli_strings.add(tuple(('Y' if k == i else 'I') for k in range(n_particles)))
        pauli_strings.add(tuple(('Z' if k == i else 'I') for k in range(n_particles)))
    
    pauli_strings_list = sorted(pauli_strings)
    
    return pauli_strings_list

def create_time_data_set_pauli_string_ising(n,num_states,hamiltonian,time,steps):
    input_states= generate_random_input_states_wavefunction(n,num_states)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        logger.info(f"step:{i}")
        output_states= evolve_states(input_states,hamiltonian,time)
        output_tensor[:,:,i] = output_states
        input_states=output_states
        logger.info("executed")
    return output_tensor

def create_time_data_set_specific_pauli_string_ising(n,num_states,hamiltonian,time,steps,indices):
    input_states= generate_specific_input_states_wavefunction(n,num_states,indices)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        logger.info(f"step:{i}")
        output_states= evolve_states(input_states,hamiltonian,time)
        output_tensor[:,:,i] = output_states
        input_states=output_states
        logger.info("executed")
    return output_tensor


# Function to compute the expectation value of a Pauli string
def pauli_string_exp_val_ising(pauli_strings, x):
    num_states,_,input_t= x.shape
    expectation_val = torch.zeros((num_states, len(pauli_strings),input_t), dtype=torch.float32)
    for idx,ps in enumerate(pauli_strings):
        logger.info(f"on {idx} and {ps}")
        pauli_string=''.join(ps)
        operator= generate_pauli_matrix(pauli_string)
        for i in range(num_states):
            for j in range(input_t):
                psi= x[i,:,j]
                expectation_val[i,idx,j] = torch.vdot(psi, torch.matmul(operator, psi)).real
        logger.info("calculated all expectations")
    return expectation_val

#function to extract input_t and output_t steps from dataset and compute expectation values
def dataset_pauli_string_ising(x,n,input_T,output_T,start_index,interactions,exp_all_values=False):
    start_index=0
    # get num_frames from the input
    input = x[:, :, start_index : start_index + input_T]
    output = x[:, :, start_index + output_T: start_index + input_T + output_T]
    if exp_all_values:
        pauli_stings= generate_pauli_strings_ising_all(n,interactions)
    else:
        pauli_stings= generate_pauli_strings_ising(n,interactions)
    input_pauli_strings= pauli_string_exp_val_ising(pauli_stings,input)
    output_pauli_strings= pauli_string_exp_val_ising(pauli_stings,output)
    return input_pauli_strings,output_pauli_strings

def data_preprocess_pauli_string_ising(x,n,input_T,output_T,num_states,train_ratio,batch_size,start_index=0,interactions=None,exp_all_values=False):
    #start_index = np.random.randint(0, x.shape[-1] - T)
    if interactions is None:
        interactions = [(i, (i + 1) % n) for i in range(n)]    
    input,train_output= dataset_pauli_string_ising(x,n,input_T,output_T,start_index,interactions,exp_all_values)
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

def get_predictions_pauli(model, test_loader, rollout_steps, spatial_grid,output_t,input_t,data,dataset):
    all_predictions = []
    diff=input_t-output_t
    with torch.no_grad():
        for batch in test_loader:
            batch_predictions = []
            x, y = batch['x'].cuda(), batch['y'].cuda()  # Move data to the model's device
            # Initial prediction without autoregressive rollout
            input_encoder1= UnitGaussianNormalizer(dim=[0,1,2])
            input_encoder1.fit(x)
            output_encoder1= UnitGaussianNormalizer(dim=[0,1,2])
            output_encoder1.fit(y)
            x= input_encoder1.transform(x)
            predictions = model(x)
            predictions= output_encoder1.inverse_transform(predictions)
            batch_predictions.append(predictions[:,:,diff:]) #batch_size,2^n,T
            # Perform auto-regressive rollout
            outputt=output_t
            for i in range(rollout_steps - 1):
                outputt= output_t+ (i+1)*output_t
                predictions = torch.cat([y, spatial_grid], dim=1) 
                ground_truth=dataset[:,:,outputt:outputt+input_t]
                ground_truth=ground_truth.to('cuda')
                
                input_encoder= UnitGaussianNormalizer(dim=[0,1,2])
                input_encoder.fit(predictions)
                output_encoder= UnitGaussianNormalizer(dim=[0,1,2])
                output_encoder.fit(ground_truth)
                predictions=input_encoder.transform(predictions)
                predictions = model(predictions)
                predictions = output_encoder.inverse_transform(predictions)
                batch_predictions.append(predictions[:,:,diff:]) 
            # Append predictions and ground truth for this batch
            batch_prediction_tensor= torch.cat(batch_predictions,dim=-1)
            all_predictions.append(batch_prediction_tensor)
    # Concatenate predictions and ground truth across batches
    all_predictions = torch.cat(all_predictions, dim=0) 
    print(all_predictions.shape)
    return all_predictions

def get_ground_truth_pauli(dataset, rollout_steps,output_t,input_t):
    ground_truth= dataset[:,:,output_t:output_t+input_t*rollout_steps]
    return ground_truth

def get_ground_truth_overlap_pauli(dataset,rollout_steps,output_t,input_t):
    diff= input_t-output_t
    ground_truth_list=[]
    for i in range(rollout_steps):
        ground_truth=dataset[:,:,output_t+diff:output_t+input_t]
        ground_truth_list.append(ground_truth)
        output_t= output_t+input_t-diff
    tensor= torch.cat(ground_truth_list,dim=-1)
    print(tensor.shape)
    return tensor
        

def autoregressive_rollout_pauli(model, test_loader,dataset, rollout_steps, spatial_grid, output_t,input_t,data,overlap=False):
    all_predictions=get_predictions_pauli(model,test_loader,rollout_steps,spatial_grid,output_t,input_t,data,dataset)
    if overlap:
        all_ground_truths= get_ground_truth_overlap_pauli(dataset,rollout_steps,output_t,input_t)
    else:
        all_ground_truths= get_ground_truth_pauli(dataset,rollout_steps,output_t,input_t)
    return all_predictions, all_ground_truths




def load_dataset(dataset,input_t,output_t,start_index,num_states,train_ratio,batch_size):
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

    input_encoder= UnitGaussianNormalizer(dim=[0,1,2]) #compare with 0,1,2
    input_encoder.fit(train_input_final)

    output_encoder=UnitGaussianNormalizer(dim=[0,1,2])
    output_encoder.fit(train_output_final)
    
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


