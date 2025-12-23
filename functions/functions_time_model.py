import numpy as np
import matplotlib.pyplot as plt
import torch

from neuralop.layers.embeddings import PositionalEmbedding

from functions.functions import *
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def create_time_data_set(n,num_states,hamiltonian,time,steps):
    input_states= generate_random_input_states_wavefunction(n,num_states)
    #times=torch.linspace(0,final_T,time_step)
    output_tensor= torch.zeros(num_states,2**n,steps+1, dtype=torch.complex64) #first would be random input + steps of time
    for i in range(steps+1):
        output_states= evolve_states(input_states,hamiltonian,time)
        output_tensor[:,:,i] = output_states
        input_states=output_states
    return output_tensor


def data_preprocess(x,input_T,output_T,num_states,train_ratio,batch_size):
    T=input_T+output_T
    #start_index = np.random.randint(0, x.shape[-1] - T)
    start_index=0
    # get num_frames from the input
    input = x[:, :, start_index : start_index + input_T]
    train_output = x[:, :, start_index + output_T: start_index + input_T + output_T]
    pos_embedding = PositionalEmbedding(2)
    timesteps = torch.linspace(start_index, start_index+input_T,input_T)
    positional_embeddings = pos_embedding(timesteps)
    pos=positional_embeddings.T.repeat(num_states, 1, 1)
    #pos= torch.linspace(0,1,input_T).unsqueeze(0).repeat(num_states, 1, 1)
    train_input = torch.cat([input,pos],dim=1)
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


def plot_comparison_with_error2(predictions, ground_truth): #in torch
    # Calculate error
    error_real = torch.abs(predictions.real - ground_truth.real)
    error_imag = torch.abs(predictions.imag - ground_truth.imag)

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

def fidelity_func(predictions,ground_truth): #in torch
    num_states,N,T=predictions.shape
    fidelity = torch.zeros((num_states, T), dtype=torch.float32)
    for i in range(num_states):
        for j in range(T):
            predictions_input=predictions[i,:,j]
            ground_truth_input= ground_truth[i,:,j]
            inner_product=torch.sum(torch.conj(predictions_input) * ground_truth_input)
            fidelity[i,j]= torch.abs(inner_product) ** 2
    average_fidelity = torch.mean(fidelity, dim=0) 
    fidelity_std = torch.std(fidelity, dim=0)  # Standard deviation along the same dimension as average_fidelity
    
    return fidelity, average_fidelity, fidelity_std
   

def get_predictions2(model, test_loader, rollout_steps, spatial_grid,output_t,input_t):
    all_predictions = []
    diff=input_t-output_t
    with torch.no_grad():
        for batch in test_loader:
            batch_predictions = []
            x, _ = batch['x'].cuda(), batch['y'].cuda()  # Move data to the model's device
            # Initial prediction without autoregressive rollout
            predictions = model(x)
            batch_predictions.append(predictions[:,:,diff:]) #batch_size,2^n,T
            y=predictions
            # Perform auto-regressive rollout
            for _ in range(rollout_steps-1 ):
                predictions = torch.cat([y, spatial_grid], dim=1) 
                predictions = model(predictions)
                batch_predictions.append(predictions[:,:,diff:]) 
                y=predictions
            # Append predictions and ground truth for this batch
            batch_prediction_tensor= torch.cat(batch_predictions,dim=-1)
            all_predictions.append(batch_prediction_tensor)
    # Concatenate predictions and ground truth across batches
    all_predictions = torch.cat(all_predictions, dim=0) 
    print(all_predictions.shape)
    return all_predictions

def get_ground_truth2(dataset, rollout_steps,output_t,input_t):
    output_t=output_t*2
    ground_truth= dataset[:,:,output_t:output_t+input_t*rollout_steps]
    return ground_truth

def get_ground_truth_overlap(dataset,rollout_steps,output_t,input_t):
    diff= input_t-output_t
    ground_truth_list=[]
    #output_t=2*output_t
    for i in range(rollout_steps):
        ground_truth=dataset[:,:,output_t+diff:output_t+input_t]
        ground_truth_list.append(ground_truth)
        output_t= output_t+input_t-diff
    tensor= torch.cat(ground_truth_list,dim=-1)
    print(tensor.shape)
    return tensor
        

def autoregressive_rollout2(model, test_loader,dataset, rollout_steps, spatial_grid, output_t,input_t,overlap=False):
    all_predictions=get_predictions2(model,test_loader,rollout_steps,spatial_grid,output_t,input_t)
    if overlap:
        all_ground_truths= get_ground_truth_overlap(dataset,rollout_steps,output_t,input_t)
    else:
        all_ground_truths= get_ground_truth2(dataset,rollout_steps,output_t,input_t)
    return all_predictions, all_ground_truths



""""
class your_data_set(torch.utils.data.Dataset):
    def __init__(x):
        
        self.x = x
    def __len__(self) -> int:
        return self.x.shape[0]
    def __getitem__(self, idx: int):
        # input
        start_index = np.random.randint(0, x.shape[-1] - T)
        # get num_frames from the input
        in =self.x[idx, :, start_index : start_index + T]
        out = self,x[idx, :, start_index + how_many_steps_after_you_wanna_predict: start_index + T + how_many_steps_after_you_wanna_predict]
    # add any normalization or std you want here
    return in, out (edited) 
"""