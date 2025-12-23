import os
import sys

# Define the relative path to the parent directory
relative_path = '..'

# Append the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)

import argparse
from functions.functions import *
import torch
import matplotlib.pyplot as plt
import numpy as np
from functions.functions_time_model import *
from functions.functions_cuda import *
import time as t

# Parse arguments
parser = argparse.ArgumentParser(description="Generate time series data.")
parser.add_argument('--num_particles', type=int, required=True, help="Number of particles")
parser.add_argument('--num_states', type=int, required=True, help="Number of input states")
parser.add_argument('--j_min', type=float, required=True, help="Minimum value for J")
parser.add_argument('--j_max', type=float, required=True, help="Maximum value for J")
parser.add_argument('--jx', type=float, required=True, help="Fixed interaction parameter Jx")
parser.add_argument('--jy', type=float, required=True, help="Fixed interaction parameter Jy")
parser.add_argument('--jz', type=float, required=True, help="Fixed interaction parameter Jz")
parser.add_argument('--h', type=float, required=True, help="Fixed external field parameter")
parser.add_argument('--time', type=float, required=True, help="Time for simulation")
parser.add_argument('--steps', type=int, required=True, help="Number of steps for simulation")
parser.add_argument('--output_file', type=str, required=True, help="Output file name")
args = parser.parse_args()

# Constants from arguments
N = args.num_particles
num_states = args.num_states
J_min = args.j_min
J_max = args.j_max
Jx = args.jx
Jy = args.jy
Jz = args.jz
h = args.h
time = args.time
steps = args.steps
output_file = args.output_file

print("Interaction parameters:", Jx, Jy, Jz, h)


start_time= t.time()
hamiltonian = construct_hamiltonian_cudaq(N, Jx, Jy, Jz, h)
print("hamiltonian generated")
dataset = create_trotter_time_data_set_cudaq(N, num_states, hamiltonian, time, steps)
endTime = t.time() - start_time
print(f"time take to generate dataset{endTime}")
# Save dataset to a file
torch.save(dataset, output_file)

print(f"Dataset generated and saved to {output_file}")
