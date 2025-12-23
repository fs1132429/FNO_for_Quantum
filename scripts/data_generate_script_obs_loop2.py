import os
import sys

# Define the relative path to the parent directory
relative_path = '..'

# Append the parent directory to the Python path
parent_directory = os.path.abspath(os.path.join(os.getcwd(), relative_path))
sys.path.append(parent_directory)
import torch
import time as t
import argparse
import cudaq
import gc
from functions.functions import *
from functions.functions_time_model import *
from function_pauli_Strings_new_inputs import *
from functions.functions_cuda import *

import subprocess

def log_gpu_usage_nvidia_smi():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,nounits,noheader"]
        ).decode("utf-8").strip()

        for i, line in enumerate(output.split("\n")):
            gpu_util, mem_used, mem_total = map(str.strip, line.split(','))
            print(f"GPU {i}: Usage {gpu_util}% | Memory {mem_used} MB / {mem_total} MB")
    except FileNotFoundError:
        print("nvidia-smi not found.")
    except Exception as e:
        print(f"Failed to get GPU usage: {e}")


# --------------------
# Parse arguments
# --------------------
parser = argparse.ArgumentParser(description="Generate time series data.")
parser.add_argument('--num_particles', type=int, required=True)
parser.add_argument('--num_states', type=int, required=True)
parser.add_argument('--j_min', type=float, required=True)
parser.add_argument('--j_max', type=float, required=True)
parser.add_argument('--jz', type=float, required=True)
parser.add_argument('--h', type=float, required=True)
parser.add_argument('--time', type=float, required=True)
parser.add_argument('--steps', type=int, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--start_state', type=int, required=True)         # NEW
parser.add_argument('--checkpoint_file', type=str, required=True)     # NEW
args = parser.parse_args()

# --------------------
# Setup
# --------------------
N = args.num_particles
num_states = args.num_states
Jz, h = args.jz, args.h
time, steps = args.time, args.steps
output_file_base = args.output_file
start_state = args.start_state
checkpoint_file = args.checkpoint_file

# Ensure checkpoint directory exists
os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

# Read last completed state from checkpoint
if os.path.exists(checkpoint_file):
    with open(checkpoint_file, "r") as f:
        last_completed = int(f.read().strip())
else:
    last_completed = 0  # Start from the beginning if no checkpoint exists

print(f"Resuming from state {last_completed} (start offset = {start_state})")


# --------------------
# Build Hamiltonian
# --------------------
hamiltonian = construct_hamiltonian_ising_cudaq(N, Jz, h)
pauli_strings = generate_pauli_strings_ising_all(N)
print(len(pauli_strings))
print("Hamiltonian generated")

# --------------------
# Main Loop
# --------------------
for i in range(last_completed, num_states):
    state_id = start_state + i  # Ensure unique IDs across GPUs
    start_time = t.time()
    print(f"Generating dataset for state {i + 1}/{num_states} (global state {state_id})")

    # Generate dataset
    dataset = create_trotter_discretized_time_data_set_with_expectations_sparse(
        N, 1, hamiltonian, time, steps, 100, pauli_strings
    )

    # Save immediately to disk
    output_file = f"{output_file_base}_state_{state_id}.pt"
    torch.save(dataset, output_file)
    print(f"Dataset for state {state_id} saved to {output_file}")

    # Free memory explicitly
    del dataset
    gc.collect()              # Python garbage collection
    torch.cuda.empty_cache()  # GPU memory

    end_time = t.time() - start_time
    print(f"Total time taken: {end_time:.2f} seconds")

    # Track GPU usage
    log_gpu_usage_nvidia_smi()

    # Update checkpoint after each iteration
    with open(checkpoint_file, "w") as f:
        f.write(str(i + 1))
