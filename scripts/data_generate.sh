#!/bin/bash
# Configuration script for running the data generation script

# Parameters for the data generation
export NUM_PARTICLES=12        # Number of particles
export NUM_STATES=6000         # Number of input states
export J_MIN=-2                # Minimum value for J
export J_MAX=2                 # Maximum value for J
export JX=0.3110300130979313   # Fixed interaction parameter Jx
export JY=-1.0706999080778115  # Fixed interaction parameter Jy
export JZ=-1.604499361441829   # Fixed interaction parameter Jz
export H=1.8111190054311277    # Fixed external field parameter
export TIME=0.00314              # Time for simulation
export STEPS=10000             # Number of steps for simulation
export OUTPUT_FILE="./dataset_12/dataset_12_trotter" # Output file name

# Run the Python script with the parameters
python3 data_generate_script_loop.py \
    --num_particles $NUM_PARTICLES \
    --num_states $NUM_STATES \
    --j_min $J_MIN \
    --j_max $J_MAX \
    --jx $JX \
    --jy $JY \
    --jz $JZ \
    --h $H \
    --time $TIME \
    --steps $STEPS \
    --output_file $OUTPUT_FILE
