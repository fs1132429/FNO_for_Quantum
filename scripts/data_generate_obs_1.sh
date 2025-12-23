#!/bin/bash
# Use mem_monitor.sh to avoid memory blowup

echo "Job launched on $(hostname) at $(date)"

nvidia-smi || echo "No GPUs visible"

# Activate venv
source ~/quantum_fno/bin/activate

# Parameters
NUM_PARTICLES=20
TOTAL_STATES=60000      # NUM_GPUS × STATES_PER_GPU
STATES_PER_GPU=10000
NUM_GPUS=7
J_MIN=-2
J_MAX=2
JZ=0.2
H=0.9
TIME=0.00314
STEPS=10000

# Output dirs
BASE_OUTPUT="./dataset"
BASE_CHECKPOINT="./checkpoints"
BASE_LOG="./logs"
mkdir -p "$BASE_OUTPUT" "$BASE_CHECKPOINT" "$BASE_LOG"

# Function
run_on_gpu() {
    GPU_ID=$1
    START_STATE=$((GPU_ID * STATES_PER_GPU))
    END_STATE=$((START_STATE + STATES_PER_GPU))

    OUTPUT_DIR="${BASE_OUTPUT}/gpu_${GPU_ID}"
    CHECKPOINT_FILE="${BASE_CHECKPOINT}/checkpoint_obs_${JZ}_${H}_gpu${GPU_ID}.txt"
    LOG_FILE="${BASE_LOG}/gpu_${GPU_ID}.log"
    
    mkdir -p "$OUTPUT_DIR"

    echo "Launching GPU $GPU_ID for states $START_STATE to $END_STATE" | tee -a "$LOG_FILE"

    while true; do
        {
            echo "[$(date)] Starting job on GPU $GPU_ID" 
            CUDA_VISIBLE_DEVICES=$GPU_ID python -u data_generate_script_obs_loop2.py \
                --num_particles $NUM_PARTICLES \
                --num_states $STATES_PER_GPU \
                --j_min $J_MIN \
                --j_max $J_MAX \
                --jz $JZ \
                --h $H \
                --time $TIME \
                --steps $STEPS \
                --output_file "${OUTPUT_DIR}/dataset_${NUM_PARTICLES}_trotter" \
                --start_state $START_STATE \
                --checkpoint_file $CHECKPOINT_FILE
            echo "[$(date)] Job finished for GPU $GPU_ID"
        } >> "$LOG_FILE" 2>&1   # Log both stdout and stderr

        # Check if checkpoint indicates completion
        if [[ -f "$CHECKPOINT_FILE" ]]; then
            LAST_COMPLETED=$(cat "$CHECKPOINT_FILE")
            if [[ $LAST_COMPLETED -ge $STATES_PER_GPU ]]; then
                echo "GPU $GPU_ID finished all $STATES_PER_GPU states. Exiting loop." | tee -a "$LOG_FILE"
                break
            fi
        fi

        echo "GPU $GPU_ID not finished yet (or crashed). Restarting in 10s..." | tee -a "$LOG_FILE"
        sleep 10
    done
}

# Launch each GPU process in the background with staggered starts
for GPU_ID in $(seq 0 $((NUM_GPUS-1))); do
    run_on_gpu $GPU_ID &
    sleep 60
done

wait
echo "All GPUs finished at $(date)"
