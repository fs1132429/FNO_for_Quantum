#!/bin/bash

# -----------------------------
# 0. Setup
# -----------------------------
mkdir -p logs
mkdir -p checkpoints

# -----------------------------
# 1. Activate virtual environment
# -----------------------------
source ~/quantum_fno/bin/activate

# -----------------------------
# 2. Define hyperparameter sets
# -----------------------------
# Format: NUM_STATES MODES HIDDEN_CHANNELS PROJ_LIFT BATCH_SIZE LR EPOCHS RANK GAMMA WEIGHT_DECAY
declare -a PARAMS_SET1=(33000 8 1024 4096 32 1e-4 1700 0.8 0.99 1e-3)
declare -a PARAMS_SET2=(33000 8 2048 4096 64 1e-4 1700 0.75 0.99 1e-3)
declare -a PARAMS_SET3=(33000 8 2048 4096 64 5e-5 1700 0.75 0.99 1e-3)
declare -a PARAMS_SET4=(33000 8 1024 2048 32 1e-4 1700 0.8 0.99 1e-3)
declare -a PARAMS_SET5=(33000 7 2048 4096 64 5e-5 1700 0.8 0.99 1e-3)
declare -a PARAMS_SET6=(33000 7 2048 4096 64 5e-5 1700 0.8 0.99 1e-3)
declare -a PARAMS_SET7=(33000 9 2048 4096 64 5e-5 1700 0.8 0.99 1e-3)


PARAM_SETS=(PARAMS_SET1 PARAMS_SET2 PARAMS_SET3 PARAMS_SET4 PARAMS_SET5 PARAMS_SET6 PARAMS_SET7)

# Common parameters
N=20
TRAIN_RATIO=0.9
INPUT_T=15
OUTPUT_T=10
STEP_SIZE=200
ARCH=3
FACTORIZATION="tucker"
LENGTH=120
SCHEDULER_TYPE="ReduceLR"
JZ=0.2
H=0.9
DATA_PATH="./dataset_from_checkpoints"

# -----------------------------
# 3. Get number of GPUs
# -----------------------------
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n1)
echo "Detected $GPU_COUNT GPUs. Sessions will be assigned in round-robin."

# -----------------------------
# 4. Launch tmux sessions
# -----------------------------
for i in ${!PARAM_SETS[@]}; do
    eval "SET=(\"\${${PARAM_SETS[i]}[@]}\")"

    NUM_STATES=${SET[0]}
    MODES=${SET[1]}
    HIDDEN_CHANNELS=${SET[2]}
    PROJ_LIFT=${SET[3]}
    BATCH_SIZE=${SET[4]}
    LR=${SET[5]}
    EPOCHS=${SET[6]}
    RANK=${SET[7]}
    GAMMA=${SET[8]}
    WEIGHT_DECAY=${SET[9]}

    # Round-robin GPU assignment
    GPU_ID=$(( i % GPU_COUNT ))

    SESSION_NAME="train_hp$((i+1))"
    LOGFILE="logs/train_eval_hp$((i+1))_$(date +'%Y%m%d_%H%M%S').log"

    echo "Launching tmux session $SESSION_NAME on GPU $GPU_ID → STATES=$NUM_STATES, MODES=$MODES, HIDDEN=$HIDDEN_CHANNELS, PROJ_LIFT=$PROJ_LIFT, BATCH_SIZE=$BATCH_SIZE, LR=$LR, EPOCHS=$EPOCHS, RANK=$RANK, GAMMA=$GAMMA, WEIGHT_DECAY=$WEIGHT_DECAY"

    tmux new-session -d -s $SESSION_NAME \
        "export CUDA_VISIBLE_DEVICES=$GPU_ID; \
        python -u train_and_eval.py \
            --N $N \
            --num_states $NUM_STATES \
            --train_ratio $TRAIN_RATIO \
            --batch_size $BATCH_SIZE \
            --hidden_channels $HIDDEN_CHANNELS \
            --input_T $INPUT_T \
            --output_T $OUTPUT_T \
            --proj_lift_channel $PROJ_LIFT \
            --epochs $EPOCHS \
            --lr $LR \
            --gamma $GAMMA \
            --weight_decay $WEIGHT_DECAY \
            --data_path $DATA_PATH \
            --step_size $STEP_SIZE \
            --arch $ARCH \
            --factorization $FACTORIZATION \
            --rank $RANK \
            --length $LENGTH \
            --modes $MODES \
            --scheduler_type $SCHEDULER_TYPE \
            --jz $JZ \
            --h $H | tee -a $LOGFILE"

    tmux pipe-pane -o -t $SESSION_NAME:0 "cat >> logs/${SESSION_NAME}_pane.log"

    sleep 5
done

echo "✅ All tmux sessions launched. Use 'tmux ls' to list and 'tmux attach -t train_hpX' to inspect."
