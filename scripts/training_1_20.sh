#!/bin/bash

# -----------------------------
# 0. Setup
# -----------------------------
mkdir -p logs
mkdir -p checkpoints

LOGFILE="logs/train_$(date +'%Y%m%d_%H%M%S').log"
echo "Job launched on $(hostname) at $(date)" | tee -a "$LOGFILE"
nvidia-smi 2>&1 | tee -a "$LOGFILE" || echo "No GPUs visible" | tee -a "$LOGFILE"

# -----------------------------
# 1. Activate your virtual environment
# -----------------------------
source ~/quantum_fno/bin/activate

# -----------------------------
# 2. Set training parameters
# -----------------------------
N=20
NUM_STATES=25000
TRAIN_RATIO=0.9
BATCH_SIZE=32
HIDDEN_CHANNELS=2048
INPUT_T=15
OUTPUT_T=10
PROJ_LIFT=4096
EPOCHS=1500
LR=1e-4
GAMMA=0.99
WEIGHT_DECAY=1e-3
STEP_SIZE=200
ARCH=3
FACTORIZATION="tucker"
RANK=0.5
LENGTH=120
MODES=8
SCHEDULER_TYPE="ReduceLR"
JZ=0.2
H=0.9

# Derived values
TRAIN_STATES=$(python3 -c "print(int($NUM_STATES * $TRAIN_RATIO))")
VAL_STATES=$(python3 -c "print($NUM_STATES - int($NUM_STATES * $TRAIN_RATIO))")

echo "=============================" | tee -a "$LOGFILE"
echo "Training Configuration:" | tee -a "$LOGFILE"
echo "  NUM_STATES: $NUM_STATES" | tee -a "$LOGFILE"
echo "  TRAIN_RATIO: $TRAIN_RATIO" | tee -a "$LOGFILE"
echo "  TRAIN_STATES: $TRAIN_STATES" | tee -a "$LOGFILE"
echo "  VAL_STATES: $VAL_STATES" | tee -a "$LOGFILE"
echo "  EPOCHS: $EPOCHS | LR: $LR | Scheduler: $SCHEDULER_TYPE" | tee -a "$LOGFILE"
echo "  Hidden Channels: $HIDDEN_CHANNELS | Modes: $MODES | Length: $LENGTH" | tee -a "$LOGFILE"
echo "=============================" | tee -a "$LOGFILE"

# -----------------------------
# 3. Collect datasets from checkpoint folders
# -----------------------------
DATASET_DIR="./dataset_from_checkpoints"

if [ -d "$DATASET_DIR" ] && [ "$(ls -A $DATASET_DIR)" ]; then
    echo "Dataset directory $DATASET_DIR already exists and is not empty. Skipping dataset collection." | tee -a "$LOGFILE"
else
    echo "Collecting datasets into $DATASET_DIR..." | tee -a "$LOGFILE"
    mkdir -p "$DATASET_DIR"
    for gpu_dir in ./dataset/gpu*; do
        if [ -d "$gpu_dir" ]; then
            echo "Collecting data from $gpu_dir" | tee -a "$LOGFILE"
            rsync -a --ignore-existing "$gpu_dir"/ "$DATASET_DIR"/
        fi
    done
fi

DATA_PATH=$DATASET_DIR

# -----------------------------
# 4. Launch training
# -----------------------------
echo "Starting training..." | tee -a "$LOGFILE"

python -u training_script.py \
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
    --h $H 2>&1 | tee -a "$LOGFILE"

echo "Training finished at $(date)" | tee -a "$LOGFILE"
