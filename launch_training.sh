#!/bin/bash

# Launch Training Script for TryOn Diffusion Model
# This script provides easy commands to start training

set -e

# Default values
CONFIG_FILE="config.json"
TRAIN_UNET=1
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=100
MIXED_PRECISION=true
USE_EMA=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --unet)
            TRAIN_UNET="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --no-mixed-precision)
            MIXED_PRECISION=false
            shift
            ;;
        --no-ema)
            USE_EMA=false
            shift
            ;;
        --distributed)
            DISTRIBUTED=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config FILE         Configuration file (default: config.json)"
            echo "  --unet NUM           UNet to train: 1=base, 2=SR (default: 1)"
            echo "  --batch_size NUM     Batch size (default: 4)"
            echo "  --lr FLOAT           Learning rate (default: 1e-4)"
            echo "  --epochs NUM         Number of epochs (default: 100)"
            echo "  --no-mixed-precision Disable mixed precision training"
            echo "  --no-ema             Disable EMA"
            echo "  --distributed        Use distributed training"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if CUDA is available
if ! python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    echo "Warning: CUDA not available. Training will be slow on CPU."
fi

# Create directories
mkdir -p dataset outputs checkpoints logs

# Create pair files if they don't exist
if [ ! -f "train-pairs.txt" ] || [ ! -f "test-pairs.txt" ]; then
    echo "Creating pair files..."
    python create_pairs.py
fi

# Print configuration
echo "Starting TryOn Diffusion Training"
echo "================================="
echo "Configuration file: $CONFIG_FILE"
echo "Training UNet: $TRAIN_UNET"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Number of epochs: $NUM_EPOCHS"
echo "Mixed precision: $MIXED_PRECISION"
echo "EMA: $USE_EMA"
echo ""

# Build command
TRAIN_CMD="python train.py"
TRAIN_CMD="$TRAIN_CMD --config $CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --train_unet $TRAIN_UNET"
TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --learning_rate $LEARNING_RATE"
TRAIN_CMD="$TRAIN_CMD --num_epochs $NUM_EPOCHS"

if [ "$MIXED_PRECISION" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --mixed_precision"
fi

if [ "$USE_EMA" = false ]; then
    TRAIN_CMD="$TRAIN_CMD --no_ema"
fi

# Execute training
if [ "$DISTRIBUTED" = true ]; then
    echo "Starting distributed training..."
    echo "Command: accelerate launch train_distributed.py $TRAIN_CMD"
    accelerate launch train_distributed.py $(echo $TRAIN_CMD | sed 's/python train.py//')
else
    echo "Starting single GPU training..."
    echo "Command: $TRAIN_CMD"
    eval $TRAIN_CMD
fi

echo ""
echo "Training completed!"
echo "Check outputs/ for generated samples"
echo "Check checkpoints/ for saved models"
echo "Check logs/ for training logs"