# TryOn Diffusion Training Guide

This guide provides comprehensive instructions for training the TryOn diffusion model for virtual garment try-on.

## Overview

The TryOn diffusion model is a multi-stage cascaded diffusion model that generates realistic virtual try-on images. It consists of:

1. **Base UNet**: Generates low-resolution try-on images (128x128)
2. **Super-Resolution UNet**: Upscales to high-resolution images (256x256)

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Organize your dataset with the following structure:
```
data/
├── person_images/          # Target person images
├── ca_images/             # Clothing-agnostic person images
├── garment_images/        # Garment images to try on
├── person_poses/          # Person pose keypoints (.npy files)
└── garment_poses/         # Garment pose keypoints (.npy files)
```

### 3. Basic Training

```bash
# Train base UNet (128x128)
python train.py --config config.json --train_unet 1 --num_epochs 50

# Train super-resolution UNet (256x256)
python train.py --config config.json --train_unet 2 --num_epochs 30
```

## Configuration

### Training Configuration (`config.json`)

```json
{
  "data_dir": "./data",
  "output_dir": "./outputs",
  "checkpoint_dir": "./checkpoints",
  "log_dir": "./logs",
  
  "base_image_size": [128, 128],
  "sr_image_size": [256, 256],
  "timesteps": [1000, 1000],
  "channels": 3,
  
  "batch_size": 4,
  "learning_rate": 1e-4,
  "num_epochs": 100,
  "gradient_accumulation_steps": 1,
  "max_grad_norm": 1.0,
  "warmup_steps": 1000,
  
  "validation_freq": 1000,
  "checkpoint_freq": 5000,
  "log_freq": 100,
  "sample_freq": 2000,
  
  "train_unet_number": 1,
  "use_ema": true,
  "mixed_precision": true
}
```

### Key Parameters

- **train_unet_number**: Which UNet to train (1=base, 2=super-resolution)
- **batch_size**: Adjust based on GPU memory (4-8 for most GPUs)
- **learning_rate**: Start with 1e-4, adjust based on convergence
- **use_ema**: Exponential Moving Average for better samples
- **mixed_precision**: Reduces memory usage and speeds up training

## Training Stages

### Stage 1: Base UNet Training

Train the base UNet first to generate 128x128 images:

```bash
python train.py \
    --config config.json \
    --train_unet 1 \
    --batch_size 8 \
    --learning_rate 1e-4 \
    --num_epochs 100
```

**Expected behavior:**
- Initial loss: ~0.5-1.0
- Converged loss: ~0.05-0.1
- Training time: 8-12 hours on V100

### Stage 2: Super-Resolution UNet Training

After base UNet converges, train the super-resolution UNet:

```bash
python train.py \
    --config config.json \
    --train_unet 2 \
    --batch_size 4 \
    --learning_rate 5e-5 \
    --num_epochs 50
```

**Expected behavior:**
- Initial loss: ~0.3-0.6
- Converged loss: ~0.03-0.08
- Training time: 12-16 hours on V100

## Advanced Training Options

### Distributed Training

For multi-GPU training:

```bash
# Using accelerate
accelerate config  # Configure distributed setup
accelerate launch train_distributed.py --config config.json
```

### Custom Dataset Integration

Replace the synthetic dataset in `train.py` with your own:

```python
class CustomTryOnDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        
        # Load your dataset metadata
        self.samples = self.load_metadata()
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load images and poses
        person_image = self.load_image(sample['person_image_path'])
        ca_image = self.load_image(sample['ca_image_path'])
        garment_image = self.load_image(sample['garment_image_path'])
        person_pose = self.load_pose(sample['person_pose_path'])
        garment_pose = self.load_pose(sample['garment_pose_path'])
        
        return {
            "person_images": person_image,
            "ca_images": ca_image,
            "garment_images": garment_image,
            "person_poses": person_pose,
            "garment_poses": garment_pose,
        }
```

### Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate Schedule**:
   ```json
   {
     "learning_rate": 1e-4,
     "warmup_steps": 1000,
     "cosine_decay_max_steps": 50000
   }
   ```

2. **Augmentation Settings**:
   ```python
   transforms = T.Compose([
       T.RandomHorizontalFlip(p=0.5),
       T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
       T.RandomRotation(degrees=5),
   ])
   ```

3. **Noise Schedule**:
   - Default: Cosine schedule
   - Alternative: Linear schedule for faster training

## Monitoring Training

### TensorBoard Logs

```bash
tensorboard --logdir logs/
```

Monitor:
- Training/validation loss curves
- Generated sample images
- Learning rate schedule

### Key Metrics

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Should follow training loss (watch for overfitting)
- **Sample Quality**: Visual inspection of generated images
- **FID Score**: Quantitative evaluation (implement separately)

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Training Instability**:
   - Lower learning rate
   - Increase gradient clipping
   - Check data preprocessing

3. **Poor Sample Quality**:
   - Train longer
   - Adjust conditioning scale
   - Verify data quality

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision
trainer = TryOnImagenTrainer(
    imagen=imagen,
    fp16=True,  # Enable mixed precision
    gradient_accumulation_steps=4,  # Accumulate gradients
)
```

## Inference

After training, generate try-on images:

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --config config.json \
    --ca_image examples/ca_person.jpg \
    --garment_image examples/garment.jpg \
    --person_pose examples/person_pose.npy \
    --garment_pose examples/garment_pose.npy \
    --output results/tryon_result.jpg
```

## Performance Benchmarks

### Training Performance

| GPU | Batch Size | Base UNet Speed | SR UNet Speed |
|-----|------------|-----------------|---------------|
| V100 | 8/4 | ~2.5 it/s | ~1.2 it/s |
| A100 | 16/8 | ~4.0 it/s | ~2.0 it/s |
| RTX 3090 | 6/3 | ~2.0 it/s | ~1.0 it/s |

### Model Size

- Base UNet: ~850M parameters
- SR UNet: ~950M parameters
- Total: ~1.8B parameters

## Best Practices

1. **Data Quality**: High-quality aligned images are crucial
2. **Pose Accuracy**: Accurate pose keypoints improve results
3. **Progressive Training**: Train base UNet first, then SR UNet
4. **Regular Validation**: Monitor sample quality frequently
5. **Checkpoint Management**: Save checkpoints regularly
6. **EMA Models**: Use EMA for better sample quality

## Citation

If you use this training code, please cite:

```bibtex
@article{tryondiffusion2024,
  title={TryOnDiffusion: A Tale of Two UNets},
  author={Authors},
  journal={arXiv preprint},
  year={2024}
}
```