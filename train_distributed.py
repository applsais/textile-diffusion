#!/usr/bin/env python3
"""
Distributed Training Script for TryOn Diffusion Model

This script provides distributed training capabilities using Accelerate
for multi-GPU and multi-node training scenarios.
"""

import argparse
import json
import logging
import os
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from train import TryOnTrainer, TrainingConfig


class DistributedTryOnTrainer(TryOnTrainer):
    """Extended trainer with distributed training support"""
    
    def __init__(self, config: TrainingConfig):
        # Initialize accelerator first
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision="fp16" if config.mixed_precision else "no",
            kwargs_handlers=[ddp_kwargs],
        )
        
        # Only setup logging on main process
        if self.accelerator.is_main_process:
            self.setup_logging_distributed(config)
        
        super().__init__(config)
        
        # Print distributed info
        if self.accelerator.is_main_process:
            logging.info(f"Distributed training with {self.accelerator.num_processes} processes")
            logging.info(f"Device: {self.accelerator.device}")
    
    def setup_logging_distributed(self, config):
        """Setup logging for distributed training"""
        os.makedirs(config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(config.log_dir, 'training_distributed.log')),
                logging.StreamHandler()
            ]
        )
    
    def train_epoch(self):
        """Override train epoch for distributed training"""
        # Ensure all processes are synchronized
        self.accelerator.wait_for_everyone()
        
        # Call parent training method
        epoch_metrics = super().train_epoch()
        
        # Synchronize at end of epoch
        self.accelerator.wait_for_everyone()
        
        return epoch_metrics
    
    def validate(self):
        """Override validation for distributed training"""
        # Only run validation on main process to avoid duplication
        if self.accelerator.is_main_process:
            return super().validate()
        else:
            return {'val_loss': 0.0}
    
    def generate_samples(self):
        """Override sample generation for distributed training"""
        # Only generate samples on main process
        if self.accelerator.is_main_process:
            super().generate_samples()


def parse_args():
    """Parse command line arguments for distributed training"""
    parser = argparse.ArgumentParser(description="Distributed Training for TryOn Diffusion")
    
    parser.add_argument("--config", type=str, default="config.json", help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--batch_size", type=int, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, help="Number of epochs")
    parser.add_argument("--train_unet", type=int, choices=[1, 2], help="Which UNet to train")
    
    return parser.parse_args()


def main():
    """Main function for distributed training"""
    args = parse_args()
    
    # Create configuration
    config = TrainingConfig(args.config)
    
    # Override config with command line arguments
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.train_unet:
        config.train_unet_number = args.train_unet
    
    # Create distributed trainer and start training
    trainer = DistributedTryOnTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()