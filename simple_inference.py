#!/usr/bin/env python3
"""
Simplified Inference Script for Dress Visualization

This script provides a quick way to visualize how a dress looks on a person
using your trained TryOn diffusion model.
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import argparse
import os

from tryondiffusion import TryOnImagen, get_unet_by_name
from train import TrainingConfig


def simple_tryon_inference(
    checkpoint_path: str,
    person_image_path: str,
    dress_image_path: str,
    output_path: str,
    config_path: str = "config.json",
    device: str = "cuda"
):
    """
    Simple function to generate try-on image with minimal inputs
    
    Args:
        checkpoint_path: Path to trained model checkpoint
        person_image_path: Path to person image
        dress_image_path: Path to dress image
        output_path: Path to save result
        config_path: Path to config file
        device: Device to use
    """
    
    # Load config
    config = TrainingConfig(config_path)
    
    # Setup model
    base_unet = get_unet_by_name("base", image_size=config.base_image_size)
    sr_unet = get_unet_by_name("sr", image_size=config.sr_image_size)
    
    imagen = TryOnImagen(
        unets=(base_unet, sr_unet),
        image_sizes=(config.base_image_size, config.sr_image_size),
        timesteps=config.timesteps,
        channels=config.channels,
        cond_drop_prob=0.0,
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        imagen.load_state_dict(checkpoint['model'])
    else:
        imagen.load_state_dict(checkpoint)
    
    imagen.to(device)
    imagen.eval()
    
    # Image preprocessing
    target_size = config.sr_image_size
    transform = T.Compose([
        T.Resize(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load and preprocess images
    person_img = Image.open(person_image_path).convert('RGB')
    dress_img = Image.open(dress_image_path).convert('RGB')
    
    person_tensor = transform(person_img).unsqueeze(0).to(device)
    dress_tensor = transform(dress_img).unsqueeze(0).to(device)
    
    # Create clothing-agnostic image (simplified - just use original for now)
    ca_tensor = person_tensor.clone()
    
    # Create dummy poses (you can replace with actual pose detection)
    dummy_pose = torch.zeros(1, 18, 2).to(device)
    
    print("Generating try-on image...")
    
    # Generate
    with torch.no_grad():
        generated_images = imagen.sample(
            ca_images=ca_tensor,
            garment_images=dress_tensor,
            person_poses=dummy_pose,
            garment_poses=dummy_pose,
            batch_size=1,
            cond_scale=2.0,
            return_pil_images=True,
            use_tqdm=True,
        )
    
    # Save result
    if isinstance(generated_images, list) and len(generated_images) > 0:
        final_images = generated_images[-1]  # Take highest resolution
        if len(final_images) > 0:
            result_image = final_images[0]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            result_image.save(output_path)
            print(f"Result saved to: {output_path}")
            return result_image
    
    raise RuntimeError("Failed to generate image")


def main():
    parser = argparse.ArgumentParser(description="Simple TryOn Inference")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--person", required=True, help="Person image path")
    parser.add_argument("--dress", required=True, help="Dress image path")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--device", default="cuda", help="Device")
    
    args = parser.parse_args()
    
    simple_tryon_inference(
        checkpoint_path=args.checkpoint,
        person_image_path=args.person,
        dress_image_path=args.dress,
        output_path=args.output,
        config_path=args.config,
        device=args.device
    )


if __name__ == "__main__":
    main()