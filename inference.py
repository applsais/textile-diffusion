#!/usr/bin/env python3
"""
Inference Script for TryOn Diffusion Model

This script provides inference capabilities for generating try-on images
using a trained TryOn diffusion model.
"""

import argparse
import os
import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np

from tryondiffusion import TryOnImagen, get_unet_by_name
from train import TrainingConfig


class TryOnInference:
    """Inference class for TryOn diffusion model"""
    
    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = "cuda"):
        """
        Initialize inference pipeline
        
        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to training configuration file
            device: Device to run inference on
        """
        self.device = device
        self.config = TrainingConfig(config_path) if config_path else TrainingConfig()
        
        # Setup model
        self.setup_model()
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        print(f"Inference pipeline initialized on {device}")
    
    def setup_model(self):
        """Setup the TryOn diffusion model"""
        # Create unets
        base_unet = get_unet_by_name("base", image_size=self.config.base_image_size)
        sr_unet = get_unet_by_name("sr", image_size=self.config.sr_image_size)
        
        # Create Imagen model
        self.imagen = TryOnImagen(
            unets=(base_unet, sr_unet),
            image_sizes=(self.config.base_image_size, self.config.sr_image_size),
            timesteps=self.config.timesteps,
            channels=self.config.channels,
            cond_drop_prob=0.0,  # No dropout during inference
        )
        
        self.imagen.to(self.device)
        self.imagen.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model weights from checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model' in checkpoint:
            # Training checkpoint format
            self.imagen.load_state_dict(checkpoint['model'])
        else:
            # Direct model state dict
            self.imagen.load_state_dict(checkpoint)
        
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def preprocess_image(self, image_path: str, target_size: tuple) -> torch.Tensor:
        """
        Preprocess input image
        
        Args:
            image_path: Path to input image
            target_size: Target size (height, width)
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Resize image
        transform = T.Compose([
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor.to(self.device)
    
    def preprocess_pose(self, pose_path: str) -> torch.Tensor:
        """
        Preprocess pose keypoints
        
        Args:
            pose_path: Path to pose keypoints file (numpy array or text file)
            
        Returns:
            Pose tensor
        """
        if pose_path.endswith('.npy'):
            pose = np.load(pose_path)
        else:
            # Assume text file with keypoints
            pose = np.loadtxt(pose_path)
        
        # Ensure pose has correct shape (18, 2) for 18 keypoints with (x, y) coordinates
        if pose.shape != (18, 2):
            print(f"Warning: Expected pose shape (18, 2), got {pose.shape}")
            # Pad or truncate as needed
            if pose.size < 36:
                pose = np.pad(pose.flatten(), (0, 36 - pose.size), 'constant')
            pose = pose.flatten()[:36].reshape(18, 2)
        
        pose_tensor = torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
        return pose_tensor.to(self.device)
    
    @torch.no_grad()
    def generate_tryon(
        self,
        ca_image_path: str,
        garment_image_path: str,
        person_pose_path: str,
        garment_pose_path: str,
        output_path: str,
        cond_scale: float = 2.0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
    ):
        """
        Generate try-on image
        
        Args:
            ca_image_path: Path to clothing-agnostic person image
            garment_image_path: Path to garment image
            person_pose_path: Path to person pose keypoints
            garment_pose_path: Path to garment pose keypoints
            output_path: Path to save generated image
            cond_scale: Conditioning scale for classifier-free guidance
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
        """
        # Determine image size based on highest resolution model
        target_size = self.config.sr_image_size
        
        # Preprocess inputs
        ca_image = self.preprocess_image(ca_image_path, target_size)
        garment_image = self.preprocess_image(garment_image_path, target_size)
        person_pose = self.preprocess_pose(person_pose_path)
        garment_pose = self.preprocess_pose(garment_pose_path)
        
        print("Generating try-on image...")
        
        # Generate image
        generated_images = self.imagen.sample(
            ca_images=ca_image,
            garment_images=garment_image,
            person_poses=person_pose,
            garment_poses=garment_pose,
            batch_size=1,
            cond_scale=cond_scale,
            return_all_unet_outputs=True,
            return_pil_images=True,
            use_tqdm=True,
        )
        
        # Save the highest resolution output
        if isinstance(generated_images, list) and len(generated_images) > 0:
            # Take the last (highest resolution) output
            final_images = generated_images[-1]
            if len(final_images) > 0:
                final_image = final_images[0]
                
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save image
                final_image.save(output_path)
                print(f"Generated try-on image saved to: {output_path}")
                
                return final_image
        
        raise RuntimeError("Failed to generate image")
    
    def batch_generate(
        self,
        input_dir: str,
        output_dir: str,
        cond_scale: float = 2.0,
    ):
        """
        Generate try-on images for a batch of inputs
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory to save generated images
            cond_scale: Conditioning scale
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all input sets (assuming naming convention)
        ca_images = sorted([f for f in os.listdir(input_dir) if f.startswith('ca_') and f.endswith('.jpg')])
        
        for ca_image_file in ca_images:
            base_name = ca_image_file.replace('ca_', '').replace('.jpg', '')
            
            # Find corresponding files
            garment_file = f"garment_{base_name}.jpg"
            person_pose_file = f"person_pose_{base_name}.npy"
            garment_pose_file = f"garment_pose_{base_name}.npy"
            
            # Check if all files exist
            input_files = [ca_image_file, garment_file, person_pose_file, garment_pose_file]
            if all(os.path.exists(os.path.join(input_dir, f)) for f in input_files):
                print(f"Processing {base_name}...")
                
                try:
                    output_path = os.path.join(output_dir, f"tryon_{base_name}.jpg")
                    
                    self.generate_tryon(
                        ca_image_path=os.path.join(input_dir, ca_image_file),
                        garment_image_path=os.path.join(input_dir, garment_file),
                        person_pose_path=os.path.join(input_dir, person_pose_file),
                        garment_pose_path=os.path.join(input_dir, garment_pose_file),
                        output_path=output_path,
                        cond_scale=cond_scale,
                    )
                except Exception as e:
                    print(f"Failed to process {base_name}: {e}")
            else:
                print(f"Missing input files for {base_name}, skipping...")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="TryOn Diffusion Inference")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, help="Path to training config file")
    parser.add_argument("--ca_image", type=str, help="Path to clothing-agnostic person image")
    parser.add_argument("--garment_image", type=str, help="Path to garment image")
    parser.add_argument("--person_pose", type=str, help="Path to person pose keypoints")
    parser.add_argument("--garment_pose", type=str, help="Path to garment pose keypoints")
    parser.add_argument("--output", type=str, help="Path to save generated image")
    parser.add_argument("--input_dir", type=str, help="Directory for batch processing")
    parser.add_argument("--output_dir", type=str, help="Output directory for batch processing")
    parser.add_argument("--cond_scale", type=float, default=2.0, help="Conditioning scale")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--batch", action="store_true", help="Run batch processing")
    
    return parser.parse_args()


def main():
    """Main inference function"""
    args = parse_args()
    
    # Initialize inference pipeline
    inference = TryOnInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )
    
    if args.batch:
        # Batch processing
        if not args.input_dir or not args.output_dir:
            raise ValueError("--input_dir and --output_dir required for batch processing")
        
        inference.batch_generate(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            cond_scale=args.cond_scale,
        )
    else:
        # Single image processing
        required_args = [args.ca_image, args.garment_image, args.person_pose, args.garment_pose, args.output]
        if not all(required_args):
            raise ValueError("All input arguments required for single image processing")
        
        inference.generate_tryon(
            ca_image_path=args.ca_image,
            garment_image_path=args.garment_image,
            person_pose_path=args.person_pose,
            garment_pose_path=args.garment_pose,
            output_path=args.output,
            cond_scale=args.cond_scale,
        )


if __name__ == "__main__":
    main()