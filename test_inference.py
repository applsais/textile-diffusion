#!/usr/bin/env python3
"""Test script to verify inference pipeline works"""

import os
import torch
from PIL import Image
import numpy as np
from simple_inference import simple_tryon_inference

def create_test_images():
    """Create simple test images for testing"""
    os.makedirs("test_images", exist_ok=True)
    
    # Create a simple person image (128x128 white with colored rectangle)
    person_img = Image.new('RGB', (128, 128), 'white')
    # Add a simple "person" shape
    pixels = np.array(person_img)
    pixels[30:90, 50:80] = [200, 180, 160]  # Skin tone rectangle
    person_img = Image.fromarray(pixels)
    person_img.save("test_images/test_person.jpg")
    
    # Create a simple dress image (red dress shape)
    dress_img = Image.new('RGB', (128, 128), 'white')
    pixels = np.array(dress_img)
    pixels[40:100, 40:90] = [200, 50, 50]  # Red dress shape
    dress_img = Image.fromarray(pixels)
    dress_img.save("test_images/test_dress.jpg")
    
    print("Test images created in test_images/")

def test_inference():
    """Test the inference pipeline"""
    
    # Create test images if they don't exist
    if not os.path.exists("test_images/test_person.jpg"):
        create_test_images()
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints/best_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        print("Please train the model first or provide a valid checkpoint path")
        return False
    
    try:
        # Run inference
        result = simple_tryon_inference(
            checkpoint_path=checkpoint_path,
            person_image_path="test_images/test_person.jpg",
            dress_image_path="test_images/test_dress.jpg",
            output_path="test_images/result.jpg",
            config_path="config.json",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("Inference test completed successfully!")
        print("Check test_images/result.jpg for the output")
        return True
        
    except Exception as e:
        print(f"Inference test failed: {e}")
        return False

if __name__ == "__main__":
    test_inference()