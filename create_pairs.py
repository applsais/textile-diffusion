#!/usr/bin/env python3
"""
Create pair files for training and testing based on your dataset structure
"""

import os
from pathlib import Path
import random

def create_pair_files(dataset_dir="./dataset"):
    """Create train-pairs.txt and test-pairs.txt files"""
    dataset_path = Path(dataset_dir)
    
    # Check for train and test directories
    train_dir = dataset_path / "train"
    test_dir = dataset_path / "test"
    
    if not train_dir.exists() or not test_dir.exists():
        print(f"Error: {train_dir} or {test_dir} does not exist")
        return
    
    # Create training pairs
    train_image_dir = train_dir / "image"
    train_cloth_dir = train_dir / "cloth"
    
    if train_image_dir.exists() and train_cloth_dir.exists():
        create_pairs_for_split(train_image_dir, train_cloth_dir, "train-pairs.txt")
    
    # Create test pairs
    test_image_dir = test_dir / "image"
    test_cloth_dir = test_dir / "cloth"
    
    if test_image_dir.exists() and test_cloth_dir.exists():
        create_pairs_for_split(test_image_dir, test_cloth_dir, "test-pairs.txt")


def create_pairs_for_split(image_dir, cloth_dir, output_file):
    """Create pairs file for a specific split"""
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        image_files.extend(image_dir.glob(ext))
    
    # Get all cloth files
    cloth_files = []
    for ext in ['*.jpg', '*.png', '*.jpeg']:
        cloth_files.extend(cloth_dir.glob(ext))
    
    if not image_files or not cloth_files:
        print(f"Warning: No image files found in {image_dir} or {cloth_dir}")
        return
    
    # Create pairs
    pairs = []
    
    # Strategy 1: Exact name matching (person_001.jpg -> cloth_001.jpg)
    for img_file in image_files:
        cloth_file = cloth_dir / img_file.name
        if cloth_file.exists():
            pairs.append((img_file.name, img_file.name))
    
    # Strategy 2: If not enough exact matches, create random pairs
    if len(pairs) < min(100, len(image_files)):
        print(f"Only found {len(pairs)} exact matches, creating random pairs...")
        
        # Add some random pairs for variety
        random.seed(42)  # For reproducibility
        for img_file in random.sample(image_files, min(200, len(image_files))):
            cloth_file = random.choice(cloth_files)
            pair = (img_file.name, cloth_file.name)
            if pair not in pairs:
                pairs.append(pair)
    
    # Write pairs file
    with open(output_file, 'w') as f:
        for person_img, cloth_img in pairs:
            f.write(f"{person_img} {cloth_img}\n")
    
    print(f"Created {output_file} with {len(pairs)} pairs")


if __name__ == "__main__":
    create_pair_files()