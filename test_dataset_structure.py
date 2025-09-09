#!/usr/bin/env python3
"""
Test script to verify dataset structure and pair file loading
"""

import os
from pathlib import Path
from train import TryOnDataset

def test_dataset_structure(data_dir="./data"):
    """Test the dataset structure and pair file loading"""
    
    print(f"Testing dataset structure in: {data_dir}")
    print("=" * 50)
    
    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        return False
    
    # Check train/test structure
    train_path = data_path / "train"
    test_path = data_path / "test"
    
    print(f"Train directory exists: {train_path.exists()}")
    print(f"Test directory exists: {test_path.exists()}")
    
    if train_path.exists():
        print(f"Train subdirs: {[d.name for d in train_path.iterdir() if d.is_dir()]}")
        
        # Check for required subdirectories
        train_image = train_path / "image"
        train_cloth = train_path / "cloth"
        
        print(f"  - train/image exists: {train_image.exists()}")
        print(f"  - train/cloth exists: {train_cloth.exists()}")
        
        if train_image.exists():
            image_files = list(train_image.glob("*.jpg")) + list(train_image.glob("*.png"))
            print(f"  - Found {len(image_files)} person images")
            if image_files:
                print(f"    Sample: {image_files[0].name}")
        
        if train_cloth.exists():
            cloth_files = list(train_cloth.glob("*.jpg")) + list(train_cloth.glob("*.png"))
            print(f"  - Found {len(cloth_files)} garment images")
            if cloth_files:
                print(f"    Sample: {cloth_files[0].name}")
    
    if test_path.exists():
        print(f"Test subdirs: {[d.name for d in test_path.iterdir() if d.is_dir()]}")
    
    print("\n" + "=" * 50)
    print("Testing pair file loading...")
    
    # Check pair files
    train_pairs = Path("train-pairs.txt")
    test_pairs = Path("test-pairs.txt")
    
    print(f"train-pairs.txt exists: {train_pairs.exists()}")
    print(f"test-pairs.txt exists: {test_pairs.exists()}")
    
    if train_pairs.exists():
        with open(train_pairs, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        print(f"Found {len(pairs)} training pairs")
        if pairs:
            print(f"Sample pair: {pairs[0]}")
    
    if test_pairs.exists():
        with open(test_pairs, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        print(f"Found {len(pairs)} test pairs")
        if pairs:
            print(f"Sample pair: {pairs[0]}")
    
    print("\n" + "=" * 50)
    print("Testing dataset class...")
    
    try:
        # Test train dataset
        train_dataset = TryOnDataset(
            data_dir=data_dir,
            image_size=(128, 128),
            use_real_data=True,
            split="train",
            download_dataset=False
        )
        
        print(f"✅ Train dataset loaded: {len(train_dataset)} samples")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print("Sample keys:", list(sample.keys()))
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {type(value)}")
        
        # Test validation dataset
        val_dataset = TryOnDataset(
            data_dir=data_dir,
            image_size=(128, 128),
            use_real_data=True,
            split="test",
            download_dataset=False
        )
        
        print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = test_dataset_structure()
    
    if success:
        print("\n🎉 Dataset structure test PASSED!")
        print("\nYou can now run training with:")
        print("python train.py --data_dir ./data --batch_size 2 --num_epochs 5")
    else:
        print("\n💥 Dataset structure test FAILED!")
        print("Please check your dataset structure and pair files.")

if __name__ == "__main__":
    main()