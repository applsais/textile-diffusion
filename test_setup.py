#!/usr/bin/env python3
"""
Test script to verify the complete setup is working
"""

import os
import torch
from pathlib import Path
import json

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    try:
        import torch
        print(f"[OK] PyTorch {torch.__version__}")
        print(f"[OK] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA version: {torch.version.cuda}")
            print(f"[OK] GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("[ERROR] PyTorch not installed")
        return False
    
    try:
        import torchvision
        print(f"[OK] TorchVision {torchvision.__version__}")
    except ImportError:
        print("[ERROR] TorchVision not installed")
        return False
    
    try:
        from PIL import Image
        print("[OK] PIL/Pillow available")
    except ImportError:
        print("[ERROR] PIL/Pillow not installed")
        return False
    
    try:
        import numpy as np
        print(f"[OK] NumPy {np.__version__}")
    except ImportError:
        print("[ERROR] NumPy not installed")
        return False
    
    return True


def test_config():
    """Test configuration file"""
    print("\nTesting configuration...")
    
    if not os.path.exists("config.json"):
        print("[ERROR] config.json not found")
        return False
    
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        print("[OK] Config file loaded successfully")
        
        required_keys = [
            "data_dir", "batch_size", "learning_rate", 
            "base_image_size", "sr_image_size"
        ]
        
        for key in required_keys:
            if key in config:
                print(f"[OK] {key}: {config[key]}")
            else:
                print(f"[ERROR] Missing config key: {key}")
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Error loading config: {e}")
        return False


def test_dataset_structure():
    """Test dataset directory structure"""
    print("\nTesting dataset structure...")
    
    # Load config to get data directory
    try:
        with open("config.json", 'r') as f:
            config = json.load(f)
        data_dir = config.get("data_dir", "./dataset")
    except:
        data_dir = "./dataset"
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"[ERROR] Dataset directory not found: {data_dir}")
        print(f"  Please create the directory and add your dataset")
        return False
    
    print(f"[OK] Dataset directory exists: {data_dir}")
    
    # Check for train/test structure
    required_dirs = [
        ("train", "image"), ("train", "cloth"),
        ("test", "image"), ("test", "cloth")
    ]
    
    for split, subdir in required_dirs:
        full_path = data_path / split / subdir
        if full_path.exists():
            file_count = len(list(full_path.glob("*")))
            print(f"[OK] {split}/{subdir}: {file_count} files")
        else:
            print(f"[WARN] {split}/{subdir}: not found")
    
    # Check for additional directories
    optional_dirs = [
        "agnostic-v3.2", "cloth-mask", "image-densepose",
        "image-parse-agnostic-v3.2", "image-parse-v3", "openpose_img", "openpose_json"
    ]
    
    for split in ["train", "test"]:
        split_path = data_path / split
        if split_path.exists():
            for opt_dir in optional_dirs:
                dir_path = split_path / opt_dir
                if dir_path.exists():
                    file_count = len(list(dir_path.glob("*")))
                    print(f"[OK] {split}/{opt_dir}: {file_count} files")
    
    return True


def test_pair_files():
    """Test pair files"""
    print("\nTesting pair files...")
    
    pair_files = ["train-pairs.txt", "test-pairs.txt"]
    
    for pair_file in pair_files:
        if os.path.exists(pair_file):
            with open(pair_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            print(f"[OK] {pair_file}: {len(lines)} pairs")
        else:
            print(f"[WARN] {pair_file}: not found (will be created automatically)")
    
    return True


def test_model_import():
    """Test if the model can be imported"""
    print("\nTesting model import...")
    
    try:
        from train import TryOnDataset, TrainingConfig, TryOnTrainer
        print("[OK] Training components imported successfully")
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        return False
    
    try:
        # Test basic model creation
        config = TrainingConfig("config.json")
        print("[OK] Training config created")
        
        # Test dataset creation (synthetic mode)
        dataset = TryOnDataset(
            data_dir=config.data_dir,
            use_real_data=False,
            num_samples=10
        )
        print(f"[OK] Synthetic dataset created: {len(dataset)} samples")
        
        # Test a sample
        sample = dataset[0]
        print(f"[OK] Sample keys: {list(sample.keys())}")
        print(f"[OK] Image shapes: {sample['person_images'].shape}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Model test failed: {e}")
        return False


def test_inference_setup():
    """Test inference script"""
    print("\nTesting inference setup...")
    
    if not os.path.exists("simple_inference.py"):
        print("[ERROR] simple_inference.py not found")
        return False
    
    try:
        # Just test import, don't run actual inference
        import simple_inference
        print("[OK] Inference script can be imported")
        return True
    except Exception as e:
        print(f"[ERROR] Inference import failed: {e}")
        return False


def main():
    """Run all tests"""
    print("TryOn Diffusion Setup Test")
    print("==========================")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_config),
        ("Dataset Structure", test_dataset_structure),
        ("Pair Files", test_pair_files),
        ("Model Import", test_model_import),
        ("Inference Setup", test_inference_setup),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    all_passed = True
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("All tests passed! Ready to train.")
        print("\nNext steps:")
        print("1. Put your dataset in './dataset/' directory")
        print("2. Run: python create_pairs.py")
        print("3. Run: python train.py --config config.json")
        print("4. Or use the launch script: bash launch_training.sh")
    else:
        print("Some tests failed. Please fix the issues above.")
    
    return all_passed


if __name__ == "__main__":
    main()