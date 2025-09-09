#!/usr/bin/env python3
"""
Test script to demonstrate dataset existence checking functionality
"""

import sys
from pathlib import Path

# Add the current directory to the path so we can import from train.py
sys.path.append(str(Path(__file__).parent))

from train import check_dataset_exists, download_kaggle_dataset


def test_dataset_existence_check():
    """Test the dataset existence checking functionality"""
    
    print("Testing dataset existence checking...")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        "./data",           # Main data directory
        "./nonexistent",    # Non-existent directory
        ".",               # Current directory
    ]
    
    for test_dir in test_cases:
        exists = check_dataset_exists(test_dir)
        print(f"Directory: {test_dir}")
        print(f"  Exists: {Path(test_dir).exists()}")
        print(f"  Valid dataset: {exists}")
        
        if Path(test_dir).exists():
            # Show what's inside
            contents = list(Path(test_dir).iterdir())[:5]  # First 5 items
            print(f"  Contents: {[item.name for item in contents]}")
        print()
    
    print("Testing download with existing dataset check...")
    print("=" * 50)
    
    # Test the smart download function
    for test_dir in ["./data", "./test_data"]:
        print(f"Testing download to: {test_dir}")
        if check_dataset_exists(test_dir):
            print(f"  ✅ Dataset exists in {test_dir}, download would be skipped")
        else:
            print(f"  ⬇️ Dataset does not exist in {test_dir}, download would proceed")
            print(f"     (Not actually downloading in this test)")
        print()


if __name__ == "__main__":
    test_dataset_existence_check()