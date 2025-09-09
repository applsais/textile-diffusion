#!/usr/bin/env python3
"""
Dataset Setup Script for TryOn Diffusion Training

This script downloads and prepares the High Resolution VITON Zalando dataset
from Kaggle for training the TryOn diffusion model.
"""

import argparse
import os
import shutil
from pathlib import Path
import kagglehub
import logging


def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def check_dataset_exists(data_dir: str) -> bool:
    """
    Check if dataset already exists with valid structure
    
    Args:
        data_dir: Directory to check for dataset
        
    Returns:
        True if valid dataset structure exists, False otherwise
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return False
    
    # Check for typical VITON dataset structure
    required_dirs = [
        ("train/image", "train/cloth"),  # Primary structure
        ("image", "cloth"),              # Alternative structure
        ("train", "test"),               # Basic structure
    ]
    
    for dir_set in required_dirs:
        if all((data_path / dir_name).exists() and 
               len(list((data_path / dir_name).glob("*"))) > 0 
               for dir_name in dir_set):
            return True
    
    return False


def download_viton_dataset(output_dir: str = "./data") -> str:
    """
    Download the High Resolution VITON Zalando dataset from Kaggle
    Only downloads if dataset doesn't already exist
    
    Args:
        output_dir: Directory to store the dataset
        
    Returns:
        Path to the downloaded dataset
    """
    # Check if dataset already exists
    if check_dataset_exists(output_dir):
        logging.info(f"Dataset already exists in {output_dir}, skipping download")
        return output_dir
    
    try:
        logging.info("Dataset not found, downloading High Resolution VITON Zalando dataset from Kaggle...")
        logging.info("This may take a while depending on your internet connection...")
        
        # Download dataset using kagglehub
        download_path = kagglehub.dataset_download("marquis03/high-resolution-viton-zalando-dataset")
        
        logging.info(f"Dataset downloaded/cached at: {download_path}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy dataset to desired location if different
        if str(Path(download_path)) != str(output_path):
            logging.info(f"Copying dataset to {output_dir}...")
            # Copy contents, not the directory itself
            for item in Path(download_path).iterdir():
                dest = output_path / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            logging.info("Dataset copied successfully")
            return str(output_path)
        
        return download_path
        
    except Exception as e:
        logging.error(f"Error downloading dataset: {e}")
        logging.error("Please ensure you have:")
        logging.error("1. kagglehub installed: pip install kagglehub")
        logging.error("2. Kaggle API credentials configured")
        logging.error("3. Internet connection")
        return None


def verify_dataset_structure(dataset_path: str) -> bool:
    """
    Verify that the downloaded dataset has the expected structure
    
    Args:
        dataset_path: Path to the dataset directory
        
    Returns:
        True if dataset structure is valid, False otherwise
    """
    dataset_path = Path(dataset_path)
    
    # Check for common VITON dataset directories
    expected_dirs = [
        "train/image",
        "train/cloth", 
        "train/image-parse",
        "train/pose",
        "test/image",
        "test/cloth"
    ]
    
    alternative_dirs = [
        "image",
        "cloth",
        "image-parse", 
        "pose"
    ]
    
    # Check primary structure
    found_dirs = []
    for dir_path in expected_dirs:
        full_path = dataset_path / dir_path
        if full_path.exists():
            found_dirs.append(dir_path)
            logging.info(f"Found: {dir_path} ({len(list(full_path.glob('*')))} files)")
    
    # Check alternative structure
    if not found_dirs:
        for dir_path in alternative_dirs:
            full_path = dataset_path / dir_path
            if full_path.exists():
                found_dirs.append(dir_path)
                logging.info(f"Found: {dir_path} ({len(list(full_path.glob('*')))} files)")
    
    if found_dirs:
        logging.info(f"Dataset structure verified. Found {len(found_dirs)} directories.")
        return True
    else:
        logging.warning("Could not verify dataset structure")
        logging.info("Expected directories:")
        for dir_path in expected_dirs:
            logging.info(f"  - {dir_path}")
        return False


def analyze_dataset(dataset_path: str):
    """
    Analyze the dataset and provide statistics
    
    Args:
        dataset_path: Path to the dataset directory
    """
    dataset_path = Path(dataset_path)
    
    logging.info("Analyzing dataset...")
    
    # Find all directories
    subdirs = [d for d in dataset_path.rglob("*") if d.is_dir()]
    
    for subdir in subdirs:
        if any(subdir.name in ["image", "cloth", "pose", "image-parse"] for name in [subdir.name]):
            file_count = len(list(subdir.glob("*")))
            if file_count > 0:
                logging.info(f"{subdir.relative_to(dataset_path)}: {file_count} files")
    
    # Sample file analysis
    image_dirs = [d for d in subdirs if "image" in d.name and d != dataset_path]
    if image_dirs:
        sample_dir = image_dirs[0]
        sample_files = list(sample_dir.glob("*.jpg"))[:5]
        if sample_files:
            logging.info(f"Sample files from {sample_dir.name}:")
            for f in sample_files:
                logging.info(f"  - {f.name}")


def create_directory_structure(output_dir: str):
    """Create the expected directory structure for training"""
    output_path = Path(output_dir)
    
    # Create main directories
    directories = [
        "data",
        "outputs", 
        "checkpoints",
        "logs",
        "samples"
    ]
    
    for dir_name in directories:
        dir_path = output_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {dir_path}")


def main():
    """Main function for dataset setup"""
    parser = argparse.ArgumentParser(description="Setup TryOn Diffusion Dataset")
    
    parser.add_argument("--output_dir", type=str, default="./data", 
                       help="Directory to store the dataset")
    parser.add_argument("--verify_only", action="store_true", 
                       help="Only verify existing dataset, don't download")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze dataset structure and contents")
    parser.add_argument("--setup_dirs", action="store_true",
                       help="Create directory structure for training")
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.setup_dirs:
        create_directory_structure(".")
        logging.info("Directory structure created")
    
    if args.verify_only:
        # Just verify existing dataset
        if os.path.exists(args.output_dir):
            verify_dataset_structure(args.output_dir)
            if args.analyze:
                analyze_dataset(args.output_dir)
        else:
            logging.error(f"Dataset directory {args.output_dir} does not exist")
    else:
        # Download dataset
        dataset_path = download_viton_dataset(args.output_dir)
        
        if dataset_path:
            # Verify downloaded dataset
            if verify_dataset_structure(dataset_path):
                logging.info("✅ Dataset setup completed successfully!")
                
                if args.analyze:
                    analyze_dataset(dataset_path)
                
                logging.info("\nNext steps:")
                logging.info("1. Run training: python train.py --config config.json --download_dataset")
                logging.info("2. Or use synthetic data: python train.py --synthetic_data")
            else:
                logging.warning("⚠️ Dataset structure verification failed")
        else:
            logging.error("❌ Dataset download failed")


if __name__ == "__main__":
    main()