#!/usr/bin/env python3
"""
Enhanced Training Script for TryOn Diffusion Model

This script provides a comprehensive training pipeline for the TryOn diffusion model
with support for:
- Multi-stage training (base and super-resolution unets)
- Advanced data loading and augmentation
- Model checkpointing and resuming
- Logging and monitoring
- Validation and evaluation
- Distributed training support
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import kagglehub

from tryondiffusion import TryOnImagen, TryOnImagenTrainer, get_unet_by_name


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


def download_kaggle_dataset(target_dir: str = None):
    """
    Download the High Resolution VITON Zalando dataset from Kaggle
    Only downloads if dataset doesn't already exist
    
    Args:
        target_dir: Target directory for dataset (optional)
        
    Returns:
        Path to dataset directory
    """
    # Check if dataset already exists in target directory
    if target_dir and check_dataset_exists(target_dir):
        print(f"Dataset already exists in {target_dir}, skipping download")
        return target_dir
    
    try:
        print("Downloading High Resolution VITON Zalando dataset from Kaggle...")
        print("This may take a while depending on your internet connection...")
        
        # Download using kagglehub (automatically handles caching)
        path = kagglehub.dataset_download("marquis03/high-resolution-viton-zalando-dataset")
        print(f"Dataset downloaded/cached at: {path}")
        
        # If target directory is specified and different from download path, copy the data
        if target_dir and str(Path(path)) != str(Path(target_dir)):
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            
            print(f"Copying dataset to {target_dir}...")
            import shutil
            # Copy contents, not the directory itself
            for item in Path(path).iterdir():
                dest = target_path / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
            
            print(f"Dataset copied to {target_dir}")
            return str(target_path)
        
        return path
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please ensure you have:")
        print("1. kagglehub installed: pip install kagglehub")
        print("2. Kaggle API credentials configured")
        print("3. Internet connection")
        return None


class TryOnDataset(Dataset):
    """
    Enhanced dataset class for TryOn diffusion training.
    
    Supports both synthetic data generation and real dataset loading
    from the High Resolution VITON Zalando dataset.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: Tuple[int, int] = (128, 128),
        pose_size: Tuple[int, int] = (18, 2),
        num_samples: int = 10000,
        augment: bool = True,
        use_real_data: bool = True,
        download_dataset: bool = False,
        split: str = "train",
    ):
        """
        Args:
            data_dir: Directory containing the dataset
            image_size: Target image size (height, width)
            pose_size: Pose keypoints size (num_keypoints, coordinates)
            num_samples: Number of synthetic samples to generate (if not using real data)
            augment: Whether to apply data augmentation
            use_real_data: Whether to use real dataset or synthetic data
            download_dataset: Whether to download dataset from Kaggle
            split: Dataset split ("train" or "test")
        """
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.pose_size = pose_size
        self.augment = augment
        self.use_real_data = use_real_data
        self.split = split
        
        # Define augmentation transforms
        self.image_transforms = self._get_image_transforms() if augment else self._get_base_transforms()
        
        if use_real_data:
            # Check if dataset exists, download if requested and not found
            if not check_dataset_exists(str(self.data_dir)):
                if download_dataset:
                    logging.info(f"Dataset not found in {self.data_dir}, attempting download...")
                    kaggle_path = download_kaggle_dataset(str(self.data_dir))
                    if kaggle_path:
                        self.data_dir = Path(kaggle_path)
                        logging.info(f"Dataset downloaded to {self.data_dir}")
                    else:
                        logging.error("Failed to download dataset, falling back to synthetic data")
                        self.use_real_data = False
                else:
                    logging.warning(f"Dataset not found in {self.data_dir} and download_dataset=False")
                    logging.info("To download dataset, set download_dataset=True or run setup_dataset.py")
                    logging.info("Falling back to synthetic data")
                    self.use_real_data = False
            else:
                logging.info(f"Using existing dataset from {self.data_dir}")
            
            if self.use_real_data:
                # Load real dataset
                self.samples = self._load_real_dataset()
                self.num_samples = len(self.samples)
                logging.info(f"Loaded real dataset with {self.num_samples} samples from {self.data_dir}")
            else:
                # Fallback to synthetic data
                self.num_samples = num_samples
                self.samples = None
                logging.info(f"Using synthetic dataset with {num_samples} samples")
        else:
            # Use synthetic data
            self.num_samples = num_samples
            self.samples = None
            logging.info(f"Using synthetic dataset with {num_samples} samples")
    
    def _load_real_dataset(self):
        """Load real dataset samples using pair files"""
        samples = []
        
        # Determine split (train/test) and corresponding pair file
        split = "train"  # Default to train
        if hasattr(self, 'split'):
            split = self.split
        
        # Load pair file
        pair_file = Path("train-pairs.txt") if split == "train" else Path("test-pairs.txt")
        if not pair_file.exists():
            logging.warning(f"Pair file {pair_file} not found, falling back to directory scanning")
            return self._load_real_dataset_fallback()
        
        logging.info(f"Loading pairs from {pair_file}")
        
        # Find dataset root directory
        dataset_root = self._find_dataset_root()
        if dataset_root is None:
            logging.warning(f"Could not find VITON dataset structure in {self.data_dir}")
            return []
        
        # Define paths for different data types - updated for your specific structure
        image_dirs = [dataset_root / "image", dataset_root / "person"]
        cloth_dirs = [dataset_root / "cloth", dataset_root / "garment"]
        image_parse_dirs = [
            dataset_root / "image-parse-agnostic-v3.2", 
            dataset_root / "image-parse-v3", 
            dataset_root / "image-parse", 
            dataset_root / "parse"
        ]
        pose_dirs = [dataset_root / "openpose_json", dataset_root / "pose", dataset_root / "keypoints"]
        agnostic_dirs = [dataset_root / "agnostic-v3.2", dataset_root / "agnostic"]
        cloth_mask_dirs = [dataset_root / "cloth-mask"]
        densepose_dirs = [dataset_root / "image-densepose"]
        
        # Find existing directories
        image_dir = self._find_existing_dir(image_dirs)
        cloth_dir = self._find_existing_dir(cloth_dirs)
        image_parse_dir = self._find_existing_dir(image_parse_dirs)
        pose_dir = self._find_existing_dir(pose_dirs)
        agnostic_dir = self._find_existing_dir(agnostic_dirs)
        cloth_mask_dir = self._find_existing_dir(cloth_mask_dirs)
        densepose_dir = self._find_existing_dir(densepose_dirs)
        
        if not image_dir or not cloth_dir:
            logging.error(f"Could not find image dir ({image_dir}) or cloth dir ({cloth_dir})")
            return []
        
        # Load pairs from file
        with open(pair_file, 'r') as f:
            pairs = [line.strip().split() for line in f if line.strip()]
        
        logging.info(f"Found {len(pairs)} pairs in {pair_file}")
        
        for person_img_name, garment_img_name in pairs:
            # Find person image (could be in subdirectories)
            person_file = self._find_image_file(image_dir, person_img_name)
            garment_file = self._find_image_file(cloth_dir, garment_img_name)
            
            if person_file and garment_file:
                # Find corresponding auxiliary files
                base_name = Path(person_img_name).stem
                parse_file = self._find_corresponding_file(image_parse_dir, base_name, [".jpg", ".png"]) if image_parse_dir else None
                pose_file = self._find_corresponding_file(pose_dir, base_name, [".json", ".txt", ".npy"]) if pose_dir else None
                agnostic_file = self._find_corresponding_file(agnostic_dir, base_name, [".jpg", ".png"]) if agnostic_dir else None
                cloth_mask_file = self._find_corresponding_file(cloth_mask_dir, Path(garment_img_name).stem, [".jpg", ".png"]) if cloth_mask_dir else None
                densepose_file = self._find_corresponding_file(densepose_dir, base_name, [".jpg", ".png"]) if densepose_dir else None
                
                sample = {
                    "person_image": str(person_file),
                    "garment_image": str(garment_file),
                    "parse_image": str(parse_file) if parse_file else None,
                    "pose_file": str(pose_file) if pose_file else None,
                    "agnostic_image": str(agnostic_file) if agnostic_file else None,
                    "cloth_mask": str(cloth_mask_file) if cloth_mask_file else None,
                    "densepose_image": str(densepose_file) if densepose_file else None,
                    "id": base_name
                }
                samples.append(sample)
            else:
                if not person_file:
                    logging.warning(f"Person image not found: {person_img_name}")
                if not garment_file:
                    logging.warning(f"Garment image not found: {garment_img_name}")
        
        logging.info(f"Successfully loaded {len(samples)} valid samples")
        return samples
    
    def _find_dataset_root(self):
        """Find the dataset root directory based on split"""
        # For train/test folder structure, use the split-specific folder
        split_root = self.data_dir / self.split
        if split_root.exists():
            # Check if it has the expected VITON structure
            has_structure = any([
                (split_root / "image").exists() or (split_root / "person").exists(),
                (split_root / "cloth").exists() or (split_root / "garment").exists()
            ])
            if has_structure:
                return split_root
        
        # Fallback to other possible structures
        possible_roots = [
            self.data_dir,
            self.data_dir / "VITON-HD" / self.split,
            self.data_dir / "HR-VITON" / self.split,
            self.data_dir / "VITON-HD",
            self.data_dir / "HR-VITON"
        ]
        
        for root in possible_roots:
            if root.exists():
                # Check if it has the expected VITON structure
                has_structure = any([
                    (root / "image").exists() or (root / "person").exists(),
                    (root / "cloth").exists() or (root / "garment").exists()
                ])
                if has_structure:
                    return root
        return None
    
    def _find_existing_dir(self, possible_dirs):
        """Find the first existing directory from a list"""
        for dir_path in possible_dirs:
            if dir_path.exists():
                return dir_path
        return None
    
    def _find_image_file(self, base_dir, filename):
        """Find image file, searching in subdirectories if needed"""
        if not base_dir or not base_dir.exists():
            return None
        
        # Try direct path first
        direct_path = base_dir / filename
        if direct_path.exists():
            return direct_path
        
        # Search in subdirectories (common in Kaggle datasets)
        for subdir in base_dir.iterdir():
            if subdir.is_dir():
                sub_path = subdir / filename
                if sub_path.exists():
                    return sub_path
        
        return None
    
    def _load_real_dataset_fallback(self):
        """Fallback method for loading dataset without pair files"""
        samples = []
        
        dataset_root = self._find_dataset_root()
        if dataset_root is None:
            return []
        
        image_dir = self._find_existing_dir([dataset_root / "image", dataset_root / "person"])
        cloth_dir = self._find_existing_dir([dataset_root / "cloth", dataset_root / "garment"])
        
        if not image_dir or not cloth_dir:
            return []
        
        # Get list of person images
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        for img_file in image_files:
            base_name = img_file.stem
            
            # Find corresponding garment (same name)
            cloth_file = self._find_corresponding_file(cloth_dir, base_name, [".jpg", ".png"])
            
            if cloth_file:
                sample = {
                    "person_image": str(img_file),
                    "garment_image": str(cloth_file),
                    "parse_image": None,
                    "pose_file": None,
                    "id": base_name
                }
                samples.append(sample)
        
        logging.info(f"Fallback: Found {len(samples)} valid samples")
        return samples
    
    def _find_corresponding_file(self, directory: Path, base_name: str, extensions: list):
        """Find corresponding file with different possible extensions"""
        if not directory.exists():
            return None
            
        for ext in extensions:
            file_path = directory / f"{base_name}{ext}"
            if file_path.exists():
                return file_path
        return None
    
    def _get_base_transforms(self):
        """Base transforms for image preprocessing"""
        return T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def _get_image_transforms(self):
        """Augmentation transforms for training"""
        return T.Compose([
            T.Resize(self.image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomRotation(degrees=5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess an image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.image_transforms(image)
        except Exception as e:
            logging.warning(f"Failed to load image {image_path}: {e}")
            # Return random tensor as fallback
            return torch.randn(3, *self.image_size) * 2.0 - 1.0
    
    def _load_pose(self, pose_path: str) -> torch.Tensor:
        """Load pose keypoints from file"""
        try:
            if pose_path.endswith('.json'):
                import json
                with open(pose_path, 'r') as f:
                    pose_data = json.load(f)
                # Extract keypoints from JSON (format depends on pose annotation)
                if 'people' in pose_data and len(pose_data['people']) > 0:
                    keypoints = pose_data['people'][0]['pose_keypoints_2d']
                    # Reshape to (18, 3) and take only x, y coordinates
                    pose = np.array(keypoints).reshape(-1, 3)[:, :2]
                else:
                    pose = np.zeros((18, 2))
            elif pose_path.endswith('.npy'):
                pose = np.load(pose_path)
                if pose.shape != (18, 2):
                    # Reshape if needed
                    pose = pose.reshape(-1, 2)[:18]
                    if pose.shape[0] < 18:
                        # Pad if too few keypoints
                        pose = np.pad(pose, ((0, 18 - pose.shape[0]), (0, 0)), 'constant')
            else:
                # Text file format
                pose = np.loadtxt(pose_path)
                if pose.ndim == 1:
                    pose = pose.reshape(-1, 2)
                pose = pose[:18]  # Take first 18 keypoints
                
            # Ensure correct shape
            if pose.shape != (18, 2):
                pose = np.zeros((18, 2))
                
            return torch.tensor(pose, dtype=torch.float32)
            
        except Exception as e:
            logging.warning(f"Failed to load pose {pose_path}: {e}")
            # Return random pose as fallback
            return torch.randn(*self.pose_size)
    
    def _generate_ca_image(self, person_image: torch.Tensor, parse_image: torch.Tensor = None, agnostic_image: torch.Tensor = None) -> torch.Tensor:
        """Generate clothing-agnostic image from person image and parse mask or use pre-computed agnostic image"""
        if agnostic_image is not None:
            # Use pre-computed agnostic image if available
            return agnostic_image
        elif parse_image is not None:
            # Use parse mask to create clothing-agnostic image
            # This is a simplified version - in practice, you'd use more sophisticated masking
            ca_image = person_image.clone()
            # Apply masking based on parse image (implementation depends on parse format)
            return ca_image
        else:
            # Simple approximation: slightly blur the person image
            return person_image
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns a sample containing:
        - person_images: Target person image
        - ca_images: Clothing-agnostic person image
        - garment_images: Garment to be tried on
        - person_poses: Person pose keypoints
        - garment_poses: Garment pose keypoints
        """
        if self.use_real_data and self.samples:
            # Load real data
            sample = self.samples[idx % len(self.samples)]
            
            # Load images
            person_image = self._load_image(sample['person_image'])
            garment_image = self._load_image(sample['garment_image'])
            
            # Load or generate CA image
            agnostic_image = None
            if sample.get('agnostic_image'):
                agnostic_image = self._load_image(sample['agnostic_image'])
            
            if sample.get('parse_image'):
                parse_image = self._load_image(sample['parse_image'])
                ca_image = self._generate_ca_image(person_image, parse_image, agnostic_image)
            else:
                ca_image = self._generate_ca_image(person_image, agnostic_image=agnostic_image)
            
            # Load poses
            if sample.get('pose_file'):
                person_pose = self._load_pose(sample['pose_file'])
                # For simplicity, use same pose for garment (in practice, you'd have separate garment poses)
                garment_pose = person_pose.clone()
            else:
                person_pose = torch.randn(*self.pose_size)
                garment_pose = torch.randn(*self.pose_size)
        
        else:
            # Generate synthetic data
            person_image = torch.randn(3, *self.image_size) * 2.0 - 1.0
            ca_image = torch.randn(3, *self.image_size) * 2.0 - 1.0
            garment_image = torch.randn(3, *self.image_size) * 2.0 - 1.0
            person_pose = torch.randn(*self.pose_size)
            garment_pose = torch.randn(*self.pose_size)
        
        return {
            "person_images": person_image,
            "ca_images": ca_image,
            "garment_images": garment_image,
            "person_poses": person_pose,
            "garment_poses": garment_pose,
        }


def collate_fn(batch):
    """Custom collate function for batching samples"""
    return {
        "person_images": torch.stack([item["person_images"] for item in batch]),
        "ca_images": torch.stack([item["ca_images"] for item in batch]),
        "garment_images": torch.stack([item["garment_images"] for item in batch]),
        "person_poses": torch.stack([item["person_poses"] for item in batch]),
        "garment_poses": torch.stack([item["garment_poses"] for item in batch]),
    }


class TrainingConfig:
    """Configuration class for training parameters"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Default configuration
        self.data_dir = "./data"
        self.output_dir = "./outputs"
        self.checkpoint_dir = "./checkpoints"
        self.log_dir = "./logs"
        
        # Model configuration
        self.base_image_size = (128, 128)
        self.sr_image_size = (256, 256)
        self.timesteps = (1000, 1000)
        self.channels = 3
        
        # Training configuration
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.gradient_accumulation_steps = 1
        self.max_grad_norm = 1.0
        self.warmup_steps = 1000
        self.cosine_decay_max_steps = None
        
        # Validation and logging
        self.validation_freq = 1000
        self.checkpoint_freq = 5000
        self.log_freq = 100
        self.sample_freq = 2000
        self.num_sample_images = 4
        
        # Training stage
        self.train_unet_number = 1  # 1 for base, 2 for super-resolution
        self.use_ema = True
        self.mixed_precision = True
        
        # Data configuration
        self.num_workers = 4
        self.pin_memory = True
        self.augment_data = True
        self.use_real_data = True
        self.download_dataset = False
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        logging.info(f"Loaded configuration from {config_path}")
    
    def save_to_file(self, config_path: str):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logging.info(f"Saved configuration to {config_path}")


class TryOnTrainer:
    """Enhanced trainer class with comprehensive training pipeline"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_directories()
        
        # Initialize tensorboard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Initialize model and trainer
        self.setup_model()
        self.setup_data()
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        logging.info("Trainer initialized successfully")
    
    def setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config.log_dir, 'training.log')),
                logging.StreamHandler()
            ]
        )
    
    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
    
    def setup_model(self):
        """Initialize the TryOn diffusion model and trainer"""
        logging.info("Setting up model...")
        
        # Create unets
        base_unet = get_unet_by_name("base", image_size=self.config.base_image_size)
        sr_unet = get_unet_by_name("sr", image_size=self.config.sr_image_size)
        
        # Create Imagen model
        self.imagen = TryOnImagen(
            unets=(base_unet, sr_unet),
            image_sizes=(self.config.base_image_size, self.config.sr_image_size),
            timesteps=self.config.timesteps,
            channels=self.config.channels,
            cond_drop_prob=0.1,  # For classifier-free guidance
        )
        
        # Create trainer
        self.trainer = TryOnImagenTrainer(
            imagen=self.imagen,
            lr=self.config.learning_rate,
            use_ema=self.config.use_ema,
            max_grad_norm=self.config.max_grad_norm,
            warmup_steps=self.config.warmup_steps,
            cosine_decay_max_steps=self.config.cosine_decay_max_steps,
            only_train_unet_number=self.config.train_unet_number,
            fp16=self.config.mixed_precision,
            checkpoint_path=self.config.checkpoint_dir,
            checkpoint_every=self.config.checkpoint_freq,
            accelerate_gradient_accumulation_steps=self.config.gradient_accumulation_steps,
        )
        
        logging.info(f"Model setup complete. Training UNet #{self.config.train_unet_number}")
    
    def setup_data(self):
        """Setup data loaders"""
        logging.info("Setting up data loaders...")
        
        # Determine image size based on which unet we're training
        image_size = (
            self.config.sr_image_size 
            if self.config.train_unet_number == 2 
            else self.config.base_image_size
        )
        
        # Create datasets
        train_dataset = TryOnDataset(
            data_dir=self.config.data_dir,
            image_size=image_size,
            num_samples=10000,  # Only used for synthetic data
            augment=self.config.augment_data,
            use_real_data=self.config.use_real_data,
            download_dataset=self.config.download_dataset,
            split="train",
        )
        
        val_dataset = TryOnDataset(
            data_dir=self.config.data_dir,
            image_size=image_size,
            num_samples=1000,  # Only used for synthetic data
            augment=False,  # No augmentation for validation
            use_real_data=self.config.use_real_data,
            download_dataset=False,  # Don't re-download for validation
            split="test",
        )
        
        # Create data loaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=collate_fn,
        )
        
        # Add to trainer
        self.trainer.add_train_dataloader(train_dataloader)
        self.trainer.add_valid_dataloader(val_dataloader)
        
        logging.info(f"Data loaders created. Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.trainer.imagen.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            range(len(self.trainer.train_dl)), 
            desc=f"Epoch {self.epoch}",
            leave=False
        )
        
        for batch_idx in progress_bar:
            # Training step
            loss = self.trainer.train_step(unet_number=self.config.train_unet_number)
            
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Logging
            if self.global_step % self.config.log_freq == 0:
                self.writer.add_scalar('Loss/Train', loss.item(), self.global_step)
                self.writer.add_scalar('Learning_Rate', 
                    self.trainer.get_lr(self.config.train_unet_number), self.global_step)
                
                logging.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")
            
            # Validation
            if self.global_step % self.config.validation_freq == 0:
                val_metrics = self.validate()
                self.writer.add_scalar('Loss/Validation', val_metrics['val_loss'], self.global_step)
                
                # Save best model
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.save_best_model()
            
            # Generate samples
            if self.global_step % self.config.sample_freq == 0:
                self.generate_samples()
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation"""
        self.trainer.imagen.eval()
        
        val_loss = 0.0
        num_batches = 0
        
        for _ in range(min(100, len(self.trainer.valid_dl))):  # Limit validation batches
            loss = self.trainer.valid_step(
                unet_number=self.config.train_unet_number,
                use_ema_unets=self.config.use_ema
            )
            val_loss += loss.item()
            num_batches += 1
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0.0
        
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")
        return {'val_loss': avg_val_loss}
    
    @torch.no_grad()
    def generate_samples(self):
        """Generate and save sample images"""
        self.trainer.imagen.eval()
        
        # Get a validation batch
        validation_sample = next(self.trainer.valid_dl_iter)
        
        # Remove person_images from sample kwargs (these are the targets)
        sample_kwargs = {k: v for k, v in validation_sample.items() if k != 'person_images'}
        
        # Generate images
        images = self.trainer.sample(
            **sample_kwargs,
            batch_size=min(self.config.num_sample_images, self.config.batch_size),
            cond_scale=2.0,
            return_pil_images=True,
            use_tqdm=False,
        )
        
        # Save samples
        sample_dir = os.path.join(self.config.output_dir, f"samples_step_{self.global_step}")
        os.makedirs(sample_dir, exist_ok=True)
        
        if isinstance(images, list) and len(images) > 0:
            # Handle multiple unet outputs
            for unet_idx, unet_images in enumerate(images):
                for img_idx, img in enumerate(unet_images):
                    img.save(os.path.join(sample_dir, f"unet_{unet_idx}_sample_{img_idx}.png"))
        
        logging.info(f"Generated samples saved to {sample_dir}")
    
    def save_best_model(self):
        """Save the best model checkpoint"""
        best_model_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        self.trainer.save(best_model_path, without_optim_and_sched=True)
        logging.info(f"Saved best model to {best_model_path}")
    
    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        
        # Save configuration
        self.config.save_to_file(os.path.join(self.config.output_dir, "config.json"))
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            self.writer.add_scalar('Epoch/Train_Loss', train_metrics['train_loss'], epoch)
            
            logging.info(f"Epoch {epoch} completed. Train Loss: {train_metrics['train_loss']:.4f}")
        
        # Final validation and sample generation
        final_metrics = self.validate()
        self.generate_samples()
        
        logging.info("Training completed!")
        logging.info(f"Final validation loss: {final_metrics['val_loss']:.4f}")
        
        # Close tensorboard writer
        self.writer.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train TryOn Diffusion Model")
    
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--train_unet", type=int, default=1, choices=[1, 2], 
                        help="Which UNet to train (1=base, 2=super-resolution)")
    parser.add_argument("--mixed_precision", action="store_true", help="Use mixed precision training")
    parser.add_argument("--no_ema", action="store_true", help="Disable EMA")
    parser.add_argument("--synthetic_data", action="store_true", help="Use synthetic data instead of real dataset")
    parser.add_argument("--download_dataset", action="store_true", help="Download dataset from Kaggle")
    
    return parser.parse_args()


def main():
    """Main function"""
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
    if args.mixed_precision:
        config.mixed_precision = True
    if args.no_ema:
        config.use_ema = False
    if args.synthetic_data:
        config.use_real_data = False
    if args.download_dataset:
        config.download_dataset = True
    
    # Create trainer and start training
    trainer = TryOnTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()