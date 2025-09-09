# Dataset Handling - Smart Download & Caching

The TryOn Diffusion training program now includes intelligent dataset handling that **automatically checks for existing datasets** and **avoids unnecessary re-downloads**.

## 🔄 How It Works

### **Smart Dataset Detection**

The system checks for existing datasets using multiple validation patterns:

```python
def check_dataset_exists(data_dir: str) -> bool:
    # Checks for common VITON dataset structures:
    # 1. train/image + train/cloth (primary structure)
    # 2. image + cloth (alternative structure)  
    # 3. train + test (basic structure)
    
    # Returns True only if directories exist AND contain files
```

### **Automatic Download Logic**

```python
# 1. Check if dataset exists in target directory
if check_dataset_exists("./data"):
    print("Dataset exists, skipping download")
    return "./data"

# 2. Only download if dataset is missing
if download_dataset and not dataset_exists:
    download_kaggle_dataset("./data")
```

## 📂 Detection Scenarios

### **Scenario 1: Dataset Already Exists**
```bash
# First run - downloads dataset
python train.py --download_dataset --data_dir ./data

# Output:
# Dataset not found in ./data, attempting download...
# Dataset downloaded to ./data

# Second run - skips download
python train.py --download_dataset --data_dir ./data

# Output:
# Using existing dataset from ./data
# Loaded real dataset with 15000 samples from ./data
```

### **Scenario 2: Dataset Missing**
```bash
# Run without existing dataset
python train.py --download_dataset --data_dir ./new_data

# Output:
# Dataset not found in ./new_data, attempting download...
# Dataset downloaded to ./new_data
```

### **Scenario 3: Dataset Detection Failed**
```bash
# Run with download_dataset=False and no existing dataset
python train.py --data_dir ./missing_data

# Output:
# Dataset not found in ./missing_data and download_dataset=False
# To download dataset, set download_dataset=True or run setup_dataset.py
# Falling back to synthetic data
```

## 🛡️ Robustness Features

### **1. Multiple Structure Detection**
Handles different VITON dataset organizations:
- `train/image/`, `train/cloth/` (HR-VITON format)
- `image/`, `cloth/` (simplified format)
- `train/`, `test/` (basic format)

### **2. Content Validation**
Not just directory existence - checks for actual files:
```python
# Ensures directories contain files, not just empty folders
len(list((data_path / dir_name).glob("*"))) > 0
```

### **3. Graceful Fallbacks**
- Missing dataset → Falls back to synthetic data
- Download failure → Falls back to synthetic data
- Invalid structure → Attempts alternative detection

### **4. Kagglehub Caching**
Leverages kagglehub's built-in caching:
- First download: Downloads to kagglehub cache
- Subsequent calls: Uses cached version
- No redundant network requests

## 🔧 Usage Examples

### **Setup Script with Smart Detection**
```bash
# Check existing dataset
python setup_dataset.py --verify_only --output_dir ./data

# Download only if missing
python setup_dataset.py --output_dir ./data
```

### **Training with Smart Downloads**
```bash
# Automatic: downloads if missing, uses existing if present
python train.py --config config.json --download_dataset

# Manual: specify exact data directory
python train.py --data_dir /path/to/existing/dataset

# Test: force synthetic data (ignores real data)
python train.py --synthetic_data
```

### **Testing Dataset Detection**
```bash
# Test the detection logic
python test_dataset_check.py

# Output shows detection results for various directories
```

## 📊 Performance Benefits

### **Time Savings**
- **First run**: 10-30 minutes download time
- **Subsequent runs**: ~2 seconds detection time
- **No redundant downloads**: Saves bandwidth and time

### **Storage Efficiency** 
- **Kagglehub caching**: Single download for multiple projects
- **Smart copying**: Only copies when needed
- **No duplicates**: Reuses existing valid datasets

### **Reliability**
- **Network failure recovery**: Uses cached/existing data
- **Partial download handling**: Detects incomplete datasets
- **Structure validation**: Ensures dataset integrity

## 🔍 Debug Information

### **Enable Detailed Logging**
```python
import logging
logging.basicConfig(level=logging.INFO)

# Will show detailed dataset detection process:
# INFO - Dataset not found in ./data, attempting download...
# INFO - Dataset downloaded/cached at: /path/to/cache
# INFO - Using existing dataset from ./data
# INFO - Loaded real dataset with 15000 samples
```

### **Check Dataset Manually**
```bash
# Verify dataset structure
python -c "
from train import check_dataset_exists
print('Dataset exists:', check_dataset_exists('./data'))
"
```

## 🎯 Best Practices

### **1. Use Consistent Data Directories**
```bash
# Good: consistent path
python train.py --data_dir ./data --download_dataset
python inference.py --data_dir ./data

# Avoid: changing paths unnecessarily
python train.py --data_dir ./dataset1 --download_dataset
python train.py --data_dir ./dataset2 --download_dataset  # Re-downloads
```

### **2. Leverage Kagglehub Caching**
```bash
# First project
cd /project1
python train.py --download_dataset  # Downloads to cache

# Second project  
cd /project2
python train.py --download_dataset  # Uses cached version
```

### **3. Verify Before Training**
```bash
# Always verify before long training runs
python setup_dataset.py --verify_only --analyze --output_dir ./data
python train.py --config config.json  # Proceeds with confidence
```

The improved dataset handling ensures efficient, reliable training workflows while minimizing redundant downloads and storage usage! ✨