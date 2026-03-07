import numpy as np
import rasterio
import os
import json
import albumentations as A

def load_tif(path):
    """
    Opens a .tif file and returns it as a numpy array
    Shape comes out as (Channels, H, W) from rasterio
    We fix it to (H, W, Channels)
    """
    with rasterio.open(path) as src:
        data = src.read()  # (C, H, W)
    
    data = np.transpose(data, (1, 2, 0))  # → (H, W, C)
    return data.astype(np.float32)  # convert to float for math


def normalize_image(image):
    """
    Apply Z-score normalization PER BAND
    Each band gets its own mean and std
    
    Input  shape: (128, 128, 12)
    Output shape: (128, 128, 12)  ← same shape, different values
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    
    for band in range(image.shape[2]):  # loop through each band
        band_data = image[:, :, band]
        
        mean = band_data.mean()
        std  = band_data.std()
        
        # avoid division by zero
        if std == 0:
            normalized[:, :, band] = 0
        else:
            normalized[:, :, band] = (band_data - mean) / std
    
    return normalized


def normalize_mask(mask):
    """
    Masks are already binary (0 or 1)
    Just make sure values are exactly 0.0 or 1.0
    """
    return (mask > 0).astype(np.float32)


def load_and_preprocess(image_path, mask_path, stats):
    """
    Full pipeline — now includes no-data handling
    
    load → clean no-data → normalize → return
    """
    image = load_tif(image_path)
    mask  = load_tif(mask_path)
    
    image = replace_nodata(image)        # ← NEW step
    image = normalize_with_stats(image, stats)
    mask  = normalize_mask(mask)
    
    return image, mask

def compute_global_stats(image_dir, image_files):
    """
    Compute mean and std per band across ALL training images.
    Now correctly handles no-data values BEFORE computing stats.
    """
    num_bands   = 12
    band_pixels = [[] for _ in range(num_bands)]
    
    print(f"Computing stats across {len(image_files)} images...")
    
    for i, fname in enumerate(image_files):
        if i % 100 == 0:
            print(f"  Processing image {i}/{len(image_files)}...")
        
        path  = os.path.join(image_dir, fname)
        image = load_tif(path)
        
        # ✅ Clean FIRST before collecting pixels
        image = replace_nodata(image, nodata_value=-9999)
        
        for band in range(num_bands):
            pixels = image[:, :, band].flatten()
            band_pixels[band].extend(pixels)
    
    stats = {"mean": [], "std": []}
    
    print("\n=== Global Stats Per Band (cleaned) ===")
    for band in range(num_bands):
        arr  = np.array(band_pixels[band])
        mean = float(arr.mean())
        std  = float(arr.std())
        
        stats["mean"].append(mean)
        stats["std"].append(std)
        
        print(f"Band {band+1:2d} → mean: {mean:10.3f}   std: {std:8.3f}")
    
    return stats


def save_stats(stats, save_path="../data/processed/stats.json"):
    """
    Save the computed stats to a JSON file
    so we never need to recompute them again
    """
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=4)
    print(f"  Stats saved to {save_path}")


def load_stats(stats_path="../data/processed/stats.json"):
    """
    Load previously saved stats
    """
    with open(stats_path, "r") as f:
        stats = json.load(f)
    print(f"  Stats loaded from {stats_path}")
    return stats


def normalize_with_stats(image, stats):
    """
    Normalize using global stats.
    Now includes NaN/Inf safety check after normalizing.
    """
    normalized = np.zeros_like(image, dtype=np.float32)
    
    for band in range(image.shape[2]):
        mean = stats["mean"][band]
        std  = stats["std"][band]
        
        if std == 0:
            normalized[:, :, band] = 0
        else:
            normalized[:, :, band] = (image[:, :, band] - mean) / std
    
    # Safety net — catch any remaining NaN or Inf
    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    
    return normalized

def replace_nodata(image, nodata_value=-9999):
    """
    Replace no-data pixels safely.
    Handles fully corrupted bands where ALL pixels are -9999.
    """
    cleaned = image.copy().astype(np.float32)
    
    for band in range(image.shape[2]):
        band_data   = cleaned[:, :, band]
        nodata_mask = (band_data == nodata_value)
        
        if nodata_mask.sum() == 0:
            continue  # no nodata, skip this band
        
        valid_pixels = band_data[~nodata_mask]
        
        if len(valid_pixels) == 0:
            # Entire band is corrupted
            replacement = 0.0
            print(f"  ⚠️  Band {band+1} fully corrupted → using 0.0")
        else:
            replacement = float(np.median(valid_pixels))
        
        # Extra safety — if somehow still NaN
        if np.isnan(replacement):
            replacement = 0.0
        
        band_data[nodata_mask] = replacement
        cleaned[:, :, band]   = band_data
    
    # Nuclear option — kill any remaining NaN/Inf
    cleaned = np.nan_to_num(cleaned, nan=0.0, posinf=0.0, neginf=0.0)
    
    return cleaned


# ======================== Augmentation Pipelines ========================
# Applied only during training. Val/Test use no augmentation.
# Works in HWC format (H, W, 12) — albumentations native.

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    # Note: No brightness/contrast or noise — z-score normalized spectral
    # bands carry meaning in their exact ratios; perturbing them hurts water detection.
])

# Validation — no augmentation
val_transform = None
