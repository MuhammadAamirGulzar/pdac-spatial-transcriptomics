"""
Modular WSI Patch Creation System
==================================
Single image-focused patch extraction with clean, extensible architecture.
Extracts circular patches from a single WSI image.
Generates patches with filename format: sampleid_patch-id_x_y.png
Includes metadata and visualization overlays.

Configuration:
  - Modify IMAGE_FILEPATHS to select which image to process
  - Update IMAGE_CONFIGS for image metadata
  - Implement custom processing in PatchProcessor.apply_custom_processing()
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter


# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

# Image Filepaths - Change which image to process by modifying SELECTED_IMAGE
IMAGE_FILEPATHS = {
    'IU_PDA_T1': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T1-S13_11414_B9.tif",
    'IU_PDA_T4': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T4-S17_19380_H6.tif",
    'IU_PDA_T11': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T11-S21_23955_F12.tif",
    'IU_PDA_HM11': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_HM11-S21_23955_E3.tif",
    'IU_PDA_HM13': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_HM13-S22_1570_C2.tif",
    'IU_PDA_T3': r"D:\Zenodo PT & HM Dataset Work\.TIF Images\image_IU_PDA_T3-S14_8805_B2.tif",
}

# Image metadata mapping
IMAGE_CONFIGS = {
    'IU_PDA_T1': {
        'image_name': 'IU_PDA_T1',
        'patient': 'PT_1',
    },
    'IU_PDA_T4': {
        'image_name': 'IU_PDA_T4',
        'patient': 'PT_4',
    },
    'IU_PDA_T11': {
        'image_name': 'IU_PDA_T11',
        'patient': 'PT_11',
    },
    'IU_PDA_HM11': {
        'image_name': 'IU_PDA_HM11',
        'patient': 'HM_11',
    },
    'IU_PDA_HM13': {
        'image_name': 'IU_PDA_HM13',
        'patient': 'HM_13',
    },
    'IU_PDA_T3': {
        'image_name': 'IU_PDA_T3',
        'patient': 'PT_3',
    },
}

# SELECT WHICH IMAGE TO PROCESS HERE (overridden by a command-line sample id)
SELECTED_IMAGE = 'IU_PDA_T3'

# ----------------------------------------------------------------------------
# Inventory-driven resolution.
#
# The hard-coded IMAGE_FILEPATHS above point at `D:\Zenodo PT & HM Dataset Work`,
# a path from the previous machine, and cover only the original 6 samples.  The
# WSIs now live elsewhere and 21/30 samples have one, so resolve paths from
# Outputs/Patient-Sample-Information/wsi_inventory.csv instead (built by
# 01_Patch_Extraction/wsi_inventory.py).  The literals stay as documentation of
# which slide each original sample came from.
#
# Only `status == ready` samples are usable here: those are already cropped to a
# single capture area, so the Visium coordinates line up.  A `needs_crop` sample
# sits inside a 4-area whole-slide TIFF and MUST NOT be fed to this script -- the
# coordinates would be silently wrong.
# ----------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
WSI_SOURCE_DIR = Path(os.environ.get(
    "WSI_DIR", r"D:\Aamir Gulzar\KSA_project3\old_project_data\ST_source_WSI_data"))
_INVENTORY = _REPO_ROOT / "Outputs" / "Patient-Sample-Information" / "wsi_inventory.csv"


def load_inventory():
    """sample -> cropped WSI path, for samples that need no cropping."""
    if not _INVENTORY.exists():
        return {}, {}
    inv = pd.read_csv(_INVENTORY)
    paths, cfgs = {}, {}
    for _, r in inv.iterrows():
        if str(r.get("status")) != "ready" or not str(r.get("cropped_tif")):
            continue
        p = WSI_SOURCE_DIR / str(r["cropped_tif"])
        if p.exists():
            paths[r["sample"]] = str(p)
            cfgs[r["sample"]] = {"image_name": r["sample"], "patient": r["patient"]}
    return paths, cfgs


_inv_paths, _inv_cfgs = load_inventory()
if _inv_paths:
    IMAGE_FILEPATHS = _inv_paths
    IMAGE_CONFIGS = _inv_cfgs

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for patch creation parameters."""
    
    # Patch extraction parameters
    PATCH_SIZE = 224
    PATCH_FORMAT = 'PNG'
    
    # Default directories -- resolved from the repo, not the old machine's paths.
    # Patches are written straight into dataset/.png patches/.png patches/<sample>/
    # where build_qc_mask.py and the vision extractors already look for them.
    WORKSPACE_DIR = Path(__file__).resolve().parent.parent
    ST_DIR = WORKSPACE_DIR / "Outputs" / "Patient-Sample-Information"
    OUTPUT_BASE_DIR = WORKSPACE_DIR / "dataset" / ".png patches" / ".png patches"
    
    # Visualization
    CIRCLE_COLOR = (0, 255, 0)  # Green
    CIRCLE_WIDTH = 2
    CREATE_OVERLAY = True


# ============================================================================
# PATCH ID MANAGEMENT
# ============================================================================

class PatchIDManager:
    """Manages continuous patch ID allocation."""
    
    def __init__(self, start_id: int = 1):
        self.current_id = start_id
        self.allocations = []
    
    def allocate_id(self) -> int:
        """Get next patch ID and increment counter."""
        current = self.current_id
        self.current_id += 1
        self.allocations.append(current)
        return current
    
    def get_current(self) -> int:
        """Get current ID without incrementing."""
        return self.current_id
    
    def save_state(self, filepath: Path):
        """Save ID manager state to JSON."""
        data = {
            'current_id': self.current_id,
            'next_id': self.current_id + 1,
            'total_allocated': len(self.allocations),
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"[ok] ID manager state saved: {filepath}")


# ============================================================================
# DATA LOADING
# ============================================================================

class SpatialDataLoader:
    """Loads spatial coordinates and scale information from CSV files."""
    
    def __init__(self, spots_csv: Path, scale_csv: Path):
        """
        Initialize loader with CSV file paths.
        
        Args:
            spots_csv: Path to spot_spatial_coordinates.csv
            scale_csv: Path to visium_scale_information.csv
        """
        self.spots_csv = spots_csv
        self.scale_csv = scale_csv
        self.spots_df = None
        self.scale_df = None
        self._load_data()
    
    def _load_data(self):
        """Load CSV files."""
        try:
            self.spots_df = pd.read_csv(self.spots_csv)
            self.scale_df = pd.read_csv(self.scale_csv)
            print(f"[ok] Loaded {len(self.spots_df)} spot coordinates")
            print(f"[ok] Loaded scale info for {len(self.scale_df)} images\n")
        except Exception as e:
            raise RuntimeError(f"Failed to load spatial data: {e}")
    
    def get_spots_for_image(self, image_name: str) -> pd.DataFrame:
        """Get all spots for a specific image."""
        return self.spots_df[self.spots_df['image'] == image_name].copy()
    
    def get_scale_info(self, image_name: str) -> Dict:
        """Get scale information for a specific image."""
        scale_row = self.scale_df[self.scale_df['image_name'] == image_name]
        if len(scale_row) == 0:
            raise ValueError(f"No scale info found for {image_name}")
        return scale_row.iloc[0].to_dict()


# ============================================================================
# PATCH EXTRACTION
# ============================================================================

class PatchExtractor:
    """Extracts circular patches from WSI images."""
    
    def __init__(self, patch_size: int = 224):
        self.patch_size = patch_size
    
    def extract_circular_patch(
        self,
        wsi: np.ndarray,
        center_row: int,
        center_col: int,
        spot_diameter: float
    ) -> Optional[Image.Image]:
        """
        Extract a circular patch centered at given coordinates.
        
        Args:
            wsi: WSI image array
            center_row: Row coordinate of patch center
            center_col: Column coordinate of patch center
            spot_diameter: Diameter of the circular patch in pixels
        
        Returns:
            PIL Image of patch or None if invalid
        """
        radius = spot_diameter / 2
        
        # Calculate extraction bounds
        row_min = max(0, int(center_row - radius))
        row_max = min(wsi.shape[0], int(center_row + radius) + 1)
        col_min = max(0, int(center_col - radius))
        col_max = min(wsi.shape[1], int(center_col + radius) + 1)
        
        # Validate region size
        if row_max - row_min < 5 or col_max - col_min < 5:
            return None
        
        # Extract region from WSI
        region = wsi[row_min:row_max, col_min:col_max]
        
        # Create patch with black background
        patch_array = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        
        # Calculate offsets to center the extracted region
        offset_row = (self.patch_size - (row_max - row_min)) // 2
        offset_col = (self.patch_size - (col_max - col_min)) // 2
        
        end_row = offset_row + (row_max - row_min)
        end_col = offset_col + (col_max - col_min)
        
        # Place region in patch
        if len(region.shape) == 3 and region.shape[2] >= 3:
            patch_array[offset_row:end_row, offset_col:end_col] = region[:, :, :3]
        else:
            patch_array[offset_row:end_row, offset_col:end_col, 0] = region.astype(np.uint8)
        
        # Apply circular mask - use actual extracted region size, not original diameter
        # This ensures the mask matches the actual region, especially at image boundaries
        actual_region_height = row_max - row_min
        actual_region_width = col_max - col_min
        actual_radius = min(actual_region_height, actual_region_width) / 2
        
        cy, cx = self.patch_size // 2, self.patch_size // 2
        y_grid, x_grid = np.ogrid[:self.patch_size, :self.patch_size]
        distances = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)
        mask = distances > actual_radius
        patch_array[mask] = 0
        
        return Image.fromarray(patch_array, 'RGB')


# ============================================================================
# STAIN NORMALIZATION
# ============================================================================

class MacenkoNormalizer:
    """Macenko stain normalization for histopathological images."""
    
    def __init__(self):
        """Initialize with reference HE stain vectors."""
        # Reference stain vectors for HE staining
        self.reference_stains = np.array([
            [0.5626, 0.5821],
            [0.7201, 0.4042],
            [0.4062, 0.8041]
        ])
        
        self.ref_concentrations = np.array([1.9705, 1.0308])
    
    def normalize(self, img: Image.Image) -> Image.Image:
        """
        Apply Macenko stain normalization to image.
        
        Args:
            img: PIL Image (RGB)
        
        Returns:
            Normalized PIL Image
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(img).astype(np.float32)
            
            # Only process if not all black (from circular mask padding)
            if np.all(img_array < 5):
                return img
            
            # Normalize
            normalized = self._macenko_normalize(img_array)
            
            # Convert back to uint8
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            return Image.fromarray(normalized, 'RGB')
        
        except Exception as e:
            # If normalization fails, return original image
            return img
    
    def _macenko_normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Perform Macenko stain normalization.
        
        Args:
            img: Image array (H, W, 3) in RGB format, float32 [0-255]
        
        Returns:
            Normalized image array
        """
        # Convert RGB to OD (optical density)
        img_rgb = img / 255.0
        
        # Avoid log(0)
        img_rgb = np.clip(img_rgb, 1e-6, 1 - 1e-6)
        
        # Calculate optical density
        od = -np.log(img_rgb)
        
        # Create tissue mask to exclude black border padding
        # Black regions have RGB ≈ [0, 0, 0], which means high OD
        # Use a threshold to identify non-tissue regions
        tissue_mask = np.any(img > 10, axis=2)  # Exclude very dark regions (black padding)
        
        # Calculate stain vectors using Macenko method
        stains, concentrations = self._get_stain_matrix(od, tissue_mask)
        
        # Normalize to reference
        normalized = self._apply_normalization(od, stains, concentrations)
        
        return normalized
    
    def _get_stain_matrix(self, od: np.ndarray, tissue_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get stain matrix from OD image.
        
        Args:
            od: Optical density image
            tissue_mask: Optional binary mask to exclude black border regions
        
        Returns:
            Stain vectors and concentrations
        """
        # Flatten OD image
        h, w, c = od.shape
        od_flat = od.reshape(-1, c)
        
        # Apply tissue mask if provided to exclude black borders
        if tissue_mask is not None:
            mask_flat = tissue_mask.flatten()
            od_tissue = od_flat[mask_flat]
        else:
            od_tissue = od_flat
        
        # Use high-percentile pixels for robust stain extraction
        # These are pixels with strong staining (high OD)
        percentile = 99
        threshold = np.percentile(od_tissue, percentile, axis=0)
        
        # Get pixels above threshold
        mask = np.all(od_tissue > threshold, axis=1)
        if np.sum(mask) < 10:
            # Not enough pixels - use reference stains
            stains = self.reference_stains[:, :2]
            return stains, np.ones(2)
        
        od_pixels = od_tissue[mask]
        
        # Compute SVD on high-OD pixels
        U, S, Vt = np.linalg.svd(od_pixels - np.mean(od_pixels, axis=0), full_matrices=False)
        
        # Get first two principal components (H and E stains)
        stains = Vt[:2, :].T  # Shape: (3, 2)
        
        # Ensure consistent orientation
        if stains[0, 0] < 0:
            stains[:, 0] *= -1
        if stains[1, 0] < 0:
            stains[:, 1] *= -1
        
        # Normalize stain vectors to unit norm
        stains = stains / np.linalg.norm(stains, axis=0)
        
        return stains, np.ones(2)
    
    def _apply_normalization(
        self,
        od: np.ndarray,
        stains: np.ndarray,
        concentrations: np.ndarray
    ) -> np.ndarray:
        """
        Apply normalization to OD image.
        
        Args:
            od: Optical density image
            stains: Stain vectors from image
            concentrations: Stain concentrations (not used in this version)
        
        Returns:
            Normalized RGB image
        """
        h, w, c = od.shape
        od_flat = od.reshape(-1, c)
        
        # Solve for source concentrations in each pixel
        # This gives us how much of each stain is in each pixel
        try:
            source_concentrations = np.linalg.lstsq(stains, od_flat.T, rcond=None)[0]
        except:
            return (np.ones_like(od) * 255).astype(np.float32)
        
        # Clip to valid range (no negative concentrations)
        source_concentrations = np.clip(source_concentrations, 0, None)
        
        # Calculate normalization factors from tissue concentrations
        # Use percentile of each stain concentration
        conc_percentile = np.percentile(source_concentrations, 99, axis=1)
        # Avoid division by zero
        conc_percentile = np.maximum(conc_percentile, 1e-6)
        
        # Get reference concentrations for normalized stains
        ref_stains = self.reference_stains[:, :stains.shape[1]]
        ref_conc = self.ref_concentrations[:stains.shape[1]]
        
        # Normalize concentrations: scale source to match reference intensity
        normalized_concentrations = source_concentrations / conc_percentile[:, None]
        normalized_concentrations = normalized_concentrations * ref_conc[:, None]
        
        # Reconstruct OD from normalized concentrations using reference stains
        od_normalized = ref_stains @ normalized_concentrations
        
        # Clip OD to safe range before exponential (prevents overflow)
        od_normalized = np.clip(od_normalized, -10, 10)
        
        # Convert OD back to RGB: RGB = exp(-OD)
        rgb_normalized = np.exp(-od_normalized)
        
        # Reshape back to image format
        rgb_normalized = rgb_normalized.T.reshape(h, w, c)
        
        # Handle any NaN/inf values from numerical issues
        rgb_normalized = np.nan_to_num(rgb_normalized, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Clip to valid RGB range [0, 1]
        rgb_normalized = np.clip(rgb_normalized, 0, 1)
        
        return (rgb_normalized * 255).astype(np.float32)


# ============================================================================
# PATCH PROCESSING & CUSTOMIZATION
# ============================================================================

class PatchProcessor:
    """Processes and customizes extracted patches."""
    
    def __init__(self, apply_stain_normalization: bool = False):
        """
        Initialize patch processor.
        
        Args:
            apply_stain_normalization: Whether to apply Macenko normalization (disabled by default)
        """
        self.apply_stain_normalization = apply_stain_normalization
        if apply_stain_normalization:
            self.normalizer = MacenkoNormalizer()
        else:
            self.normalizer = None
    
    def apply_custom_processing(
        self,
        patch_img: Image.Image,
        patch_metadata: Dict
    ) -> Image.Image:
        """
        Apply custom processing to patch image.
        
        Includes:
        - Macenko stain normalization
        - Custom augmentations (placeholder)
        
        Args:
            patch_img: PIL Image of the patch
            patch_metadata: Dictionary with patch information
        
        Returns:
            Processed PIL Image
        """
        # Apply Macenko stain normalization
        if self.apply_stain_normalization and self.normalizer:
            patch_img = self.normalizer.normalize(patch_img)
        
        # TODO: Add additional custom processing here
        # Example: Convert to grayscale, apply augmentations, etc.
        
        return patch_img


# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageProcessor:
    """Main processor for extracting patches from a single WSI image."""
    
    def __init__(
        self,
        wsi_path: Path,
        sample_id: str,
        output_dir: Path,
        spatial_loader: Optional[SpatialDataLoader] = None,
        image_name: Optional[str] = None
    ):
        """
        Initialize image processor.
        
        Args:
            wsi_path: Path to WSI TIFF file
            sample_id: Sample identifier (e.g., 'PT_1')
            output_dir: Output directory for patches
            spatial_loader: Optional SpatialDataLoader instance
            image_name: Image name in spatial data (if using spatial loader)
        """
        self.wsi_path = Path(wsi_path)
        self.sample_id = sample_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.spatial_loader = spatial_loader
        self.image_name = image_name
        
        self.extractor = PatchExtractor(patch_size=Config.PATCH_SIZE)
        self.processor = PatchProcessor(apply_stain_normalization=False)
        self.id_manager = PatchIDManager(start_id=1)  # Always start from 1 for each sample
    
    def process(self) -> Dict:
        """
        Main processing pipeline: load image, extract patches, save results.
        
        Returns:
            Metadata dictionary with extraction summary
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING: {self.sample_id}")
        print(f"{'='*80}\n")
        
        # Load WSI
        wsi = self._load_wsi()
        
        # Get spatial information if available
        spot_diameter = None
        image_spots = None
        
        if self.spatial_loader and self.image_name:
            image_spots = self.spatial_loader.get_spots_for_image(self.image_name)
            scale_info = self.spatial_loader.get_scale_info(self.image_name)
            spot_diameter = scale_info['spot_diameter_pixels']
            print(f"Total spots: {len(image_spots)}")
            print(f"Spot diameter: {spot_diameter:.2f} px\n")
        
        # Extract patches
        metadata = self._extract_patches(wsi, image_spots, spot_diameter)
        
        # Save metadata
        self._save_metadata(metadata)
        
        return metadata
    
    def _load_wsi(self) -> np.ndarray:
        """Load WSI TIFF image."""
        print(f"Loading WSI: {self.wsi_path}")
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.wsi_path}")
        
        wsi = tifffile.imread(str(self.wsi_path))
        print(f"Dimensions: {wsi.shape}\n")
        return wsi
    
    def _extract_patches(
        self,
        wsi: np.ndarray,
        image_spots: Optional[pd.DataFrame] = None,
        spot_diameter: Optional[float] = None
    ) -> Dict:
        """
        Extract patches from WSI.
        
        If spatial data is provided, extract at spot locations.
        Otherwise, use a grid-based approach or custom coordinates.
        """
        start_patch_id = self.id_manager.get_current()
        patches = []
        failed = 0
        
        if image_spots is not None and spot_diameter is not None:
            patches, failed = self._extract_from_spots(
                wsi, image_spots, spot_diameter
            )
        else:
            print("No spatial data provided. Using grid-based extraction.")
            patches, failed = self._extract_from_grid(wsi)
        
        # Create metadata
        end_patch_id = self.id_manager.get_current() - 1
        
        metadata = {
            'image_info': {
                'sample_id': self.sample_id,
                'wsi_path': str(self.wsi_path),
                'wsi_dimensions': f"{wsi.shape[1]} x {wsi.shape[0]}",
                'patch_size': Config.PATCH_SIZE,
            },
            'patches': patches,
            'extraction_summary': {
                'start_patch_id': start_patch_id,
                'end_patch_id': end_patch_id,
                'patches_extracted': len(patches),
                'patches_failed': failed,
                'total_attempted': len(patches) + failed
            }
        }
        
        print(f"Extraction complete:")
        print(f"  Extracted: {len(patches)} patches")
        print(f"  Failed: {failed} patches")
        print(f"  ID range: {start_patch_id}-{end_patch_id}\n")
        
        return metadata
    
    def _extract_from_spots(
        self,
        wsi: np.ndarray,
        image_spots: pd.DataFrame,
        spot_diameter: float
    ) -> Tuple[List[Dict], int]:
        """Extract patches at spatial spot locations."""
        patches = []
        failed = 0
        
        print("Extracting patches from spot locations...")
        for idx, (_, spot_row) in enumerate(image_spots.iterrows(), 1):
            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(image_spots)} spots")
            
            try:
                patch_id = self.id_manager.allocate_id()
                
                # Extract coordinates from spatial data
                barcode = spot_row.get('spot_barcode', '')
                tissue_row = int(spot_row.get('row', 0))
                tissue_col = int(spot_row.get('col', 0))
                image_row = int(spot_row.get('imagerow', 0))
                image_col = int(spot_row.get('imagecol', 0))
                
                # Extract patch
                patch_img = self.extractor.extract_circular_patch(
                    wsi, image_row, image_col, spot_diameter
                )
                
                if patch_img is None:
                    failed += 1
                    continue
                
                # Apply custom processing
                patch_img = self.processor.apply_custom_processing(
                    patch_img,
                    {
                        'patch_id': patch_id,
                        'barcode': barcode,
                        'tissue_row': tissue_row,
                        'tissue_col': tissue_col
                    }
                )
                
                # Save patch with new filename format
                patch_filename = (
                    f"{self.sample_id}_patch-{patch_id:06d}_{tissue_row}_{tissue_col}.png"
                )
                patch_path = self.output_dir / patch_filename
                patch_img.save(str(patch_path), Config.PATCH_FORMAT)
                
                # Record metadata
                patches.append({
                    'patch_id': patch_id,
                    'barcode': barcode,
                    'tissue_row': tissue_row,
                    'tissue_col': tissue_col,
                    'image_row': image_row,
                    'image_col': image_col,
                    'patch_filename': patch_filename
                })
                
            except Exception as e:
                failed += 1
                if idx <= 3:
                    print(f"  Error on spot {idx}: {e}")
        
        return patches, failed
    
    def _extract_from_grid(
        self,
        wsi: np.ndarray,
        stride: int = 224
    ) -> Tuple[List[Dict], int]:
        """
        Extract patches using grid-based sampling (fallback).
        
        PLACEHOLDER: Modify stride and extraction logic as needed.
        """
        patches = []
        failed = 0
        
        print(f"Extracting patches on {stride}px grid...")
        spot_diameter = stride
        
        for row in range(stride // 2, wsi.shape[0], stride):
            for col in range(stride // 2, wsi.shape[1], stride):
                try:
                    patch_id = self.id_manager.allocate_id()
                    
                    patch_img = self.extractor.extract_circular_patch(
                        wsi, row, col, spot_diameter
                    )
                    
                    if patch_img is None:
                        failed += 1
                        continue
                    
                    # Apply custom processing
                    patch_img = self.processor.apply_custom_processing(
                        patch_img,
                        {'patch_id': patch_id}
                    )
                    
                    # Save with new filename format
                    patch_filename = f"{self.sample_id}_patch-{patch_id:06d}_{row}_{col}.png"
                    patch_path = self.output_dir / patch_filename
                    patch_img.save(str(patch_path), Config.PATCH_FORMAT)
                    
                    patches.append({
                        'patch_id': patch_id,
                        'row': row,
                        'col': col,
                        'patch_filename': patch_filename
                    })
                    
                except Exception as e:
                    failed += 1
        
        return patches, failed
    
    def _save_metadata(self, metadata: Dict):
        """Save extraction metadata to JSON."""
        metadata_file = self.output_dir / "patches_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"[ok] Metadata saved: {metadata_file}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # `python create_patches.py SAMPLE [SAMPLE ...]`, or "all" for every ready
        # sample that has no patches yet.  Falls back to SELECTED_IMAGE.
        argv = sys.argv[1:]
        if argv == ["all"]:
            # Count actual PNGs -- a failed run leaves an empty directory behind, so
            # "directory exists" is not evidence that the sample was patched.
            def n_patches(s):
                d = Config.OUTPUT_BASE_DIR / s
                return len(list(d.glob("*.png"))) if d.is_dir() else 0

            expected = {}
            spots_csv = Config.ST_DIR / "spot_spatial_coordinates.csv"
            if spots_csv.exists():
                sp = pd.read_csv(spots_csv)
                expected = sp.groupby("image").size().to_dict()
            todo = [s for s in sorted(IMAGE_FILEPATHS)
                    if n_patches(s) < expected.get(s, 1)]
            for s in todo:
                print(f"  {s}: {n_patches(s)}/{expected.get(s, '?')} patches -> (re)build")
            print(f"Samples to patch: {todo}")
        elif argv:
            todo = argv
        else:
            todo = [SELECTED_IMAGE]

        if len(todo) != 1:
            # subprocess, not os.system: the repo path contains spaces and cmd.exe
            # mangles the quoting ("The filename, directory name, or volume label
            # syntax is incorrect").
            import subprocess
            failed = []
            for s in todo:
                rc = subprocess.call([sys.executable, __file__, s])
                if rc != 0:
                    failed.append(s)
                    print(f"  {s}: FAILED (exit {rc})")
            print(f"\nBatch done: {len(todo) - len(failed)}/{len(todo)} succeeded"
                  + (f", failed: {failed}" if failed else ""))
            sys.exit(1 if failed else 0)
        SELECTED_IMAGE = todo[0]

        # Validate selected image
        if SELECTED_IMAGE not in IMAGE_FILEPATHS:
            raise ValueError(
                f"Invalid image '{SELECTED_IMAGE}'. "
                f"Valid options: {sorted(IMAGE_FILEPATHS.keys())}"
            )

        # Get image configuration
        wsi_path = IMAGE_FILEPATHS[SELECTED_IMAGE]
        image_config = IMAGE_CONFIGS[SELECTED_IMAGE]
        sample_id = image_config['image_name']
        image_name = image_config['image_name']
        patient_id = image_config['patient']
        
        # Create output directory: Outputs/Patches/{IU_PDA_T1}
        output_dir = Config.OUTPUT_BASE_DIR / sample_id
        
        # Initialize spatial loader if data files exist
        spatial_loader = None
        spots_csv = Config.ST_DIR / "spot_spatial_coordinates.csv"
        scale_csv = Config.ST_DIR / "visium_scale_information.csv"
        
        if spots_csv.exists() and scale_csv.exists():
            try:
                spatial_loader = SpatialDataLoader(spots_csv, scale_csv)
            except Exception as e:
                print(f"Warning: Could not load spatial data: {e}")
        
        # Create processor
        processor = ImageProcessor(
            wsi_path=wsi_path,
            sample_id=sample_id,
            output_dir=output_dir,
            spatial_loader=spatial_loader,
            image_name=image_name
        )
        
        # Process image
        print(f"\n{'='*80}")
        print(f"PATCH CREATION SYSTEM")
        print(f"{'='*80}")
        print(f"Selected Image: {SELECTED_IMAGE}")
        print(f"Sample ID: {sample_id}")
        print(f"Patient ID: {patient_id}")
        print(f"Output Directory: {output_dir}\n")
        
        metadata = processor.process()
        
        print(f"[ok] Patch creation completed successfully!")
        print(f"  Total patches: {metadata['extraction_summary']['patches_extracted']}")
        sys.exit(0)
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
