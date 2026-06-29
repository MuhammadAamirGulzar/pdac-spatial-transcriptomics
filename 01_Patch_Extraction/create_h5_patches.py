"""
Modular H5 Patch Archive Creation System
=========================================
Creates HDF5 archives containing patches and metadata for foundational model processing.
Includes WSI reference, tissue mask, patch parameters, and patient metadata.

Configuration:
  - Modify IMAGE_FILEPATHS and SELECTED_IMAGE to choose which image to process
  - Update IMAGE_CONFIGS and PATIENT_INFO for your dataset
  - Customize H5 structure in Config class
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
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

# Patient information mapping
PATIENT_INFO = {
    'PT_1': {
        'sample_type': 'PT',  # PT (Patient Tissue) or HM (Healthy Margin)
        'patient_id': 'PT_1',
    },
    'PT_4': {
        'sample_type': 'PT',
        'patient_id': 'PT_4',
    },
    'PT_11': {
        'sample_type': 'PT',
        'patient_id': 'PT_11',
    },
    'HM_11': {
        'sample_type': 'HM',
        'patient_id': 'HM_11',
    },
    'HM_13': {
        'sample_type': 'HM',
        'patient_id': 'HM_13',
    },
    'PT_3': {
        'sample_type': 'PT',
        'patient_id': 'PT_3',
    },
}

# SELECT WHICH IMAGE TO PROCESS HERE
SELECTED_IMAGE = 'IU_PDA_T3'  # Change to 'IU_PDA_T4' or 'IU_PDA_T11'


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for H5 file creation."""
    
    # Patch parameters
    PATCH_SIZE = 224
    
    # Default directories
    WORKSPACE_DIR = Path(r"D:\Zenodo PT & HM Dataset Work")
    ST_DIR = WORKSPACE_DIR / "Outputs" / "Patient-Sample-Information"
    PATCHES_BASE_DIR = WORKSPACE_DIR / "Outputs" / "Patches" / ".png patches"
    H5_OUTPUT_BASE_DIR = WORKSPACE_DIR / "Outputs" / "Patches" / "h5 patches"
    
    # H5 file settings
    H5_FILENAME = "{sample_id}_patches.h5"
    COMPRESSION = 'gzip'
    COMPRESSION_LEVEL = 4


# ============================================================================
# DATA LOADING
# ============================================================================

class MetadataLoader:
    """Loads patch metadata from JSON file."""
    
    def __init__(self, metadata_path: Path):
        """Initialize metadata loader."""
        self.metadata_path = metadata_path
        self.metadata = None
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata JSON file."""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")
        
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
            print(f"✓ Loaded metadata with {len(self.metadata['patches'])} patches\n")
        except Exception as e:
            raise RuntimeError(f"Failed to load metadata: {e}")
    
    def get_patches(self) -> List[Dict]:
        """Get list of all patches."""
        return self.metadata.get('patches', [])
    
    def get_image_info(self) -> Dict:
        """Get image information."""
        return self.metadata.get('image_info', {})
    
    def get_extraction_summary(self) -> Dict:
        """Get extraction summary."""
        return self.metadata.get('extraction_summary', {})


class SpatialDataLoader:
    """Loads spatial coordinates and scale information from CSV files."""
    
    def __init__(self, spots_csv: Path, scale_csv: Path):
        """Initialize spatial data loader."""
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
            print(f"✓ Loaded {len(self.spots_df)} spot coordinates")
            print(f"✓ Loaded scale info for {len(self.scale_df)} images\n")
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
    
    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Apply Macenko stain normalization to image array.
        
        Args:
            img: Image array (H, W, 3) in RGB format, uint8 [0-255]
        
        Returns:
            Normalized image array (H, W, 3) uint8
        """
        try:
            # Convert to float
            img_array = img.astype(np.float32)
            
            # Normalize
            normalized = self._macenko_normalize(img_array)
            
            # Convert back to uint8
            normalized = np.clip(normalized, 0, 255).astype(np.uint8)
            
            return normalized
        
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
# TISSUE MASK GENERATION
# ============================================================================

class TissueMaskGenerator:
    """Generates binary tissue masks from WSI images."""
    
    def __init__(self):
        pass
    
    def generate_tissue_mask(self, wsi: np.ndarray, scale: int = 32) -> np.ndarray:
        """Generate binary tissue mask from WSI."""
        # Downsample for faster processing
        thumb = wsi[::scale, ::scale]
        gray = np.mean(thumb, axis=2) if len(thumb.shape) == 3 else thumb
        mask = (gray < 200).astype(np.uint8)
        return mask


# ============================================================================
# H5 FILE CREATION
# ============================================================================

class H5Creator:
    """Creates HDF5 archives with patches and metadata."""
    
    def __init__(self, apply_stain_normalization: bool = False, apply_wsi_storage: bool = False):
        """
        Initialize H5 creator.
        
        Args:
            apply_stain_normalization: Whether to apply Macenko normalization (disabled by default for stability)
            apply_wsi_storage: Whether to store full WSI image (default False to save space)
        """
        self.apply_stain_normalization = apply_stain_normalization
        self.apply_wsi_storage = apply_wsi_storage
        if apply_stain_normalization:
            self.normalizer = MacenkoNormalizer()
        else:
            self.normalizer = None
    
    def create_h5_archive(
        self,
        output_path: Path,
        wsi: np.ndarray,
        tissue_mask: np.ndarray,
        patches: List[Dict],
        patch_parameters: Dict,
        patient_metadata: Dict,
        image_metadata: Dict,
        png_patches_dir: Optional[Path] = None
    ):
        """
        Create HDF5 archive with patches and metadata.
        
        Args:
            output_path: Path to save H5 file
            wsi: WSI image array
            tissue_mask: Binary tissue mask
            patches: List of patch metadata
            patch_parameters: Dictionary with patch configuration
            patient_metadata: Dictionary with patient information
            image_metadata: Dictionary with image information
            png_patches_dir: Path to directory containing PNG patch files
        """
        print("Creating H5 archive...")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(str(output_path), 'w') as h5file:
            # Create main groups
            if self.apply_wsi_storage:
                h5file.create_group('wsi')
            h5file.create_group('tissue_mask')
            h5file.create_group('patch_parameters')
            h5file.create_group('patient_metadata')
            h5file.create_group('image_metadata')
            h5file.create_group('patches')
            
            # Store WSI (optional, only if enabled)
            if self.apply_wsi_storage:
                self._store_wsi(h5file['wsi'], wsi)
            
            # Store tissue mask
            self._store_tissue_mask(h5file['tissue_mask'], tissue_mask)
            
            # Store patch parameters
            self._store_patch_parameters(h5file['patch_parameters'], patch_parameters)
            
            # Store patient metadata
            self._store_patient_metadata(h5file['patient_metadata'], patient_metadata)
            
            # Store image metadata
            self._store_image_metadata(h5file['image_metadata'], image_metadata)
            
            # Store patch metadata and pixel data
            self._store_patches_metadata(h5file['patches'], patches)
            if png_patches_dir:
                self._store_patch_pixels(h5file['patches'], patches, png_patches_dir)
            
            # Add creation timestamp
            h5file.attrs['created'] = datetime.now().isoformat()
            h5file.attrs['total_patches'] = len(patches)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ H5 archive saved: {output_path} ({file_size_mb:.2f} MB)\n")
    
    # CORRECT — store raw WSI, normalize per-patch only
    def _store_wsi(self, group: h5py.Group, wsi: np.ndarray):
        print("  Storing WSI image (raw)...")
        wsi_to_store = wsi if wsi.dtype == np.uint8 else \
                    np.clip(wsi / wsi.max() * 255, 0, 255).astype(np.uint8)
        
        group.create_dataset(
            'image',
            data=wsi_to_store,
            compression=Config.COMPRESSION,
            compression_opts=Config.COMPRESSION_LEVEL
        )
        group['image'].attrs['shape'] = wsi_to_store.shape
        group['image'].attrs['dtype'] = str(wsi_to_store.dtype)
        group['image'].attrs['stain_normalized'] = False  # normalized at patch level
    
    def _store_tissue_mask(self, group: h5py.Group, mask: np.ndarray):
        """Store binary tissue mask."""
        print("  Storing tissue mask...")
        group.create_dataset(
            'mask',
            data=mask,
            compression=Config.COMPRESSION,
            compression_opts=Config.COMPRESSION_LEVEL
        )
        group['mask'].attrs['shape'] = mask.shape
        group['mask'].attrs['dtype'] = str(mask.dtype)
        group['mask'].attrs['description'] = '1 = tissue, 0 = background'
        group['mask'].attrs['downsample_factor'] = 32
    
    def _store_patch_parameters(self, group: h5py.Group, params: Dict):
        """Store patch extraction parameters."""
        print("  Storing patch parameters...")
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
            elif isinstance(value, dict):
                subgroup = group.create_group(key)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, str, bool)):
                        subgroup.attrs[sub_key] = sub_value
    
    def _store_patient_metadata(self, group: h5py.Group, metadata: Dict):
        """Store patient metadata as attributes."""
        print("  Storing patient metadata...")
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
    
    def _store_image_metadata(self, group: h5py.Group, metadata: Dict):
        """Store image metadata as attributes."""
        print("  Storing image metadata...")
        for key, value in metadata.items():
            if isinstance(value, (int, float, str, bool)):
                group.attrs[key] = value
    
    def _store_patches_metadata(self, group: h5py.Group, patches: List[Dict]):
        """Store patch metadata."""
        print(f"  Storing {len(patches)} patch metadata entries...")
        
        # Create arrays for key fields
        patch_ids = []
        tissue_rows = []
        tissue_cols = []
        image_rows = []
        image_cols = []
        barcodes = []
        filenames = []
        
        for patch in patches:
            patch_ids.append(patch.get('patch_id', 0))
            tissue_rows.append(patch.get('tissue_row', 0))
            tissue_cols.append(patch.get('tissue_col', 0))
            image_rows.append(patch.get('image_row', 0))
            image_cols.append(patch.get('image_col', 0))
            barcodes.append(patch.get('barcode', ''))
            filenames.append(patch.get('patch_filename', ''))
        
        # Store as datasets
        group.create_dataset('patch_id', data=np.array(patch_ids), compression=Config.COMPRESSION)
        group.create_dataset('tissue_row', data=np.array(tissue_rows), compression=Config.COMPRESSION)
        group.create_dataset('tissue_col', data=np.array(tissue_cols), compression=Config.COMPRESSION)
        group.create_dataset('image_row', data=np.array(image_rows), compression=Config.COMPRESSION)
        group.create_dataset('image_col', data=np.array(image_cols), compression=Config.COMPRESSION)
        group.create_dataset('barcode', data=np.array(barcodes, dtype=h5py.string_dtype()), compression=Config.COMPRESSION)
        group.create_dataset('filename', data=np.array(filenames, dtype=h5py.string_dtype()), compression=Config.COMPRESSION)
    
    def _store_patch_pixels(self, group: h5py.Group, patches: List[Dict], png_patches_dir: Path):
        """Load PNG patch files and store pixel arrays in H5."""
        print(f"  Loading and storing {len(patches)} patch pixel arrays...")
        
        patch_size = Config.PATCH_SIZE
        pixels_list = []
        loaded_count = 0
        missing_count = 0
        
        for patch in patches:
            patch_filename = patch.get('patch_filename', '')
            png_path = Path(png_patches_dir) / patch_filename
            
            if png_path.exists():
                try:
                    # Load PNG and convert to RGB array
                    img = Image.open(png_path).convert('RGB')
                    img_array = np.array(img, dtype=np.uint8)
                    pixels_list.append(img_array)
                    loaded_count += 1
                except Exception as e:
                    # On error, append black patch as placeholder
                    pixels_list.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                    missing_count += 1
                    if missing_count <= 3:
                        print(f"    Warning: Could not load {png_path}: {e}")
            else:
                # File not found, append black patch as placeholder
                pixels_list.append(np.zeros((patch_size, patch_size, 3), dtype=np.uint8))
                missing_count += 1
                if missing_count <= 3:
                    print(f"    Warning: PNG file not found: {png_path}")
        
        # Stack into single array (N, H, W, 3)
        if pixels_list:
            pixels_array = np.stack(pixels_list, axis=0)
            
            # Store as dataset
            group.create_dataset(
                'pixels',
                data=pixels_array,
                compression=Config.COMPRESSION,
                compression_opts=Config.COMPRESSION_LEVEL
            )
            group['pixels'].attrs['shape'] = pixels_array.shape
            group['pixels'].attrs['dtype'] = str(pixels_array.dtype)
            group['pixels'].attrs['patch_size'] = patch_size
            print(f"    ✓ Stored {loaded_count} patch pixels ({patch_size}×{patch_size}×3)")
            if missing_count > 0:
                print(f"    (Note: {missing_count} patches missing, stored as black placeholders)")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class H5ArchivePipeline:
    """Main pipeline for creating H5 patch archives."""
    
    def __init__(
        self,
        wsi_path: Path,
        sample_id: str,
        patient_id: str,
        sample_type: str,
        metadata_path: Path,
        output_dir: Path,
        spatial_loader: Optional[SpatialDataLoader] = None
    ):
        """Initialize H5 archive pipeline."""
        self.wsi_path = Path(wsi_path)
        self.sample_id = sample_id
        self.patient_id = patient_id
        self.sample_type = sample_type
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.spatial_loader = spatial_loader
        
        self.metadata_loader = MetadataLoader(self.metadata_path)
        self.mask_generator = TissueMaskGenerator()
        self.h5_creator = H5Creator(apply_stain_normalization=False)
    
    def run(self):
        """Main processing pipeline."""
        print(f"\n{'='*80}")
        print(f"H5 PATCH ARCHIVE CREATION SYSTEM")
        print(f"{'='*80}")
        print(f"Sample ID: {self.sample_id}")
        print(f"Patient ID: {self.patient_id}")
        print(f"Sample Type: {self.sample_type}")
        print(f"WSI Path: {self.wsi_path}\n")
        
        # Load WSI
        wsi = self._load_wsi()
        
        # Generate tissue mask
        tissue_mask = self.mask_generator.generate_tissue_mask(wsi)
        
        # Load patch metadata
        patches = self.metadata_loader.get_patches()
        image_info = self.metadata_loader.get_image_info()
        extraction_summary = self.metadata_loader.get_extraction_summary()
        
        # Get scale information
        spot_diameter = None
        magnification = None
        if self.spatial_loader:
            try:
                scale_info = self.spatial_loader.get_scale_info(self.sample_id)
                spot_diameter = scale_info.get('spot_diameter_pixels')
                magnification = scale_info.get('magnification', 20)
            except:
                magnification = 20  # Default
        
        if magnification is None:
            magnification = 20  # Default magnification
        
        # Prepare metadata dictionaries
        patch_parameters = {
            'patch_size': Config.PATCH_SIZE,
            'spot_diameter_pixels': spot_diameter,
            'magnification_level': magnification,
            'total_patches': len(patches),
            'extraction_summary': {
                'patches_extracted': extraction_summary.get('patches_extracted', 0),
                'patches_failed': extraction_summary.get('patches_failed', 0),
            }
        }
        
        patient_metadata = {
            'patient_id': self.patient_id,
            'sample_type': self.sample_type,
            'sample_id': self.sample_id,
        }
        
        # Create H5 archive
        h5_path = self.output_dir / Config.H5_FILENAME.format(sample_id=self.sample_id)
        png_patches_dir = Config.PATCHES_BASE_DIR / self.sample_id
        self.h5_creator.create_h5_archive(
            h5_path,
            wsi,
            tissue_mask,
            patches,
            patch_parameters,
            patient_metadata,
            image_info,
            png_patches_dir=png_patches_dir
        )
        
        print(f"✓ H5 archive creation completed successfully!")
    
    def _load_wsi(self) -> np.ndarray:
        """Load WSI TIFF image."""
        print(f"Loading WSI: {self.wsi_path}")
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.wsi_path}")
        
        wsi = tifffile.imread(str(self.wsi_path))
        print(f"Dimensions: {wsi.shape}\n")
        return wsi


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    try:
        # Validate selected image
        if SELECTED_IMAGE not in IMAGE_FILEPATHS:
            raise ValueError(
                f"Invalid image '{SELECTED_IMAGE}'. "
                f"Valid options: {list(IMAGE_FILEPATHS.keys())}"
            )
        
        # Get image configuration
        wsi_path = IMAGE_FILEPATHS[SELECTED_IMAGE]
        image_config = IMAGE_CONFIGS[SELECTED_IMAGE]
        sample_id = image_config['image_name']
        patient_id = image_config['patient']
        
        # Get patient information
        patient_info = PATIENT_INFO.get(patient_id, {})
        sample_type = patient_info.get('sample_type', 'PT')
        
        # Paths
        patches_metadata_dir = Config.PATCHES_BASE_DIR / sample_id
        metadata_path = patches_metadata_dir / "patches_metadata.json"
        h5_output_dir = Config.H5_OUTPUT_BASE_DIR / sample_id
        
        # Initialize spatial loader if data files exist
        spatial_loader = None
        spots_csv = Config.ST_DIR / "spot_spatial_coordinates.csv"
        scale_csv = Config.ST_DIR / "visium_scale_information.csv"
        
        if spots_csv.exists() and scale_csv.exists():
            try:
                spatial_loader = SpatialDataLoader(spots_csv, scale_csv)
            except Exception as e:
                print(f"Warning: Could not load spatial data: {e}")
        
        # Create pipeline
        pipeline = H5ArchivePipeline(
            wsi_path=wsi_path,
            sample_id=sample_id,
            patient_id=patient_id,
            sample_type=sample_type,
            metadata_path=metadata_path,
            output_dir=h5_output_dir,
            spatial_loader=spatial_loader
        )
        
        # Run pipeline
        pipeline.run()
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
