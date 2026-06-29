"""
Modular WSI Patch Overlay Creation System
==========================================
Creates circular patch overlays on WSI images with labels.
Each circular boundary shows: patch_id, tissue_row, tissue_col
Saves as TIFF with green circular boundaries.

Configuration:
  - Modify IMAGE_FILEPATHS and SELECTED_IMAGE to choose which image to process
  - Customize circle appearance in Config class
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tifffile
from PIL import Image, ImageDraw, ImageFont


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

# SELECT WHICH IMAGE TO PROCESS HERE
SELECTED_IMAGE = 'IU_PDA_HM13'  # Change to 'IU_PDA_T4' or 'IU_PDA_T11'


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration for overlay creation parameters."""
    
    # Default directories
    WORKSPACE_DIR = Path(r"D:\Zenodo PT & HM Dataset Work")
    ST_DIR = WORKSPACE_DIR / "Outputs" / "Patient-Sample-Information"
    PATCHES_BASE_DIR = WORKSPACE_DIR / "Outputs" / "Patches" / ".png patches"
    OVERLAY_BASE_DIR = WORKSPACE_DIR / "Outputs" / "Patches" / "patch overlay"
    
    # Circle appearance
    CIRCLE_COLOR = (0, 255, 0)  # Green
    CIRCLE_WIDTH = 3
    
    # Text appearance
    TEXT_COLOR = (0, 255, 0)  # Green
    TEXT_SIZE = 12  # Pixel size for text (will be adjusted based on zoom level)
    
    # Output
    OVERLAY_FILENAME = "{sample_id}_overlay.tiff"


# ============================================================================
# DATA LOADING
# ============================================================================

class MetadataLoader:
    """Loads patch metadata from JSON file."""
    
    def __init__(self, metadata_path: Path):
        """
        Initialize metadata loader.
        
        Args:
            metadata_path: Path to patches_metadata.json
        """
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
    
    def get_scale_info(self) -> Optional[float]:
        """Get spot diameter if available in metadata."""
        image_info = self.metadata.get('image_info', {})
        return image_info.get('spot_diameter_pixels')


class SpatialDataLoader:
    """Loads scale information from CSV files."""
    
    def __init__(self, scale_csv: Path):
        """
        Initialize loader with scale CSV path.
        
        Args:
            scale_csv: Path to visium_scale_information.csv
        """
        self.scale_csv = scale_csv
        self.scale_df = None
        self._load_data()
    
    def _load_data(self):
        """Load scale CSV file."""
        try:
            self.scale_df = pd.read_csv(self.scale_csv)
            print(f"✓ Loaded scale info for {len(self.scale_df)} images\n")
        except Exception as e:
            raise RuntimeError(f"Failed to load scale data: {e}")
    
    def get_scale_info(self, image_name: str) -> Optional[float]:
        """Get spot diameter for a specific image."""
        scale_row = self.scale_df[self.scale_df['image_name'] == image_name]
        if len(scale_row) == 0:
            return None
        return scale_row.iloc[0].get('spot_diameter_pixels')


# ============================================================================
# OVERLAY CREATION
# ============================================================================

class OverlayCreator:
    """Creates circular patch overlays on WSI images."""
    
    def __init__(self, circle_color: tuple = (0, 255, 0), circle_width: int = 3):
        """
        Initialize overlay creator.
        
        Args:
            circle_color: RGB tuple for circle color
            circle_width: Width of circle boundary in pixels
        """
        self.circle_color = circle_color
        self.circle_width = circle_width
    
    def create_overlay(
        self,
        wsi: np.ndarray,
        patches: List[Dict],
        spot_diameter: float,
        output_path: Path
    ):
        """
        Create overlay with circular patch boundaries and labels.
        
        Args:
            wsi: WSI image array
            patches: List of patch metadata dictionaries
            spot_diameter: Diameter of circular patches in pixels
            output_path: Path to save overlay TIFF
        """
        print("Creating overlay with circular patch boundaries...")
        
        # Convert WSI to PIL Image
        if len(wsi.shape) == 3:
            pil_image = Image.fromarray(wsi)
        else:
            pil_image = Image.fromarray(wsi).convert('RGB')
        
        draw = ImageDraw.Draw(pil_image)
        radius = spot_diameter / 2
        
        # Try to load font, fallback to default if not available
        try:
            font = ImageFont.truetype(
                "C:/Windows/Fonts/arial.ttf",
                Config.TEXT_SIZE
            )
        except:
            font = ImageFont.load_default()
        
        # Draw circles and labels for each patch
        for patch_info in patches:
            if 'image_col' not in patch_info:
                continue
            
            patch_id = patch_info.get('patch_id', '')
            tissue_row = patch_info.get('tissue_row', '')
            tissue_col = patch_info.get('tissue_col', '')
            image_row = patch_info['image_row']
            image_col = patch_info['image_col']
            
            # Draw circular boundary
            x0 = image_col - radius
            y0 = image_row - radius
            x1 = image_col + radius
            y1 = image_row + radius
            
            draw.ellipse(
                [x0, y0, x1, y1],
                outline=self.circle_color,
                width=self.circle_width
            )
            
            # Draw text label inside circle
            label = f"P{patch_id}\nR{tissue_row}\nC{tissue_col}"
            
            # Get text bounding box to center it
            bbox = draw.textbbox(
                (image_col, image_row - 15),
                label,
                font=font
            )
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Center text above circle
            text_x = image_col - text_width // 2
            text_y = image_row - radius - text_height - 5
            
            draw.text(
                (text_x, text_y),
                label,
                fill=self.circle_color,
                font=font
            )
        
        # Save overlay
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(str(output_path), 'TIFF', compression=None)
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✓ Overlay saved: {output_path} ({file_size_mb:.2f} MB)\n")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class OverlayPipeline:
    """Main pipeline for creating patch overlays."""
    
    def __init__(
        self,
        wsi_path: Path,
        sample_id: str,
        metadata_path: Path,
        output_dir: Path
    ):
        """
        Initialize overlay pipeline.
        
        Args:
            wsi_path: Path to WSI TIFF file
            sample_id: Sample identifier
            metadata_path: Path to patches_metadata.json
            output_dir: Output directory for overlay
        """
        self.wsi_path = Path(wsi_path)
        self.sample_id = sample_id
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        
        self.metadata_loader = MetadataLoader(self.metadata_path)
        self.overlay_creator = OverlayCreator(
            circle_color=Config.CIRCLE_COLOR,
            circle_width=Config.CIRCLE_WIDTH
        )
    
    def run(self):
        """Main processing pipeline."""
        print(f"\n{'='*80}")
        print(f"OVERLAY CREATION SYSTEM")
        print(f"{'='*80}")
        print(f"Sample ID: {self.sample_id}")
        print(f"WSI Path: {self.wsi_path}")
        print(f"Metadata Path: {self.metadata_path}\n")
        
        # Load WSI
        wsi = self._load_wsi()
        
        # Get patches metadata
        patches = self.metadata_loader.get_patches()
        
        # Get spot diameter
        spot_diameter = self._get_spot_diameter()
        
        if not spot_diameter:
            raise ValueError("Cannot determine spot diameter. Check metadata or scale CSV.")
        
        print(f"Spot diameter: {spot_diameter:.2f} px")
        print(f"Number of patches: {len(patches)}\n")
        
        # Create overlay
        overlay_path = self.output_dir / f"{self.sample_id}_overlay.tiff"
        self.overlay_creator.create_overlay(wsi, patches, spot_diameter, overlay_path)
        
        print(f"✓ Overlay creation completed successfully!")
    
    def _load_wsi(self) -> np.ndarray:
        """Load WSI TIFF image."""
        print(f"Loading WSI: {self.wsi_path}")
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {self.wsi_path}")
        
        wsi = tifffile.imread(str(self.wsi_path))
        print(f"Dimensions: {wsi.shape}\n")
        return wsi
    
    def _get_spot_diameter(self) -> Optional[float]:
        """Get spot diameter from metadata or spatial data."""
        # Try to get from metadata first
        spot_diameter = self.metadata_loader.get_scale_info()
        if spot_diameter:
            return spot_diameter
        
        # Try to get from spatial data CSV
        scale_csv = Config.ST_DIR / "visium_scale_information.csv"
        if scale_csv.exists():
            try:
                spatial_loader = SpatialDataLoader(scale_csv)
                spot_diameter = spatial_loader.get_scale_info(self.sample_id)
                if spot_diameter:
                    return spot_diameter
            except:
                pass
        
        return None


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
        
        # Paths
        patches_metadata_dir = Config.PATCHES_BASE_DIR / sample_id
        metadata_path = patches_metadata_dir / "patches_metadata.json"
        overlay_output_dir = Config.OVERLAY_BASE_DIR / sample_id
        
        # Create pipeline
        pipeline = OverlayPipeline(
            wsi_path=wsi_path,
            sample_id=sample_id,
            metadata_path=metadata_path,
            output_dir=overlay_output_dir
        )
        
        # Run pipeline
        pipeline.run()
        sys.exit(0)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
