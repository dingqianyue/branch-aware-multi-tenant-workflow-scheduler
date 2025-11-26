# ============================================
# FILE: workflow-backend/app/workers/smart_tiling.py
# ============================================
"""
Smart Tiling with Tissue Detection and Coordinate Tracking
Key Features:
- Tissue detection to skip background tiles (60-70% speedup)
- Precise coordinate tracking for each tile
- Gaussian blending for seamless merging
- Support for various WSI formats (SVS, NDPI, etc.)
"""

import os
import json
import logging
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from pathlib import Path

# Try importing OpenSlide
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    openslide = None
    logging.warning("OpenSlide not available. WSI processing will be limited.")

# Image processing libraries
try:
    from skimage import filters, morphology
    import cv2
    IMAGING_AVAILABLE = True
except ImportError:
    IMAGING_AVAILABLE = False
    logging.warning("Image processing libraries not available")

logger = logging.getLogger(__name__)


class SmartTiler:
    """
    Intelligent tiling strategy for whole-slide images with coordinate tracking
    
    Key Features:
    1. Tissue detection at thumbnail level (fast)
    2. Skip background tiles (60-70% speedup)
    3. Precise coordinate tracking for each tile
    4. Overlap with Gaussian blending (seamless merging)
    5. Memory-efficient processing
    """
    
    def __init__(self, 
                 tile_size: int = 1024, 
                 overlap: int = 128, 
                 min_tissue_ratio: float = 0.1,
                 thumbnail_size: int = 2000):
        """
        Initialize SmartTiler
        
        Args:
            tile_size: Size of each tile in pixels
            overlap: Overlap between adjacent tiles
            min_tissue_ratio: Minimum tissue ratio to include tile (0-1)
            thumbnail_size: Size of thumbnail for tissue detection
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_tissue_ratio = min_tissue_ratio
        self.thumbnail_size = thumbnail_size
        
        # Validate parameters
        if tile_size <= 0:
            raise ValueError("tile_size must be positive")
        if overlap >= tile_size:
            raise ValueError("overlap must be less than tile_size")
        if not 0 <= min_tissue_ratio <= 1:
            raise ValueError("min_tissue_ratio must be between 0 and 1")
        
        logger.info(f"SmartTiler initialized: tile_size={tile_size}, overlap={overlap}, "
                   f"min_tissue={min_tissue_ratio}")
    
    def is_wsi_supported(self, wsi_path: str) -> bool:
        """Check if WSI file is supported"""
        if not OPENSLIDE_AVAILABLE:
            return False
        
        try:
            # Try to open with OpenSlide
            slide = openslide.OpenSlide(wsi_path)
            slide.close()
            return True
        except:
            return False
    
    def get_wsi_info(self, wsi_path: str) -> Dict:
        """
        Get WSI metadata and properties
        
        Returns:
            Dict with WSI information
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required for WSI processing")
        
        if not os.path.exists(wsi_path):
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")
        
        slide = openslide.OpenSlide(wsi_path)
        
        info = {
            'path': wsi_path,
            'dimensions': slide.dimensions,  # (width, height)
            'level_count': slide.level_count,
            'level_dimensions': slide.level_dimensions,
            'level_downsamples': slide.level_downsamples,
            'properties': dict(slide.properties),
            'vendor': slide.properties.get(openslide.PROPERTY_NAME_VENDOR, 'Unknown'),
            'mpp_x': slide.properties.get(openslide.PROPERTY_NAME_MPP_X, None),
            'mpp_y': slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, None),
        }
        
        # Calculate file size
        info['file_size_mb'] = os.path.getsize(wsi_path) / (1024 * 1024)
        
        slide.close()
        return info
    
    def detect_tissue_fast(self, wsi_path: str, 
                          return_visualization: bool = False) -> Tuple[np.ndarray, Tuple[int, int], Optional[np.ndarray]]:
        """
        Fast tissue detection using thumbnail
        
        Strategy:
        1. Load low-resolution thumbnail (fast)
        2. Convert to grayscale
        3. Apply Otsu thresholding
        4. Morphological operations to clean up
        
        Args:
            wsi_path: Path to WSI file
            return_visualization: If True, return RGB visualization
        
        Returns:
            (tissue_mask, full_dimensions, visualization)
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required")
        
        logger.info(f"Detecting tissue in: {wsi_path}")
        slide = openslide.OpenSlide(wsi_path)
        
        # Get thumbnail
        thumbnail = slide.get_thumbnail((self.thumbnail_size, self.thumbnail_size))
        thumb_array = np.array(thumbnail)
        logger.info(f"Thumbnail size: {thumbnail.size}, Full size: {slide.dimensions}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(thumb_array, cv2.COLOR_RGB2GRAY) if IMAGING_AVAILABLE else np.mean(thumb_array, axis=2)
        
        # Apply Otsu thresholding
        tissue_mask = self._apply_tissue_detection(gray)
        
        # Clean up with morphological operations
        if IMAGING_AVAILABLE:
            # Remove small noise
            tissue_mask = morphology.remove_small_objects(tissue_mask, min_size=50)
            # Fill small holes
            tissue_mask = morphology.remove_small_holes(tissue_mask, area_threshold=100)
        
        tissue_percentage = tissue_mask.mean() * 100
        logger.info(f"Tissue coverage: {tissue_percentage:.1f}%")
        
        # Create visualization if requested
        visualization = None
        if return_visualization:
            visualization = self._create_tissue_visualization(thumb_array, tissue_mask)
        
        full_dimensions = slide.dimensions
        slide.close()
        
        return tissue_mask, full_dimensions, visualization
    
    def _apply_tissue_detection(self, gray: np.ndarray) -> np.ndarray:
        """Apply tissue detection algorithm"""
        if IMAGING_AVAILABLE:
            try:
                # Otsu thresholding
                threshold = filters.threshold_otsu(gray)
                tissue_mask = gray < threshold
            except Exception as e:
                logger.warning(f"Otsu failed, using fallback: {e}")
                tissue_mask = gray < 200
        else:
            # Simple threshold without scikit-image
            threshold = np.mean(gray)
            tissue_mask = gray < threshold
        
        return tissue_mask
    
    def _create_tissue_visualization(self, thumbnail: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Create RGB visualization of tissue detection"""
        vis = thumbnail.copy()
        
        # Overlay mask with transparency
        overlay = np.zeros_like(vis)
        overlay[mask] = [0, 255, 0]  # Green for tissue
        overlay[~mask] = [255, 0, 0]  # Red for background
        
        # Blend with original
        alpha = 0.3
        vis = cv2.addWeighted(vis, 1-alpha, overlay, alpha, 0) if IMAGING_AVAILABLE else vis
        
        return vis
    
    def generate_smart_tiles(self, wsi_path: str, 
                           save_tile_map: bool = False,
                           output_dir: Optional[str] = None) -> List[Dict]:
        """
        Generate tiles ONLY where tissue exists with full coordinate tracking
        
        Args:
            wsi_path: Path to WSI file
            save_tile_map: Save tile map as JSON
            output_dir: Directory to save tile map
        
        Returns:
            List of tile info dicts with coordinates
        """
        # Detect tissue
        tissue_mask, (full_width, full_height), _ = self.detect_tissue_fast(wsi_path)
        
        # Calculate scale factors
        scale_x = tissue_mask.shape[1] / full_width
        scale_y = tissue_mask.shape[0] / full_height
        
        logger.debug(f"Scale factors: x={scale_x:.6f}, y={scale_y:.6f}")
        
        # Generate tiles with coordinates
        tiles = []
        stride = self.tile_size - self.overlap
        
        # Track statistics
        total_possible = 0
        skipped = 0
        row_count = 0
        
        for y in range(0, full_height - self.overlap, stride):
            row_count += 1
            col_count = 0
            
            for x in range(0, full_width - self.overlap, stride):
                col_count += 1
                total_possible += 1
                
                # Calculate actual tile dimensions (handle edges)
                actual_width = min(self.tile_size, full_width - x)
                actual_height = min(self.tile_size, full_height - y)
                
                # Map to thumbnail coordinates
                thumb_x = int(x * scale_x)
                thumb_y = int(y * scale_y)
                thumb_w = int(actual_width * scale_x)
                thumb_h = int(actual_height * scale_y)
                
                # Ensure bounds
                thumb_x = min(thumb_x, tissue_mask.shape[1] - 1)
                thumb_y = min(thumb_y, tissue_mask.shape[0] - 1)
                thumb_w = min(thumb_w, tissue_mask.shape[1] - thumb_x)
                thumb_h = min(thumb_h, tissue_mask.shape[0] - thumb_y)
                
                # Sample tissue mask
                if thumb_w > 0 and thumb_h > 0:
                    tile_region = tissue_mask[
                        thumb_y:thumb_y+thumb_h,
                        thumb_x:thumb_x+thumb_w
                    ]
                    tissue_ratio = tile_region.mean() if tile_region.size > 0 else 0
                else:
                    tissue_ratio = 0
                
                # Include tile if sufficient tissue
                if tissue_ratio >= self.min_tissue_ratio:
                    tile_info = {
                        'tile_id': f"tile_{row_count}_{col_count}",
                        'index': len(tiles),
                        'row': row_count - 1,
                        'col': col_count - 1,
                        'x': int(x),
                        'y': int(y),
                        'width': int(actual_width),
                        'height': int(actual_height),
                        'center_x': int(x + actual_width // 2),
                        'center_y': int(y + actual_height // 2),
                        'tissue_ratio': float(tissue_ratio),
                        'bounds': {
                            'left': int(x),
                            'top': int(y),
                            'right': int(x + actual_width),
                            'bottom': int(y + actual_height)
                        }
                    }
                    tiles.append(tile_info)
                else:
                    skipped += 1
        
        # Calculate statistics
        skip_percentage = (skipped / total_possible) * 100 if total_possible > 0 else 0
        
        # Create summary
        summary = {
            'wsi_path': wsi_path,
            'dimensions': {'width': full_width, 'height': full_height},
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'min_tissue_ratio': self.min_tissue_ratio,
            'total_possible_tiles': total_possible,
            'tiles_with_tissue': len(tiles),
            'tiles_skipped': skipped,
            'skip_percentage': skip_percentage,
            'speedup_factor': total_possible / len(tiles) if len(tiles) > 0 else 1
        }
        
        logger.info(f"Tile generation complete:")
        logger.info(f"  Total possible: {total_possible}")
        logger.info(f"  With tissue: {len(tiles)}")
        logger.info(f"  Skipped: {skipped} ({skip_percentage:.1f}%)")
        logger.info(f"  SPEEDUP: {summary['speedup_factor']:.2f}x")
        
        # Save tile map if requested
        if save_tile_map:
            self._save_tile_map(tiles, summary, output_dir)
        
        return tiles
    
    def _save_tile_map(self, tiles: List[Dict], summary: Dict, 
                      output_dir: Optional[str] = None):
        """Save tile map as JSON for visualization/processing"""
        if output_dir is None:
            output_dir = './tile_maps'
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename from WSI path
        wsi_name = Path(summary['wsi_path']).stem
        output_path = os.path.join(output_dir, f'{wsi_name}_tilemap.json')
        
        tile_map = {
            'summary': summary,
            'tiles': tiles
        }
        
        with open(output_path, 'w') as f:
            json.dump(tile_map, f, indent=2)
        
        logger.info(f"Saved tile map to {output_path}")
    
    def extract_tile_at_coordinates(self, wsi_path: str, x: int, y: int,
                                   width: Optional[int] = None, 
                                   height: Optional[int] = None,
                                   level: int = 0) -> np.ndarray:
        """
        Extract a specific tile from WSI at given coordinates
        
        Args:
            wsi_path: Path to WSI file
            x, y: Top-left coordinates in level 0 reference frame
            width, height: Tile dimensions (defaults to tile_size)
            level: Pyramid level to read from (0 = highest resolution)
        
        Returns:
            Numpy array of the tile image
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required")
        
        slide = openslide.OpenSlide(wsi_path)
        
        # Use default tile size if not specified
        if width is None:
            width = self.tile_size
        if height is None:
            height = self.tile_size
        
        # Validate coordinates
        slide_width, slide_height = slide.dimensions
        if x < 0 or y < 0 or x >= slide_width or y >= slide_height:
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds for {slide_width}x{slide_height}")
        
        # Adjust size if at edge
        width = min(width, slide_width - x)
        height = min(height, slide_height - y)
        
        # Extract region
        tile = slide.read_region((x, y), level, (width, height))
        tile_array = np.array(tile)
        
        # Remove alpha channel if present
        if tile_array.shape[2] == 4:
            tile_array = tile_array[:, :, :3]
        
        slide.close()
        
        return tile_array
    
    def extract_tiles_batch(self, wsi_path: str, 
                           tile_coords: List[Dict]) -> List[np.ndarray]:
        """
        Extract multiple tiles efficiently
        
        Args:
            wsi_path: Path to WSI file
            tile_coords: List of dicts with 'x', 'y', 'width', 'height'
        
        Returns:
            List of tile arrays
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required")
        
        slide = openslide.OpenSlide(wsi_path)
        tiles = []
        
        for coord in tile_coords:
            x = coord['x']
            y = coord['y']
            w = coord.get('width', self.tile_size)
            h = coord.get('height', self.tile_size)
            
            tile = slide.read_region((x, y), 0, (w, h))
            tile_array = np.array(tile)
            
            if tile_array.shape[2] == 4:
                tile_array = tile_array[:, :, :3]
            
            tiles.append(tile_array)
        
        slide.close()
        return tiles
    
    def create_blend_weights(self, tile_size: Optional[int] = None, 
                           overlap: Optional[int] = None) -> np.ndarray:
        """
        Create Gaussian blending weights for seamless tile merging
        
        Args:
            tile_size: Size of tile (defaults to self.tile_size)
            overlap: Overlap amount (defaults to self.overlap)
        
        Returns:
            2D array of blend weights
        """
        if tile_size is None:
            tile_size = self.tile_size
        if overlap is None:
            overlap = self.overlap
        
        weights = np.ones((tile_size, tile_size), dtype=np.float32)
        
        if overlap == 0:
            return weights
        
        # Create linear fade
        for i in range(overlap):
            fade = (i + 1) / (overlap + 1)
            
            # Top edge
            weights[i, :] *= fade
            # Bottom edge
            weights[-(i+1), :] *= fade
            # Left edge
            weights[:, i] *= fade
            # Right edge
            weights[:, -(i+1)] *= fade
        
        # Smooth corners with 2D Gaussian if cv2 available
        if IMAGING_AVAILABLE and overlap > 0:
            kernel_size = overlap // 2
            if kernel_size % 2 == 0:
                kernel_size += 1
            weights = cv2.GaussianBlur(weights, (kernel_size, kernel_size), 0)
        
        return weights
    
    def merge_tiles_with_blending(self, 
                                 tile_results: List[Dict],
                                 full_size: Tuple[int, int],
                                 blend_mode: str = 'gaussian') -> np.ndarray:
        """
        Merge overlapping tiles with advanced blending
        
        Args:
            tile_results: List of dicts with 'x', 'y', 'mask' keys
            full_size: (width, height) of output
            blend_mode: 'gaussian', 'linear', or 'max'
        
        Returns:
            Merged result array
        """
        logger.info(f"Merging {len(tile_results)} tiles into {full_size} using {blend_mode} blending")
        
        # Initialize arrays (numpy uses height, width order)
        height, width = full_size[1], full_size[0]
        result = np.zeros((height, width), dtype=np.float32)
        weights_sum = np.zeros((height, width), dtype=np.float32) + 1e-8  # Avoid division by zero
        
        # Get blend weights
        if blend_mode == 'gaussian':
            blend_weights = self.create_blend_weights()
        else:
            blend_weights = np.ones((self.tile_size, self.tile_size), dtype=np.float32)
        
        # Process each tile
        for i, tile_result in enumerate(tile_results):
            x = tile_result['x']
            y = tile_result['y']
            mask = tile_result['mask']
            
            h, w = mask.shape[:2] if len(mask.shape) >= 2 else (mask.shape[0], 1)
            
            # Resize blend weights if needed
            if h != self.tile_size or w != self.tile_size:
                if IMAGING_AVAILABLE:
                    tile_weights = cv2.resize(blend_weights[:self.tile_size, :self.tile_size], 
                                             (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    tile_weights = blend_weights[:h, :w]
            else:
                tile_weights = blend_weights[:h, :w]
            
            # Apply blending mode
            if blend_mode == 'max':
                result[y:y+h, x:x+w] = np.maximum(result[y:y+h, x:x+w], mask)
            else:
                result[y:y+h, x:x+w] += mask * tile_weights
                weights_sum[y:y+h, x:x+w] += tile_weights
            
            if (i + 1) % 100 == 0:
                logger.debug(f"Merged {i+1}/{len(tile_results)} tiles")
        
        # Normalize
        if blend_mode != 'max':
            result = np.divide(result, weights_sum, where=weights_sum > 1e-8)
        
        logger.info("Tile merging complete")
        return result
    
    def get_coordinate_at_pixel(self, wsi_path: str, 
                               pixel_x: int, pixel_y: int) -> Dict:
        """
        Get tile coordinates containing a specific pixel
        
        Args:
            wsi_path: Path to WSI
            pixel_x, pixel_y: Pixel coordinates in level 0
        
        Returns:
            Dict with tile information containing this pixel
        """
        tiles = self.generate_smart_tiles(wsi_path)
        
        for tile in tiles:
            if (tile['x'] <= pixel_x < tile['x'] + tile['width'] and
                tile['y'] <= pixel_y < tile['y'] + tile['height']):
                return tile
        
        return None
    
    def save_tissue_mask_visualization(self, wsi_path: str, 
                                      output_path: str,
                                      show_grid: bool = True):
        """
        Save visualization of tissue detection with optional tile grid
        
        Args:
            wsi_path: Path to WSI file
            output_path: Output path for visualization
            show_grid: Overlay tile grid on visualization
        """
        tissue_mask, dimensions, vis = self.detect_tissue_fast(wsi_path, return_visualization=True)
        
        if vis is None:
            # Create basic visualization
            vis = np.zeros((*tissue_mask.shape, 3), dtype=np.uint8)
            vis[tissue_mask] = [0, 255, 0]
            vis[~tissue_mask] = [100, 100, 100]
        
        # Add tile grid if requested
        if show_grid and IMAGING_AVAILABLE:
            tiles = self.generate_smart_tiles(wsi_path)
            scale_x = tissue_mask.shape[1] / dimensions[0]
            scale_y = tissue_mask.shape[0] / dimensions[1]
            
            for tile in tiles[:100]:  # Draw first 100 tiles
                x1 = int(tile['x'] * scale_x)
                y1 = int(tile['y'] * scale_y)
                x2 = int((tile['x'] + tile['width']) * scale_x)
                y2 = int((tile['y'] + tile['height']) * scale_y)
                
                # Draw rectangle
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
        
        # Save visualization
        if IMAGING_AVAILABLE:
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        else:
            # Basic save without cv2
            from PIL import Image
            Image.fromarray(vis).save(output_path)
        
        logger.info(f"Saved tissue visualization to {output_path}")


# Utility function for quick testing
def test_smart_tiler(wsi_path: str):
    """Test SmartTiler with a WSI file"""
    tiler = SmartTiler()
    
    # Get WSI info
    info = tiler.get_wsi_info(wsi_path)
    print(f"WSI Info: {info['dimensions']}, {info['level_count']} levels")
    
    # Generate tiles
    tiles = tiler.generate_smart_tiles(wsi_path, save_tile_map=True)
    print(f"Generated {len(tiles)} tiles")
    
    # Save visualization
    output_path = "tissue_detection.png"
    tiler.save_tissue_mask_visualization(wsi_path, output_path, show_grid=True)
    print(f"Saved visualization to {output_path}")
    
    return tiles
