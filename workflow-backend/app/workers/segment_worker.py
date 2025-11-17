# ============================================
# FILE: workflow-backend/app/workers/segment_worker.py
# ============================================
"""
InstanSeg Worker with Smart Tiling
This is where the magic happens - optimized segmentation
"""

from app.celery_app import celery_app
from app.workers.smart_tiling import SmartTiler
from app.scheduler.branch_scheduler import scheduler

# Lazy import openslide to avoid errors if system library not installed
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    openslide = None

import numpy as np
import logging
import time

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.workers.segment_wsi")
def segment_wsi(self, job_data):
    """
    Segment whole-slide image using InstanSeg with smart tiling
    
    Args:
        job_data: {
            'job_id': str,
            'image': str (path to .svs file),
            'user_id': str,
            'workflow_id': str,
            'branch_id': str
        }
    """
    job_id = job_data['job_id']
    wsi_path = job_data['image']
    branch_id = job_data['branch_id']

    user_id = job_data['user_id']
    workflow_id = job_data['workflow_id']

    # Get workflow name from scheduler
    workflow = scheduler.get_workflow_status(workflow_id)
    workflow_name = workflow['name'] if workflow else 'unknown_workflow'
    # Sanitize workflow name for file system
    workflow_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in workflow_name)

    logger.info(f"Starting segmentation job {job_id} for {wsi_path} (workflow: {workflow_name})")

    # Check if file is WSI (.svs) or regular image
    from pathlib import Path
    import cv2
    from PIL import Image

    image_path = Path(wsi_path)
    is_wsi = image_path.suffix.lower() in ['.svs', '.ndpi', '.scn']

    # Process regular images without OpenSlide
    if not is_wsi or not OPENSLIDE_AVAILABLE:
        try:
            logger.info(f"Processing regular image: {image_path}")

            # Update state: Loading
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'loading', 'progress': 10, 'message': 'Loading image...'}
            )

            start_time = time.time()

            # Load image
            img = Image.open(image_path).convert('RGB')
            img_array = np.array(img)
            original_height, original_width = img_array.shape[:2]

            logger.info(f"Loaded image: {original_width}x{original_height}")

            # FORCE SQUARE dimensions - InstanSeg requires this due to internal pooling
            # Take the larger dimension and make it square at the next power of 2
            max_dim = max(original_width, original_height)

            # Find next power of 2 for the largest dimension
            if max_dim <= 256:
                target_size = 256
            elif max_dim <= 512:
                target_size = 512
            elif max_dim <= 1024:
                target_size = 1024
            elif max_dim <= 2048:
                target_size = 2048
            else:
                target_size = 2048  # Cap at 2048

            # Make it SQUARE
            target_width = target_size
            target_height = target_size

            if target_width != original_width or target_height != original_height:
                logger.info(f"Resizing image from {original_width}x{original_height} to "
                          f"SQUARE {target_width}x{target_height} (InstanSeg requires square power-of-2)")
                # Pad to square first to maintain aspect ratio
                if original_width != original_height:
                    # Pad the shorter dimension to make it square
                    if original_width > original_height:
                        # Pad height
                        pad_top = (original_width - original_height) // 2
                        pad_bottom = original_width - original_height - pad_top
                        img_array = np.pad(img_array, ((pad_top, pad_bottom), (0, 0), (0, 0)), mode='constant', constant_values=0)
                    else:
                        # Pad width
                        pad_left = (original_height - original_width) // 2
                        pad_right = original_height - original_width - pad_left
                        img_array = np.pad(img_array, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
                    logger.info(f"Padded to square: {img_array.shape[1]}x{img_array.shape[0]}")

                # Now resize the square image to target size
                img_resized = cv2.resize(img_array, (target_width, target_height),
                                        interpolation=cv2.INTER_LINEAR)
                img_array = img_resized
            else:
                logger.info(f"Image already square power-of-2: {original_width}x{original_height}")

            height, width = img_array.shape[:2]

            # Update state: Segmenting
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'segmenting', 'progress': 40, 'message': f'Running InstanSeg on {width}x{height} image...'}
            )

            # Run InstanSeg
            try:
                mask = run_instanseg_batch([img_array])[0]
                num_cells = len(np.unique(mask)) - 1  # -1 for background
            except RuntimeError as e:
                error_msg = str(e)
                if "shape mismatch" in error_msg or "cannot be broadcast" in error_msg:
                    logger.error(f"InstanSeg shape mismatch error for image {width}x{height}. "
                                f"This is a known issue with certain image dimensions. "
                                f"Try resizing your image to dimensions divisible by 16.")
                    raise RuntimeError(f"Image segmentation failed due to dimension incompatibility "
                                     f"(size: {width}x{height}). Please resize image to dimensions "
                                     f"divisible by 16 (e.g., 512x512, 1024x1024, 2048x2048).") from e
                else:
                    raise

            logger.info(f"Segmented {num_cells} cells/objects")

            # Update state: Saving
            self.update_state(
                state='PROGRESS',
                meta={'stage': 'saving', 'progress': 80, 'message': f'Saving results ({num_cells} cells)...'}
            )

            # Save results in outputs/{user_id}/{workflow_name}/
            output_dir = Path(f"outputs/{user_id}/{workflow_name}")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save segmentation mask
            mask_path = output_dir / f"{job_id}_mask.png"
            if num_cells > 0:
                mask_vis = (mask.astype(float) / mask.max() * 255).astype(np.uint8)
            else:
                mask_vis = np.zeros_like(mask, dtype=np.uint8)
            cv2.imwrite(str(mask_path), mask_vis)

            # Create colored overlay
            overlay_path = output_dir / f"{job_id}_overlay.png"
            overlay = img_array.copy()
            colored_mask = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

            # Save annotated result
            result_path = output_dir / f"{job_id}_result.png"
            result_img = img_array.copy()
            cv2.putText(result_img, f"Cells: {num_cells}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

            total_time = time.time() - start_time

            logger.info(f"Job {job_id} completed in {total_time:.2f}s")

            return {
                'status': 'SUCCESS',
                'job_id': job_id,
                'message': f'Segmentation complete: {num_cells} cells detected',
                'num_cells': num_cells,
                'processing_time': total_time,
                'results': {
                    'mask': str(mask_path),
                    'overlay': str(overlay_path),
                    'result': str(result_path),
                    'original': str(image_path)
                }
            }

        except Exception as e:
            logger.error(f"Regular image processing failed for {job_id}: {e}", exc_info=True)
            raise

    # Process WSI with OpenSlide (original code)
    try:
        # Initialize tiler
        tiler = SmartTiler(tile_size=1024, overlap=128, min_tissue_ratio=0.1)
        
        # STEP 1: Generate smart tiles (skip background)
        self.update_state(
            state='PROGRESS',
            meta={
                'stage': 'tiling',
                'progress': 5,
                'message': 'Detecting tissue and generating tiles...'
            }
        )
        
        start_time = time.time()
        tiles = tiler.generate_smart_tiles(wsi_path)
        tiling_time = time.time() - start_time
        
        logger.info(f"Tiling complete in {tiling_time:.2f}s: {len(tiles)} tiles")
        
        if len(tiles) == 0:
            logger.warning(f"No tiles with tissue found in {wsi_path}")
            return {
                'status': 'SUCCESS',
                'message': 'No tissue found in image',
                'tiles_processed': 0,
                'total_time': tiling_time
            }
        
        # STEP 2: Process tiles (batch processing for efficiency)
        self.update_state(
            state='PROGRESS',
            meta={
                'stage': 'segmenting',
                'progress': 10,
                'tiles_total': len(tiles),
                'tiles_done': 0,
                'message': 'Segmenting cells...'
            }
        )
        
        tile_results = []
        slide = openslide.OpenSlide(wsi_path)
        
        # Process tiles in batches
        batch_size = 16  # Process 16 tiles at once (GPU optimization)
        
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i+batch_size]
            
            # Load batch of tiles
            batch_images = []
            for tile_info in batch:
                tile_img = slide.read_region(
                    (tile_info['x'], tile_info['y']),
                    0,  # level (full resolution)
                    (tile_info['width'], tile_info['height'])
                )
                # Convert RGBA to RGB
                tile_img = np.array(tile_img.convert('RGB'))
                batch_images.append(tile_img)
            
            # Run InstanSeg on batch (placeholder - replace with real model)
            batch_masks = run_instanseg_batch(batch_images)
            
            # Store results
            for j, mask in enumerate(batch_masks):
                tile_results.append({
                    'x': batch[j]['x'],
                    'y': batch[j]['y'],
                    'mask': mask
                })
            
            # Update progress
            tiles_done = i + len(batch)
            progress = 10 + int((tiles_done / len(tiles)) * 80)
            
            self.update_state(
                state='PROGRESS',
                meta={
                    'stage': 'segmenting',
                    'progress': progress,
                    'tiles_total': len(tiles),
                    'tiles_done': tiles_done,
                    'message': f'Processing tiles: {tiles_done}/{len(tiles)}'
                }
            )
            
            logger.debug(f"Processed batch {i//batch_size + 1}: {tiles_done}/{len(tiles)} tiles")
        
        segmentation_time = time.time() - start_time - tiling_time
        
        # STEP 3: Merge tiles with blending
        self.update_state(
            state='PROGRESS',
            meta={
                'stage': 'merging',
                'progress': 95,
                'message': 'Merging tiles...'
            }
        )
        
        merge_start = time.time()
        final_mask = tiler.merge_tiles_with_blending(
            tile_results,
            slide.dimensions
        )
        merge_time = time.time() - merge_start

        # Count cells in final mask
        num_cells = len(np.unique(final_mask)) - 1  # -1 for background
        logger.info(f"Detected {num_cells} cells in merged mask")

        # Save results (downsampled for large WSI)
        self.update_state(
            state='PROGRESS',
            meta={
                'stage': 'saving',
                'progress': 98,
                'message': f'Saving results ({num_cells} cells)...'
            }
        )

        # Save results in outputs/{user_id}/{workflow_name}/
        output_dir = Path(f"outputs/{user_id}/{workflow_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Downsample mask for visualization (WSI are huge, save at 10% resolution)
        max_dim = 4096  # Max dimension for output
        scale = min(1.0, max_dim / max(final_mask.shape))
        if scale < 1.0:
            new_size = (int(final_mask.shape[1] * scale), int(final_mask.shape[0] * scale))
            mask_vis = cv2.resize(final_mask.astype(np.float32), new_size, interpolation=cv2.INTER_NEAREST)
            logger.info(f"Downsampled mask from {final_mask.shape} to {mask_vis.shape}")
        else:
            mask_vis = final_mask

        # Save segmentation mask
        mask_path = output_dir / f"{job_id}_mask.png"
        if num_cells > 0:
            mask_norm = (mask_vis.astype(float) / mask_vis.max() * 255).astype(np.uint8)
        else:
            mask_norm = np.zeros_like(mask_vis, dtype=np.uint8)
        cv2.imwrite(str(mask_path), mask_norm)

        # Load thumbnail of original image for overlay
        thumbnail = slide.get_thumbnail((mask_vis.shape[1], mask_vis.shape[0]))
        thumbnail_np = np.array(thumbnail.convert('RGB'))

        # Create colored overlay
        overlay_path = output_dir / f"{job_id}_overlay.png"
        colored_mask = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(thumbnail_np, 0.7, colored_mask, 0.3, 0)
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        # Save annotated result with stats
        result_path = output_dir / f"{job_id}_result.png"
        result_img = thumbnail_np.copy()
        # Add text annotations
        cv2.putText(result_img, f"Cells: {num_cells}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(result_img, f"Tiles: {len(tiles)}", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(result_img, f"Time: {total_time:.1f}s", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imwrite(str(result_path), cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))

        slide.close()

        total_time = time.time() - start_time

        logger.info(f"Job {job_id} completed in {total_time:.2f}s")
        logger.info(f"  Tiling: {tiling_time:.2f}s")
        logger.info(f"  Segmentation: {segmentation_time:.2f}s")
        logger.info(f"  Merging: {merge_time:.2f}s")

        # Return result - FastAPI job_executor will poll this and update scheduler
        return {
            'status': 'SUCCESS',
            'job_id': job_id,
            'message': f'Segmentation complete: {num_cells} cells detected',
            'num_cells': num_cells,
            'tiles_processed': len(tiles),
            'processing_time': total_time,
            'tiling_time': tiling_time,
            'segmentation_time': segmentation_time,
            'merge_time': merge_time,
            'output_shape': final_mask.shape,
            'results': {
                'mask': str(mask_path),
                'overlay': str(overlay_path),
                'result': str(result_path),
                'original': str(image_path)
            }
        }
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        # Don't manually update state to FAILURE - let Celery handle it automatically
        # Re-raising the exception will cause Celery to mark the task as failed
        raise


# ============================================
# INSTANSEG MODEL INITIALIZATION
# ============================================

# Global model instance (loaded once)
_instanseg_model = None
_model_lock = None

def get_instanseg_model():
    """
    Get or initialize InstanSeg model (singleton pattern)

    Returns:
        InstanSeg model instance or None if not available
    """
    global _instanseg_model

    if _instanseg_model is None:
        try:
            from instanseg import InstanSeg

            logger.info("Loading InstanSeg model...")

            # Load pre-trained model using InstanSeg class
            # Available models: 'brightfield_nuclei', 'fluorescence_nuclei'
            try:
                # Try brightfield model (most common for histology)
                _instanseg_model = InstanSeg(
                    "brightfield_nuclei",
                    verbosity=0  # Reduce logging noise
                )
                logger.info("Loaded brightfield_nuclei model successfully")
            except Exception as e:
                logger.warning(f"Failed to load brightfield model: {e}")
                try:
                    # Fallback to fluorescence model
                    _instanseg_model = InstanSeg(
                        "fluorescence_nuclei",
                        verbosity=0
                    )
                    logger.info("Loaded fluorescence_nuclei model successfully")
                except Exception as e2:
                    logger.error(f"Failed to load fluorescence model: {e2}")
                    raise

            logger.info(f"InstanSeg model loaded successfully")

        except ImportError as e:
            logger.error(f"InstanSeg not installed: {e}")
            _instanseg_model = None
        except Exception as e:
            logger.error(f"Error loading InstanSeg model: {e}", exc_info=True)
            _instanseg_model = None

    return _instanseg_model


def run_instanseg_batch(images: list) -> list:
    """
    Run InstanSeg on a batch of images

    Args:
        images: List of numpy arrays (RGB images, shape: [H, W, 3])

    Returns:
        List of instance segmentation masks (numpy arrays)
    """
    model = get_instanseg_model()

    if model is None:
        raise RuntimeError("InstanSeg model not available. Please ensure InstanSeg is properly installed.")

    # InstanSeg inference
    try:
        logger.info(f"Running InstanSeg on batch of {len(images)} images")

        masks = []

        # Process each image individually (InstanSeg API)
        for img in images:
            # InstanSeg expects numpy array in [H, W, C] format
            # with values in range [0, 255] (uint8)
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Run InstanSeg inference using eval_small_image method
            # This returns a tuple: (labeled_output, image_tensor)
            try:
                # eval_small_image returns (labeled_output, image_tensor)
                # We only need the labeled_output which contains instance segmentation
                labeled_output, _ = model.eval_small_image(img)
                mask = labeled_output
            except (AttributeError, ValueError) as e:
                # Fallback to eval method if eval_small_image doesn't work
                logger.warning(f"eval_small_image failed, trying eval: {e}")
                result = model.eval(img)
                # eval may return just the mask or a dict/tuple
                if isinstance(result, tuple):
                    mask = result[0]
                elif isinstance(result, dict):
                    mask = result.get('labels', result)
                else:
                    mask = result

            # Ensure mask is in the right format
            if hasattr(mask, 'cpu'):  # torch.Tensor
                mask = mask.cpu().numpy()

            # Convert to numpy array if needed and ensure it's a proper 2D array
            if not isinstance(mask, np.ndarray):
                mask = np.array(mask)

            # Ensure 2D shape for instance segmentation mask
            if mask.ndim > 2:
                # If it has extra dimensions, squeeze them
                mask = np.squeeze(mask)

            # Convert to uint16 for instance segmentation
            mask = mask.astype(np.uint16)
            masks.append(mask)

        logger.info(f"InstanSeg inference complete for {len(masks)} images")
        return masks

    except Exception as e:
        logger.error(f"InstanSeg inference failed: {e}", exc_info=True)
        raise RuntimeError(f"InstanSeg inference failed: {str(e)}") from e
