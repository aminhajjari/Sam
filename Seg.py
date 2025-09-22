#!/usr/bin/env python3
"""
MILK10k Dataset Segmentation using SAM-2
A streamlined pipeline for segmenting images from a dataset using the offline SAM-2 model.
"""
import os
import sys
import cv2
import torch
import numpy as np
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
from tqdm import tqdm
import time
import traceback
import json

# Ignore warnings for a cleaner output
warnings.filterwarnings('ignore')

# Setup logging for clear feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ---
## Configuration and Utility Classes
---
@dataclass
class SegmentationConfig:
    """Configuration class for segmentation pipeline parameters."""
    
    # The base path is the parent directory of 'Sam' folder.
    base_path: Path = Path(os.getcwd()).parent
    
    # Paths are now constructed relative to the base_path
    dataset_path: Path = base_path / "MILK10k_Training_Input"
    output_path: Path = base_path / "SegConOutputs" / "segmented"
    sam2_model_path: Path = base_path / "segment-anything-2"
    
    # Model files, based on your downloaded files
    sam2_checkpoint: str = "sam2.1_hiera_large.pt"
    sam2_config: str = "sam2_hiera_l.yaml"
    
    # Processing parameters
    # The key change is here: set max_folders to 50
    max_folders: int = 50
    save_masks: bool = True
    save_visualizations: bool = True
    
    # Output file settings
    results_file: str = "segmentation_results.json"


def _setup_device() -> str:
    """Determines and sets up the device (CUDA or CPU) for computation."""
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        return "cuda"
    else:
        logger.warning("CUDA not available, using CPU. This will be much slower.")
        return "cpu"

# ---
## Main Segmentation Pipeline
---
class DatasetSegmenter:
    def __init__(self, config: SegmentationConfig):
        self.config = config
        self.device = _setup_device()
        self.sam_predictor = None
        self.results = []
        
        self._initialize_pipeline()
    
    def _initialize_pipeline(self):
        """Initializes the output directories and loads the SAM-2 model."""
        self.config.output_path.mkdir(parents=True, exist_ok=True)
        self._load_sam2()
        logger.info("Pipeline initialization complete.")
    
    def _load_sam2(self):
        """Loads the SAM-2 model from the local checkpoint and config files."""
        logger.info("Loading SAM-2 model from local files...")
        
        sys.path.insert(0, str(self.config.sam2_model_path))
        
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as e:
            raise ImportError(f"Cannot import SAM2 modules. Please ensure the local repo is correctly placed and the necessary dependencies are installed: {e}")

        config_path = self.config.sam2_model_path / "sam2_configs" / self.config.sam2_config
        checkpoint_path = self.config.sam2_model_path / "checkpoints" / self.config.sam2_checkpoint

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM-2 checkpoint not found at: {checkpoint_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"SAM-2 config not found at: {config_path}")

        try:
            sam2_model = build_sam2(str(config_path), str(checkpoint_path), device=self.device)
            self.sam_predictor = SAM2ImagePredictor(sam2_model)
            logger.info("SAM-2 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load SAM-2 model from local files: {e}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Failed to initialize SAM-2 predictor.")
    
    def segment_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Performs image segmentation using SAM-2 with a center point prompt."""
        h, w = image.shape[:2]
        masks = []
        
        try:
            with torch.inference_mode():
                self.sam_predictor.set_image(image)
                
                point_coords = np.array([[w // 2, h // 2]])
                point_labels = np.array([1])
                
                output_masks, scores, _ = self.sam_predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True
                )
                
                best_mask_idx = np.argmax(scores)
                masks.append(output_masks[best_mask_idx].astype(np.uint8))
                
        except Exception as e:
            logger.warning(f"Segmentation failed for image. Falling back to a full image mask. Error: {e}")
            full_mask = np.ones((h, w), dtype=np.uint8)
            masks.append(full_mask)
            
        return masks

    def process_dataset(self):
        """Iterates through the dataset and segments each image, saving the results."""
        dataset_path = self.config.dataset_path
        folders = [f for f in dataset_path.iterdir() if f.is_dir()]
        
        if self.config.max_folders:
            folders = folders[:self.config.max_folders]
        
        logger.info(f"Starting segmentation of {len(folders)} folders...")
        
        for folder in tqdm(folders, desc="Processing Folders"):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in image_extensions:
                image_files.extend(folder.glob(f"*{ext}"))
                image_files.extend(folder.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.warning(f"No images found in {folder.name}. Skipping.")
                continue
            
            output_folder = self.config.output_path / folder.name
            output_folder.mkdir(parents=True, exist_ok=True)
            
            for img_path in image_files:
                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        logger.warning(f"Could not load image: {img_path}. Skipping.")
                        continue
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    masks = self.segment_image(image_rgb)
                    
                    image_results = {
                        "image_path": str(img_path.relative_to(self.config.dataset_path)),
                        "image_name": img_path.name,
                        "num_masks": len(masks)
                    }
                    self.results.append(image_results)

                    for i, mask in enumerate(masks):
                        mask_filename = f"{img_path.stem}_mask_{i}.png"
                        mask_path = output_folder / mask_filename
                        
                        if self.config.save_masks:
                            cv2.imwrite(str(mask_path), mask * 255)
                        
                        if self.config.save_visualizations:
                            segmented_image = self._apply_mask(image, mask)
                            viz_filename = f"{img_path.stem}_seg_viz_{i}.png"
                            viz_path = output_folder / viz_filename
                            cv2.imwrite(str(viz_path), segmented_image)
                            
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    logger.error(traceback.format_exc())
                    continue
            
            if self.device == "cuda":
                torch.cuda.empty_cache()

        self._save_results()

    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Applies a mask to the image for visualization purposes."""
        masked_image = image.copy()
        color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)
        masked_image[mask == 1] = masked_image[mask == 1] * 0.7 + color * 0.3
        return masked_image

    def _save_results(self):
        """Saves the segmentation results to a JSON file."""
        output_file_path = self.config.output_path.parent / self.config.results_file
        try:
            with open(output_file_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            logger.info(f"Segmentation results saved to {output_file_path}")
        except Exception as e:
            logger.error(f"Failed to save results to JSON file: {e}")
            logger.error(traceback.format_exc())

# ---
## Script Execution
---
if __name__ == "__main__":
    try:
        config = SegmentationConfig()
        segmenter = DatasetSegmenter(config)
        segmenter.process_dataset()
        logger.info("Dataset segmentation completed successfully.")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
