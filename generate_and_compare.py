#!/usr/bin/env python3
"""
Unified Image Generation and Comparison Script

This script:
1. Generates images using Stable Diffusion v1.5.
2. Saves images to results/SD_Images.
3. Generates a comparison figure (Real World vs SD).
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
import pandas as pd
import torch
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler
)

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from generation.generate_images import load_prompts_from_csv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class UnifiedGenerator:
    def __init__(self, device="auto"):
        self.device = self._get_device(device)
        self.sd_model_id = "runwayml/stable-diffusion-v1-5"
        
        self.sd_pipeline = None
        
        # Output directories
        self.base_dir = Path("results")
        self.sd_dir = self.base_dir / "SD_Images"
        
        self.sd_dir.mkdir(parents=True, exist_ok=True)

    def _get_device(self, device):
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device

    def load_sd_model(self):
        if self.sd_pipeline is not None:
            return

        logger.info(f"Loading Stable Diffusion: {self.sd_model_id}")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        self.sd_pipeline = StableDiffusionPipeline.from_pretrained(
            self.sd_model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False
        )
        # Use DPMSolver for better quality
        self.sd_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            self.sd_pipeline.scheduler.config
        )
        self.sd_pipeline = self.sd_pipeline.to(self.device)
        
        if hasattr(self.sd_pipeline, "enable_memory_efficient_attention"):
            self.sd_pipeline.enable_memory_efficient_attention()

    def unload_model(self, model_type):
        if model_type == "sd" and self.sd_pipeline is not None:
            del self.sd_pipeline
            self.sd_pipeline = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    def enhance_prompt(self, prompt):
        """Add quality boosters to prompt."""
        quality_boosters = ", high quality, photorealistic, 8k, highly detailed, professional photography"
        return prompt + quality_boosters

    def generate_images(self, prompts_file, num_images=1, model_types=["sd"]):
        prompts_data = load_prompts_from_csv(prompts_file)
        
        if "sd" in model_types:
            self.load_sd_model()
            logger.info("Generating images with Stable Diffusion...")
            
            for i, item in enumerate(prompts_data):
                prompt = self.enhance_prompt(item['natural_prompt'])
                category = item['category']
                prompt_id = item['prompt_id']
                
                # Create category directory
                cat_dir = self.sd_dir / category
                cat_dir.mkdir(exist_ok=True)
                
                # Check if images already exist
                if all((cat_dir / f"{prompt_id}_{j}.png").exists() for j in range(num_images)):
                    logger.debug(f"SD [{i+1}/{len(prompts_data)}]: Skipping {prompt_id}, images already exist.")
                    continue

                logger.info(f"SD [{i+1}/{len(prompts_data)}]: {prompt[:50]}...")
                
                images = self.sd_pipeline(
                    prompt=prompt,
                    negative_prompt="blurry, low quality, distorted, deformed, ugly, bad anatomy, cartoon, illustration, drawing",
                    num_images_per_prompt=num_images,
                    num_inference_steps=30,
                    guidance_scale=7.5,
                    height=512,
                    width=512
                ).images
                
                for j, img in enumerate(images):
                    filename = f"{prompt_id}_{j}.png"
                    img.save(cat_dir / filename)
            
            self.unload_model("sd")

def main():
    parser = argparse.ArgumentParser(description="Generate and Compare Images")
    #parser.add_argument("--prompts", default="negative_role_prompts_merged_clean_frontview.csv", help="Prompts CSV file")
    parser.add_argument("--prompts", default="test_prompts.csv", help="Prompts CSV file")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images per prompt")
    parser.add_argument("--models", nargs="+", default=["sd"], help="Models to run (sd)")
    parser.add_argument("--device", default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.prompts):
        logger.error(f"Prompts file not found: {args.prompts}")
        sys.exit(1)
        
    generator = UnifiedGenerator(device=args.device)
    generator.generate_images(args.prompts, num_images=args.num_images, model_types=args.models)
    
    # Run Classification
    logger.info("Running demographic classification...")
    try:
        from classification.classify_demographics import FairFaceClassifier
        
        # Initialize classifier (CPU to avoid VRAM issues after generation, or use same device if managed well)
        # Using CPU for safety as per m2_Flux.py recommendation
        classifier = FairFaceClassifier(device="cpu")
        
        # Gather all images
        image_paths = []
        for ext in ['*.png', '*.jpg']:
            image_paths.extend(generator.sd_dir.rglob(ext))
        
        image_paths = [str(p) for p in image_paths]
        
        if image_paths:
            results = classifier.classify_batch(image_paths, output_path=str(generator.base_dir / "classification_results.json"))
            logger.info(f"Classified {len(image_paths)} images.")
        else:
            logger.warning("No images found to classify.")
            
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Run Visualization
    logger.info("Generating comparison figure...")
    try:
        from visualization.compare_images import create_comparison_figure
        create_comparison_figure(base_dir=str(generator.base_dir), output_file="comparison_figure.png")
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # Run Data Visualization (Bias Plots)
    logger.info("Generating bias analysis plots...")
    try:
        from visualization.plot_results import BiasVisualizer
        import json
        
        results_file = generator.base_dir / "classification_results.json"
        if results_file.exists():
            # We need to convert the flat list of results into the expected format for BiasVisualizer
            # BiasVisualizer expects a dictionary with keys like 'race_bias', 'gender_bias' containing distributions
            # But currently we only have a list of predictions. 
            # We need to calculate the distributions first.
            
            with open(results_file, 'r') as f:
                predictions = json.load(f)
            
            # Calculate distributions
            race_counts = {}
            gender_counts = {}
            total = len(predictions)
            
            for p in predictions:
                r = p.get('race', {}).get('prediction')
                g = p.get('gender', {}).get('prediction')
                if r: race_counts[r] = race_counts.get(r, 0) + 1
                if g: gender_counts[g] = gender_counts.get(g, 0) + 1
            
            race_dist = {k: v/total for k, v in race_counts.items()}
            gender_dist = {k: v/total for k, v in gender_counts.items()}
            
            # Construct analysis object
            analysis = {
                "race_bias": {"race_distribution": race_dist, "real_world_distribution": {}}, # Placeholder for real world
                "gender_bias": {"gender_distribution": gender_dist, "real_world_distribution": {}}
            }
            
            visualizer = BiasVisualizer(str(generator.base_dir))
            visualizer.create_all_visualizations(analysis)
            logger.info("Bias plots generated.")
            
    except Exception as e:
        logger.error(f"Data visualization failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

    logger.info("Generation, classification, and visualization complete.")

if __name__ == "__main__":
    main()
