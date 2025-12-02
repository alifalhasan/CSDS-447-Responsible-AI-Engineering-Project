#!/usr/bin/env python3
"""
Comparison Visualization Script

This script creates a comparison figure showing Real World, SD, and Flux images
side-by-side for each category.
"""

import os
import sys
import argparse
import logging
import math
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_comparison_figure(base_dir="results", output_file="comparison_figure.png"):
    base_path = Path(base_dir)
    sd_dir = base_path / "SD_Images"
    sd_dir = base_path / "SD_Images"
    real_world_dir = base_path / "Real_World_Images" # Placeholder
    
    # Get all categories
    if not sd_dir.exists():
        logger.error("SD Results directory not found.")
        return

    categories = sorted([d.name for d in sd_dir.iterdir() if d.is_dir()])
    
    if not categories:
        logger.warning("No categories found in SD_Images.")
        return

    logger.info(f"Found {len(categories)} categories: {categories}")

    # Setup figure
    num_cols = 1 # SD Images only
    num_rows = len(categories)
    
    fig_width = 6
    fig_height = 5 * num_rows
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    
    # Handle single column case (axes is 1D array)
    if num_rows == 1:
        axes = [axes]
    
    # Column titles
    cols = ["Stable Diffusion v1.5"]
    # If num_rows > 1, axes is 1D array of axes, so we can't index axes[0] like before if num_cols=1
    # Actually if num_cols=1, axes is a 1D array of length num_rows.
    # We need to set title on the first axis if we want a column title, or just suptitle.
    
    # Let's use a simpler approach for 1 column
    if num_rows > 1:
        axes[0].set_title(cols[0], fontsize=16, fontweight='bold')
    else:
        axes[0].set_title(cols[0], fontsize=16, fontweight='bold')

    # Load classification results
    classification_results = {}
    results_file = base_path / "classification_results.json"
    if results_file.exists():
        import json
        with open(results_file, 'r') as f:
            data = json.load(f)
            # Create a map from filename to result
            for item in data:
                classification_results[item.get('image_name')] = item
    
    def get_annotation(image_path):
        if not image_path: return ""
        filename = image_path.name
        if filename in classification_results:
            res = classification_results[filename]
            race = res.get('race', {}).get('prediction', 'Unknown')
            gender = res.get('gender', {}).get('prediction', 'Unknown')
            return f"Race: {race}\nGender: {gender}"
        return ""

    for i, category in enumerate(categories):
        ax = axes[i] if num_rows > 1 else axes[0]
        
        # Row label (Category)
        ax.set_ylabel(category, fontsize=14, fontweight='bold', rotation=90, labelpad=20)

        # SD Image
        sd_cat_dir = sd_dir / category
        sd_images = sorted(list(sd_cat_dir.glob("*.png")))
        if sd_images:
            img_path = sd_images[0]
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            
            # Add annotation
            text = get_annotation(img_path)
            if text:
                ax.set_xlabel(text, fontsize=10, fontweight='bold', color='blue')
        else:
            ax.text(0.5, 0.5, "No SD Image", ha='center', va='center')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    output_path = base_path / output_file
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Comparison figure saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Create Comparison Figure")
    parser.add_argument("--base-dir", default="results", help="Base results directory")
    parser.add_argument("--output", default="comparison_figure.png", help="Output filename")
    
    args = parser.parse_args()
    
    create_comparison_figure(args.base_dir, args.output)

if __name__ == "__main__":
    main()
