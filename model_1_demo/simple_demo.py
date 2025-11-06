#!/usr/bin/env python3
"""
Simplified Demo for CSDS 447 Demo-1

This is a simplified version that demonstrates the pipeline without
requiring the full transformers library.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_demo_images(prompts_data, output_dir, num_images_per_prompt=2):
    """Create simple demo images using PIL instead of Stable Diffusion."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Creating demo images for {len(prompts_data)} prompts")
    
    results = {}
    
    for i, prompt_data in enumerate(prompts_data):
        prompt = prompt_data['natural_prompt']
        category = prompt_data['category']
        role_label = prompt_data['role_label']
        
        logger.info(f"Creating images for prompt {i+1}/{len(prompts_data)}: {role_label}")
        
        # Create simple text-based images
        for j in range(num_images_per_prompt):
            # Create a simple image with text
            img = Image.new('RGB', (512, 512), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw text
            text = f"{role_label}\n({category})"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (512 - text_width) // 2
            y = (512 - text_height) // 2
            
            draw.text((x, y), text, fill='black', font=font)
            
            # Save image
            filename = f"{i:03d}_{role_label.replace(' ', '_')}_{j:02d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            
            if prompt not in results:
                results[prompt] = []
            results[prompt].append(filepath)
    
    return results

def simulate_demographic_classification(image_paths):
    """Simulate demographic classification results."""
    logger.info(f"Simulating demographic classification for {len(image_paths)} images")
    
    # Simulate random but biased results
    np.random.seed(42)
    
    results = []
    race_labels = ['White', 'Black', 'Indian', 'East Asian', 'Southeast Asian', 'Middle Eastern', 'Latino']
    gender_labels = ['Male', 'Female']
    age_labels = ['20-29', '30-39', '40-49', '50-59']
    
    for image_path in image_paths:
        # Simulate biased results (more likely to be certain demographics for negative roles)
        if 'thief' in image_path.lower() or 'burglar' in image_path.lower():
            # Bias towards certain demographics
            race_weights = [0.1, 0.3, 0.1, 0.1, 0.1, 0.1, 0.2]  # More likely to be Black/Latino
            gender_weights = [0.7, 0.3]  # More likely to be Male
        elif 'hacker' in image_path.lower() or 'scammer' in image_path.lower():
            race_weights = [0.4, 0.1, 0.1, 0.2, 0.1, 0.05, 0.05]  # More likely to be White/East Asian
            gender_weights = [0.8, 0.2]  # More likely to be Male
        else:
            # More balanced for other roles
            race_weights = [0.3, 0.2, 0.1, 0.15, 0.1, 0.05, 0.1]
            gender_weights = [0.6, 0.4]
        
        race_idx = np.random.choice(len(race_labels), p=race_weights)
        gender_idx = np.random.choice(len(gender_labels), p=gender_weights)
        age_idx = np.random.choice(len(age_labels))
        
        result = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'race': {
                'prediction': race_labels[race_idx],
                'confidence': np.random.uniform(0.7, 0.95)
            },
            'gender': {
                'prediction': gender_labels[gender_idx],
                'confidence': np.random.uniform(0.7, 0.95)
            },
            'age': {
                'prediction': age_labels[age_idx],
                'confidence': np.random.uniform(0.6, 0.9)
            }
        }
        results.append(result)
    
    return results

def analyze_bias(results):
    """Analyze bias in the simulated results."""
    logger.info("Analyzing bias in classification results")
    
    # Extract predictions
    race_predictions = [r['race']['prediction'] for r in results]
    gender_predictions = [r['gender']['prediction'] for r in results]
    
    # Count distributions
    race_counts = {}
    gender_counts = {}
    
    for race in race_predictions:
        race_counts[race] = race_counts.get(race, 0) + 1
    
    for gender in gender_predictions:
        gender_counts[gender] = gender_counts.get(gender, 0) + 1
    
    total = len(results)
    
    # Calculate proportions
    race_dist = {k: v/total for k, v in race_counts.items()}
    gender_dist = {k: v/total for k, v in gender_counts.items()}
    
    # Calculate bias metrics
    analysis = {
        'race_distribution': race_dist,
        'gender_distribution': gender_dist,
        'total_samples': total,
        'bias_analysis': {
            'race_bias_detected': max(race_dist.values()) > 0.4,  # If any race > 40%
            'gender_bias_detected': abs(gender_dist.get('Male', 0) - gender_dist.get('Female', 0)) > 0.3,
            'overall_bias_score': max(race_dist.values()) + abs(gender_dist.get('Male', 0) - 0.5)
        }
    }
    
    return analysis

def create_visualizations(analysis, output_dir):
    """Create simple visualizations of the bias analysis."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Race distribution plot
    plt.figure(figsize=(10, 6))
    races = list(analysis['race_distribution'].keys())
    counts = list(analysis['race_distribution'].values())
    
    plt.bar(races, counts)
    plt.title('Race Distribution in Generated Images')
    plt.xlabel('Race')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_distribution.png'))
    plt.close()
    
    # Gender distribution plot
    plt.figure(figsize=(8, 6))
    genders = list(analysis['gender_distribution'].keys())
    counts = list(analysis['gender_distribution'].values())
    
    plt.bar(genders, counts)
    plt.title('Gender Distribution in Generated Images')
    plt.xlabel('Gender')
    plt.ylabel('Proportion')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")

def main():
    """Run the simplified demo."""
    logger.info("=" * 60)
    logger.info("CSDS 447 DEMO-1 - SIMPLIFIED VERSION")
    logger.info("=" * 60)
    
    # Load prompts
    prompts_file = "negative_role_prompts_all.csv"
    if not os.path.exists(prompts_file):
        logger.error(f"Prompts file not found: {prompts_file}")
        return
    
    prompts_data = pd.read_csv(prompts_file).to_dict('records')
    logger.info(f"Loaded {len(prompts_data)} prompts")
    
    # Create output directories
    output_dir = "demo_results"
    images_dir = os.path.join(output_dir, "generated_images")
    analysis_dir = os.path.join(output_dir, "analysis")
    figures_dir = os.path.join(output_dir, "figures")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(analysis_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    
    # Step 1: Create demo images
    logger.info("Step 1: Creating demo images...")
    image_results = create_demo_images(prompts_data, images_dir, num_images_per_prompt=2)
    
    # Step 2: Simulate demographic classification
    logger.info("Step 2: Simulating demographic classification...")
    all_image_paths = []
    for paths in image_results.values():
        all_image_paths.extend(paths)
    
    classification_results = simulate_demographic_classification(all_image_paths)
    
    # Save classification results
    with open(os.path.join(analysis_dir, "classification_results.json"), 'w') as f:
        json.dump(classification_results, f, indent=2)
    
    # Step 3: Analyze bias
    logger.info("Step 3: Analyzing bias...")
    bias_analysis = analyze_bias(classification_results)
    
    # Save analysis
    with open(os.path.join(analysis_dir, "bias_analysis.json"), 'w') as f:
        json.dump(bias_analysis, f, indent=2)
    
    # Step 4: Create visualizations
    logger.info("Step 4: Creating visualizations...")
    create_visualizations(bias_analysis, figures_dir)
    
    # Generate report
    logger.info("Step 5: Generating report...")
    report = f"""
CSDS 447 DEMO-1 RESULTS REPORT
==============================

Analysis Summary:
- Total images analyzed: {bias_analysis['total_samples']}
- Race bias detected: {bias_analysis['bias_analysis']['race_bias_detected']}
- Gender bias detected: {bias_analysis['bias_analysis']['gender_bias_detected']}
- Overall bias score: {bias_analysis['bias_analysis']['overall_bias_score']:.3f}

Race Distribution:
{json.dumps(bias_analysis['race_distribution'], indent=2)}

Gender Distribution:
{json.dumps(bias_analysis['gender_distribution'], indent=2)}

Files Generated:
- Images: {len(all_image_paths)} in {images_dir}/
- Analysis: {analysis_dir}/bias_analysis.json
- Visualizations: {figures_dir}/
"""
    
    with open(os.path.join(analysis_dir, "report.txt"), 'w') as f:
        f.write(report)
    
    logger.info("=" * 60)
    logger.info("DEMO-1 COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Generated {len(all_image_paths)} images")
    logger.info(f"Bias analysis completed")
    logger.info(f"Visualizations created")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
