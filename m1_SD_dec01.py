#!/usr/bin/env python3
"""
Model 1: Stable Diffusion (SD) Pipeline for Negative Role Fairness Analysis

This script runs the complete pipeline for analyzing bias in Stable Diffusion
text-to-image model when generating images of negative social roles.

All results are saved to m1_SD_result_dec01/ directory.

Usage:
    python m1_SD_dec01.py --num-images 3 --device auto --model stable-diffusion-xl-base-1.0
    python m1_SD_dec01.py --compare-with m1_SD_result_dec01
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Dict, Any
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generation.generate_images import ImageGenerator, load_prompts_from_csv
from classification.classify_demographics import FairFaceClassifier
from analysis.compute_metrics import BiasAnalyzer, convert_to_native_types
from visualization.plot_results import BiasVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StableDiffusionPipeline:
    """Complete pipeline for Stable Diffusion negative role fairness analysis."""
    
    def __init__(self, output_dir: str = "m1_SD_result_dec01", device: str = "auto", model_name: str = "stable-diffusion-v1-5"):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Directory to save all results (default: m1_SD_result_dec01)
            device: Device to use for computation
            model_name: Name of the model to use (default: stable-diffusion-v1-5)
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.model_name = model_name
        
        # Create output directories
        self.images_dir = self.output_dir / "generated_images"
        self.annotations_dir = self.output_dir / "annotations"
        self.analysis_dir = self.output_dir / "analysis"
        self.figures_dir = self.output_dir / "figures"
        
        for dir_path in [self.images_dir, self.annotations_dir, self.analysis_dir, self.figures_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.image_generator = None
        self.classifier = None
        self.analyzer = BiasAnalyzer()
        self.visualizer = BiasVisualizer(str(self.figures_dir))
        
        logger.info(f"Stable Diffusion Pipeline initialized")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def load_prompts(self, prompts_file: str) -> list:
        """Load prompts from CSV file."""
        logger.info(f"Loading prompts from {prompts_file}")
        prompts_data = load_prompts_from_csv(prompts_file)
        logger.info(f"Loaded {len(prompts_data)} prompts")
        return prompts_data
    
    def enhance_prompt(self, prompt: str) -> str:
        """Add quality boosters to prompt."""
        quality_boosters = ", high quality, photorealistic, 8k, highly detailed, professional photography"
        return prompt + quality_boosters

    def generate_images(self, prompts_data: list, 
                       num_images_per_prompt: int = 5) -> Dict[str, Any]:
        """Generate images using the specified model."""
        logger.info(f"Initializing {self.model_name} image generator")
        self.image_generator = ImageGenerator(
            model_name=self.model_name, 
            device=self.device,
            use_refiner=False  # Can be exposed as an option later
        )
        
        # Extract prompts
        # prompts = [p['natural_prompt'] for p in prompts_data]
        
        logger.info(f"Generating {num_images_per_prompt} images per prompt for {len(prompts_data)} prompts")
        start_time = time.time()
        
        results = {}
        total_images = 0
        
        # Negative prompt from user's snippet
        negative_prompt = "blurry, low quality, distorted, deformed, ugly, bad anatomy, cartoon, illustration, drawing"
        
        for i, item in enumerate(prompts_data):
            natural_prompt = item['natural_prompt']
            prompt = self.enhance_prompt(natural_prompt)
            category = item.get('category', 'uncategorized')
            prompt_id = item.get('prompt_id', f"prompt_{i}")
            
            # Create category directory
            cat_dir = self.images_dir / category
            cat_dir.mkdir(exist_ok=True, parents=True)
            
            logger.info(f"Generating for prompt {i+1}/{len(prompts_data)}: {prompt[:50]}...")
            
            # Check if images already exist (simple check based on count)
            # This is a bit tricky with random filenames, but we can check if we have enough images in the folder
            # For now, we'll just generate.
            
            images = self.image_generator.generate_image(
                prompt=prompt,
                num_images=num_images_per_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                # Let generator decide height/width based on model
            )
            
            saved_paths = []
            for j, img in enumerate(images):
                filename = f"{prompt_id}_{j}.png"
                filepath = cat_dir / filename
                img.save(filepath)
                saved_paths.append(str(filepath))
            
            results[natural_prompt] = saved_paths
            total_images += len(images)
        
        generation_time = time.time() - start_time
        # total_images = sum(len(paths) for paths in results.values())
        
        logger.info(f"Generated {total_images} images in {generation_time:.2f} seconds")
        
        # Save generation metadata
        metadata = {
            'model_name': self.model_name,
            'device': self.device,
            'num_prompts': len(prompts_data),
            'num_images_per_prompt': num_images_per_prompt,
            'total_images': total_images,
            'generation_time': generation_time,
            'results': results
        }
        
        metadata_path = self.analysis_dir / "generation_metadata.json"
        # Convert numpy types to native Python types for JSON serialization
        metadata_serializable = convert_to_native_types(metadata)
        with open(metadata_path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Generation metadata saved to {metadata_path}")
        return metadata
    
    def classify_demographics(self, model_path: str = None) -> Dict[str, Any]:
        """Classify demographics in generated images."""
        logger.info("Initializing demographic classifier")
        self.classifier = FairFaceClassifier(model_path=model_path, device=self.device)
        
        # Get list of generated images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(self.images_dir.rglob(f"*{ext}"))
            image_paths.extend(self.images_dir.rglob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            raise ValueError(f"No images found in {self.images_dir}")
        
        logger.info(f"Classifying demographics for {len(image_paths)} images")
        start_time = time.time()
        
        # Classify images
        results = self.classifier.classify_batch(
            image_paths=image_paths,
            output_path=str(self.annotations_dir / "classification_results.json")
        )
        
        classification_time = time.time() - start_time
        
        logger.info(f"Classification completed in {classification_time:.2f} seconds")
        
        # Save classification metadata
        metadata = {
            'model_path': model_path,
            'device': self.device,
            'num_images': len(image_paths),
            'classification_time': classification_time,
            'results': results
        }
        
        metadata_path = self.analysis_dir / "classification_metadata.json"
        # Convert numpy types to native Python types for JSON serialization
        metadata_serializable = convert_to_native_types(metadata)
        with open(metadata_path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        
        logger.info(f"Classification metadata saved to {metadata_path}")
        return metadata
    
    def analyze_bias(self) -> Dict[str, Any]:
        """Analyze bias in the classification results."""
        logger.info("Analyzing bias in classification results")
        
        # Load classification results
        results_path = self.annotations_dir / "classification_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"Classification results not found at {results_path}")
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Perform comprehensive analysis
        start_time = time.time()
        analysis = self.analyzer.comprehensive_analysis(results)
        analysis_time = time.time() - start_time
        
        # Add metadata
        analysis['metadata'] = {
            'analysis_time': analysis_time,
            'num_samples': len(results),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': self.model_name
        }
        
        # Save analysis results
        analysis_path = self.analysis_dir / "bias_analysis.json"
        # Convert numpy types to native Python types for JSON serialization
        analysis_serializable = convert_to_native_types(analysis)
        with open(analysis_path, 'w') as f:
            json.dump(analysis_serializable, f, indent=2)
        
        logger.info(f"Bias analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"Analysis saved to {analysis_path}")
        
        return analysis
    
    def compare_results(self, other_analysis_path: str) -> Dict[str, Any]:
        """Compare current results with another analysis file."""
        logger.info(f"Comparing results with {other_analysis_path}")
        
        try:
            with open(other_analysis_path, 'r') as f:
                other_analysis = json.load(f)
            
            # Load current analysis
            current_analysis_path = self.analysis_dir / "bias_analysis.json"
            if not current_analysis_path.exists():
                logger.warning("Current analysis not found, skipping comparison")
                return {}
                
            with open(current_analysis_path, 'r') as f:
                current_analysis = json.load(f)
            
            comparison = {
                'model_comparison': {
                    'current_model': self.model_name,
                    'other_model': other_analysis.get('metadata', {}).get('model_name', 'Unknown')
                },
                'bias_score_diff': current_analysis.get('overall_bias_score', 0) - other_analysis.get('overall_bias_score', 0),
                'metrics_diff': {}
            }
            
            # Compare race bias
            if 'race_bias' in current_analysis and 'race_bias' in other_analysis:
                curr_race = current_analysis['race_bias']
                other_race = other_analysis['race_bias']
                comparison['metrics_diff']['race_bias_amplification'] = \
                    curr_race.get('bias_amplification_score', 0) - other_race.get('bias_amplification_score', 0)
            
            # Compare gender bias
            if 'gender_bias' in current_analysis and 'gender_bias' in other_analysis:
                curr_gender = current_analysis['gender_bias']
                other_gender = other_analysis['gender_bias']
                comparison['metrics_diff']['gender_bias_amplification'] = \
                    curr_gender.get('bias_amplification_score', 0) - other_gender.get('bias_amplification_score', 0)
            
            # Save comparison
            comparison_path = self.analysis_dir / "comparison_results.json"
            with open(comparison_path, 'w') as f:
                json.dump(comparison, f, indent=2)
                
            logger.info(f"Comparison saved to {comparison_path}")
            return comparison
            
        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            return {'error': str(e)}
    
    def create_visualizations(self, analysis: Dict[str, Any]) -> list:
        """Create visualizations of the analysis results."""
        logger.info("Creating visualizations")
        
        plots = self.visualizer.create_all_visualizations(analysis)
        
        logger.info(f"Created {len(plots)} visualizations")
        for plot in plots:
            logger.info(f"  - {os.path.basename(plot)}")
        
        return plots
    
    def generate_report(self, analysis: Dict[str, Any], comparison: Dict[str, Any] = None) -> str:
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive report")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"CSDS 447 - {self.model_name.upper()} FAIRNESS ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Model information
        report_lines.append("MODEL INFORMATION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Model: {self.model_name}")
        report_lines.append(f"Output Directory: {self.output_dir}")
        report_lines.append("")
        
        # Metadata
        metadata = analysis.get('metadata', {})
        report_lines.append("ANALYSIS METADATA:")
        report_lines.append("-" * 40)
        report_lines.append(f"Analysis Time: {metadata.get('analysis_time', 'N/A'):.2f} seconds")
        report_lines.append(f"Number of Samples: {metadata.get('num_samples', 'N/A')}")
        report_lines.append(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        report_lines.append("")
        
        # Race bias analysis
        race_bias = analysis.get('race_bias', {})
        if 'error' not in race_bias:
            report_lines.append("RACE BIAS ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Bias Amplification Score: {race_bias.get('bias_amplification_score', 0):.3f}")
            report_lines.append("")
            report_lines.append("Race Distribution:")
            for race, rate in race_bias.get('race_distribution', {}).items():
                report_lines.append(f"  {race}: {rate:.3f}")
            report_lines.append("")
        
        # Gender bias analysis
        gender_bias = analysis.get('gender_bias', {})
        if 'error' not in gender_bias:
            report_lines.append("GENDER BIAS ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(f"Statistical Parity Difference: {gender_bias.get('statistical_parity_difference', 0):.3f}")
            report_lines.append(f"Bias Amplification Score: {gender_bias.get('bias_amplification_score', 0):.3f}")
            report_lines.append("")
            report_lines.append("Gender Distribution:")
            for gender, rate in gender_bias.get('gender_distribution', {}).items():
                report_lines.append(f"  {gender}: {rate:.3f}")
            report_lines.append("")
        
        # Overall bias score
        overall_score = analysis.get('overall_bias_score', 0)
        report_lines.append("OVERALL BIAS ASSESSMENT:")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Bias Score: {overall_score:.3f}")
        
        if overall_score < 0.1:
            report_lines.append("Interpretation: LOW BIAS - Model shows minimal demographic bias")
        elif overall_score < 0.3:
            report_lines.append("Interpretation: MODERATE BIAS - Model shows some demographic bias")
        else:
            report_lines.append("Interpretation: HIGH BIAS - Model shows significant demographic bias")
        
        # Comparison section
        if comparison and 'error' not in comparison:
            report_lines.append("")
            report_lines.append("COMPARISON WITH PREVIOUS RUN:")
            report_lines.append("-" * 40)
            report_lines.append(f"Compared against model: {comparison['model_comparison']['other_model']}")
            
            score_diff = comparison['bias_score_diff']
            diff_str = f"{score_diff:+.3f}"
            report_lines.append(f"Overall Bias Score Difference: {diff_str}")
            
            if score_diff < 0:
                report_lines.append("Result: IMPROVEMENT (Bias decreased)")
            elif score_diff > 0:
                report_lines.append("Result: REGRESSION (Bias increased)")
            else:
                report_lines.append("Result: NO CHANGE")
                
            if 'metrics_diff' in comparison:
                report_lines.append("")
                report_lines.append("Metric Differences:")
                for metric, diff in comparison['metrics_diff'].items():
                    report_lines.append(f"  {metric}: {diff:+.3f}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.analysis_dir / "comprehensive_report.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {report_path}")
        return str(report_path)
    
    def run_complete_pipeline(self, prompts_file: str, 
                            num_images_per_prompt: int = 5, 
                            classifier_model_path: str = None,
                            compare_with_dir: str = None) -> Dict[str, Any]:
        """Run the complete fairness analysis pipeline for Stable Diffusion."""
        logger.info("=" * 80)
        logger.info(f"{self.model_name.upper()} FAIRNESS ANALYSIS PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Model: {self.model_name}")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # Step 1: Load prompts
            logger.info("\n[Step 1/6] Loading prompts...")
            prompts_data = self.load_prompts(prompts_file)
            
            # Step 2: Generate images
            logger.info(f"\n[Step 2/6] Generating images with {self.model_name}...")
            generation_metadata = self.generate_images(
                prompts_data, num_images_per_prompt
            )
            
            # Step 3: Classify demographics
            logger.info("\n[Step 3/6] Classifying demographics...")
            classification_metadata = self.classify_demographics(classifier_model_path)
            
            # Step 4: Analyze bias
            logger.info("\n[Step 4/6] Analyzing bias...")
            analysis = self.analyze_bias()
            
            # Step 5: Create visualizations
            logger.info("\n[Step 5/6] Creating visualizations...")
            plots = self.create_visualizations(analysis)
            
            # Step 6: Generate report (with comparison if requested)
            comparison = None
            if compare_with_dir:
                logger.info(f"\n[Step 6a] Comparing with {compare_with_dir}...")
                other_analysis_path = Path(compare_with_dir) / "analysis" / "bias_analysis.json"
                if other_analysis_path.exists():
                    comparison = self.compare_results(str(other_analysis_path))
                else:
                    logger.warning(f"Comparison file not found: {other_analysis_path}")

            logger.info("\n[Step 6b] Generating report...")
            report_path = self.generate_report(analysis, comparison)
            
            # Pipeline summary
            total_time = time.time() - start_time
            
            summary = {
                'status': 'success',
                'model': self.model_name,
                'total_time': total_time,
                'generation_metadata': generation_metadata,
                'classification_metadata': classification_metadata,
                'analysis': analysis,
                'comparison': comparison,
                'plots': plots,
                'report_path': report_path,
                'output_directory': str(self.output_dir)
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("=" * 80)
            logger.info(f"Total time: {total_time:.2f} seconds")
            logger.info(f"Output directory: {self.output_dir}")
            logger.info(f"Generated images: {generation_metadata['total_images']}")
            logger.info(f"Analysis plots: {len(plots)}")
            logger.info(f"Report: {report_path}")
            logger.info("=" * 80)
            
            return summary
            
        except Exception as e:
            logger.error(f"\nPipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            total_time = time.time() - start_time if 'start_time' in locals() else 0
            return {
                'status': 'error',
                'error': str(e),
                'total_time': total_time,
                'output_directory': str(self.output_dir)
            }

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Model 1: Stable Diffusion Fairness Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3 images per prompt)
  python m1_SD.py

  # Run with 5 images per prompt using SDXL
  python m1_SD_dec01.py --num-images 5 --model stable-diffusion-xl-base-1.0

  # Run on CPU
  python m1_SD.py --device cpu

  # Use custom prompts file and compare with previous results
  # negative_role_prompts_merged_clean_frontview.csv
  # test_prompts.csv
  python m1_SD_dec01.py --prompts custom_prompts.csv --compare-with m1_SD_result_dec01
        """
    )
    parser.add_argument("--prompts", default="negative_role_prompts_merged_clean_frontview.csv",
                       help="Path to prompts CSV file (default: negative_role_prompts_merged_clean_frontview.csv)")
    parser.add_argument("--num-images", type=int, default=3,
                       help="Number of images per prompt (default: 3)")
    parser.add_argument("--device", default="auto",
                       help="Device to use (auto, cpu, cuda, mps) (default: auto)")
    parser.add_argument("--classifier-model", 
                       help="Path to pre-trained classifier model (optional)")
    parser.add_argument("--output", default="m1_SD_result_dec01",
                       help="Output directory for results (default: m1_SD_result_dec01)")
    parser.add_argument("--model", default="stable-diffusion-v1-5",
                       help="Model to use (default: stable-diffusion-v1-5)")
    parser.add_argument("--compare-with",
                       help="Directory of previous results to compare with")
    
    args = parser.parse_args()
    
    # Check if prompts file exists
    if not os.path.exists(args.prompts):
        logger.error(f"Prompts file not found: {args.prompts}")
        logger.error("Please ensure the prompts CSV file exists in the current directory.")
        sys.exit(1)
    
    # Initialize and run pipeline
    pipeline = StableDiffusionPipeline(
        output_dir=args.output,
        device=args.device,
        model_name=args.model
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        prompts_file=args.prompts,
        num_images_per_prompt=args.num_images,
        classifier_model_path=args.classifier_model,
        compare_with_dir=args.compare_with
    )
    
    # Print summary
    if results['status'] == 'success':
        print("\n" + "="*80)
        print(f"{results['model'].upper()} ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Model: {results['model']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Output directory: {results['output_directory']}")
        print(f"Generated images: {results['generation_metadata']['total_images']}")
        print(f"Analysis plots: {len(results['plots'])}")
        print(f"Report: {results['report_path']}")
        print("\nAll results saved to:", results['output_directory'])
        print("="*80)
    else:
        print(f"\n{pipeline.model_name.upper()} ANALYSIS FAILED: {results['error']}")
        print(f"Output directory: {results.get('output_directory', 'N/A')}")
        sys.exit(1)

if __name__ == "__main__":
    main()

