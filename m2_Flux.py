#!/usr/bin/env python3
"""
Model 2: FLUX.1-dev Pipeline for Negative Role Fairness Analysis

This script runs the complete pipeline for analyzing bias in FLUX.1-dev
text-to-image model when generating images of negative social roles.

All results are saved to m2_FLUX_result/ directory.

Usage:
    python m2_FLUX.py --num-images 3 --device auto
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
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from generation.generate_images import ImageGenerator, load_prompts_from_csv
from classification.classify_demographics import FairFaceClassifier
from analysis.compute_metrics import BiasAnalyzer, convert_to_native_types
from visualization.plot_results import BiasVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class FLUXPipeline:
    """Complete pipeline for FLUX.1-dev negative role fairness analysis."""

    def __init__(self, output_dir: str = "m2_FLUX_result", device: str = "auto"):
        """
        Initialize the pipeline.

        Args:
            output_dir: Directory to save all results (default: m2_FLUX_result)
            device: Device to use for computation
        """
        self.output_dir = Path(output_dir)
        self.device = device
        self.model_name = "black-forest-labs/FLUX.1-dev"  # FLUX.1-dev model

        # Create output directories
        self.images_dir = self.output_dir / "generated_images"
        self.annotations_dir = self.output_dir / "annotations"
        self.analysis_dir = self.output_dir / "analysis"
        self.figures_dir = self.output_dir / "figures"

        for dir_path in [
            self.images_dir,
            self.annotations_dir,
            self.analysis_dir,
            self.figures_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.image_generator = None
        self.classifier = None
        self.analyzer = BiasAnalyzer()
        self.visualizer = BiasVisualizer(str(self.figures_dir))

        logger.info(f"FLUX.1-dev Pipeline initialized")
        logger.info(f"Model: {self.model_name}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_prompts(self, prompts_file: str) -> list:
        """Load prompts from CSV file."""
        logger.info(f"Loading prompts from {prompts_file}")
        prompts_data = load_prompts_from_csv(prompts_file)
        logger.info(f"Loaded {len(prompts_data)} prompts")
        return prompts_data

    def generate_images(
        self, prompts_data: list, num_images_per_prompt: int = 5
    ) -> Dict[str, Any]:
        """Generate images using FLUX.1-dev."""
        logger.info(f"Initializing FLUX.1-dev image generator")
        self.image_generator = ImageGenerator(
            model_name=self.model_name,
            device=self.device,
            use_refiner=False,  # FLUX doesn't use a separate refiner
        )

        # Extract prompts
        prompts = [p["natural_prompt"] for p in prompts_data]

        logger.info(
            f"Generating {num_images_per_prompt} images per prompt for {len(prompts)} prompts"
        )
        start_time = time.time()

        results = self.image_generator.generate_batch(
            prompts=prompts,
            output_dir=str(self.images_dir),
            num_images_per_prompt=num_images_per_prompt,
            seed=42,  # For reproducibility
        )

        generation_time = time.time() - start_time
        total_images = sum(len(paths) for paths in results.values())

        logger.info(f"Generated {total_images} images in {generation_time:.2f} seconds")

        # Save generation metadata
        metadata = {
            "model_name": self.model_name,
            "device": self.device,
            "num_prompts": len(prompts),
            "num_images_per_prompt": num_images_per_prompt,
            "total_images": total_images,
            "generation_time": generation_time,
            "results": results,
        }

        metadata_path = self.analysis_dir / "generation_metadata.json"
        # Convert numpy types to native Python types for JSON serialization
        metadata_serializable = convert_to_native_types(metadata)
        with open(metadata_path, "w") as f:
            json.dump(metadata_serializable, f, indent=2)

        logger.info(f"Generation metadata saved to {metadata_path}")
        return metadata

    def classify_demographics(self, model_path: str = None) -> Dict[str, Any]:
        """Classify demographics in generated images."""
        logger.info("Initializing demographic classifier")
        self.classifier = FairFaceClassifier(model_path=model_path, device=self.device)

        # Get list of generated images
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_paths = []

        for ext in image_extensions:
            image_paths.extend(self.images_dir.glob(f"*{ext}"))
            image_paths.extend(self.images_dir.glob(f"*{ext.upper()}"))

        image_paths = [str(p) for p in image_paths]

        if not image_paths:
            raise ValueError(f"No images found in {self.images_dir}")

        logger.info(f"Classifying demographics for {len(image_paths)} images")
        start_time = time.time()

        # Classify images
        results = self.classifier.classify_batch(
            image_paths=image_paths,
            output_path=str(self.annotations_dir / "classification_results.json"),
        )

        classification_time = time.time() - start_time

        logger.info(f"Classification completed in {classification_time:.2f} seconds")

        # Save classification metadata
        metadata = {
            "model_path": model_path,
            "device": self.device,
            "num_images": len(image_paths),
            "classification_time": classification_time,
            "results": results,
        }

        metadata_path = self.analysis_dir / "classification_metadata.json"
        # Convert numpy types to native Python types for JSON serialization
        metadata_serializable = convert_to_native_types(metadata)
        with open(metadata_path, "w") as f:
            json.dump(metadata_serializable, f, indent=2)

        logger.info(f"Classification metadata saved to {metadata_path}")
        return metadata

    def analyze_bias(self) -> Dict[str, Any]:
        """Analyze bias in the classification results."""
        logger.info("Analyzing bias in classification results")

        # Load classification results
        results_path = self.annotations_dir / "classification_results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"Classification results not found at {results_path}"
            )

        with open(results_path, "r") as f:
            results = json.load(f)

        # Perform comprehensive analysis
        start_time = time.time()
        analysis = self.analyzer.comprehensive_analysis(results)
        analysis_time = time.time() - start_time

        # Add metadata
        analysis["metadata"] = {
            "analysis_time": analysis_time,
            "num_samples": len(results),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": self.model_name,
        }

        # Save analysis results
        analysis_path = self.analysis_dir / "bias_analysis.json"
        # Convert numpy types to native Python types for JSON serialization
        analysis_serializable = convert_to_native_types(analysis)
        with open(analysis_path, "w") as f:
            json.dump(analysis_serializable, f, indent=2)

        logger.info(f"Bias analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"Analysis saved to {analysis_path}")

        return analysis

    def create_visualizations(self, analysis: Dict[str, Any]) -> list:
        """Create visualizations of the analysis results."""
        logger.info("Creating visualizations")

        plots = self.visualizer.create_all_visualizations(analysis)

        logger.info(f"Created {len(plots)} visualizations")
        for plot in plots:
            logger.info(f"  - {os.path.basename(plot)}")

        return plots

    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a comprehensive analysis report."""
        logger.info("Generating comprehensive report")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("CSDS 447 - FLUX.1-dev FAIRNESS ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Model information
        report_lines.append("MODEL INFORMATION:")
        report_lines.append("-" * 40)
        report_lines.append(f"Model: {self.model_name}")
        report_lines.append(f"Output Directory: {self.output_dir}")
        report_lines.append("")

        # Metadata
        metadata = analysis.get("metadata", {})
        report_lines.append("ANALYSIS METADATA:")
        report_lines.append("-" * 40)
        report_lines.append(
            f"Analysis Time: {metadata.get('analysis_time', 'N/A'):.2f} seconds"
        )
        report_lines.append(f"Number of Samples: {metadata.get('num_samples', 'N/A')}")
        report_lines.append(f"Timestamp: {metadata.get('timestamp', 'N/A')}")
        report_lines.append("")

        # Race bias analysis
        race_bias = analysis.get("race_bias", {})
        if "error" not in race_bias:
            report_lines.append("RACE BIAS ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(
                f"Bias Amplification Score: {race_bias.get('bias_amplification_score', 0):.3f}"
            )
            report_lines.append("")
            report_lines.append("Race Distribution:")
            for race, rate in race_bias.get("race_distribution", {}).items():
                report_lines.append(f"  {race}: {rate:.3f}")
            report_lines.append("")

        # Gender bias analysis
        gender_bias = analysis.get("gender_bias", {})
        if "error" not in gender_bias:
            report_lines.append("GENDER BIAS ANALYSIS:")
            report_lines.append("-" * 40)
            report_lines.append(
                f"Statistical Parity Difference: {gender_bias.get('statistical_parity_difference', 0):.3f}"
            )
            report_lines.append(
                f"Bias Amplification Score: {gender_bias.get('bias_amplification_score', 0):.3f}"
            )
            report_lines.append("")
            report_lines.append("Gender Distribution:")
            for gender, rate in gender_bias.get("gender_distribution", {}).items():
                report_lines.append(f"  {gender}: {rate:.3f}")
            report_lines.append("")

        # Overall bias score
        overall_score = analysis.get("overall_bias_score", 0)
        report_lines.append("OVERALL BIAS ASSESSMENT:")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Bias Score: {overall_score:.3f}")

        if overall_score < 0.1:
            report_lines.append(
                "Interpretation: LOW BIAS - Model shows minimal demographic bias"
            )
        elif overall_score < 0.3:
            report_lines.append(
                "Interpretation: MODERATE BIAS - Model shows some demographic bias"
            )
        else:
            report_lines.append(
                "Interpretation: HIGH BIAS - Model shows significant demographic bias"
            )

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_content = "\n".join(report_lines)
        report_path = self.analysis_dir / "comprehensive_report.txt"
        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def run_complete_pipeline(
        self,
        prompts_file: str,
        num_images_per_prompt: int = 5,
        classifier_model_path: str = None,
    ) -> Dict[str, Any]:
        """Run the complete fairness analysis pipeline for FLUX.1-dev."""
        logger.info("=" * 80)
        logger.info("FLUX.1-dev FAIRNESS ANALYSIS PIPELINE")
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
            logger.info("\n[Step 2/6] Generating images with FLUX.1-dev...")
            generation_metadata = self.generate_images(
                prompts_data, num_images_per_prompt
            )

            # Step 3: Classify demographics
            logger.info("\n[Step 3/6] Classifying demographics...")

            # Clear PyTorch GPU memory before TensorFlow classification
            if self.device == "cuda":
                import torch

                torch.cuda.empty_cache()
                logger.info("Cleared PyTorch GPU cache")

            # Force TensorFlow to use CPU to avoid GPU conflicts
            import os

            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            logger.info(
                "Set classification to use CPU (avoiding TensorFlow/PyTorch GPU conflict)"
            )

            classification_metadata = self.classify_demographics(classifier_model_path)

            # Step 4: Analyze bias
            logger.info("\n[Step 4/6] Analyzing bias...")
            analysis = self.analyze_bias()

            # Step 5: Create visualizations
            logger.info("\n[Step 5/6] Creating visualizations...")
            plots = self.create_visualizations(analysis)

            # Step 6: Generate report
            logger.info("\n[Step 6/6] Generating report...")
            report_path = self.generate_report(analysis)

            # Pipeline summary
            total_time = time.time() - start_time

            summary = {
                "status": "success",
                "model": self.model_name,
                "total_time": total_time,
                "generation_metadata": generation_metadata,
                "classification_metadata": classification_metadata,
                "analysis": analysis,
                "plots": plots,
                "report_path": report_path,
                "output_directory": str(self.output_dir),
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
            total_time = time.time() - start_time if "start_time" in locals() else 0
            return {
                "status": "error",
                "error": str(e),
                "total_time": total_time,
                "output_directory": str(self.output_dir),
            }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Model 2: FLUX.1-dev Fairness Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (3 images per prompt)
  python m2_FLUX.py

  # Run with 5 images per prompt
  python m2_FLUX.py --num-images 5

  # Run on CPU
  python m2_FLUX.py --device cpu

  # Use custom prompts file
  python m2_FLUX.py --prompts custom_prompts.csv
        """,
    )
    parser.add_argument(
        "--prompts",
        default="negative_role_prompts_all.csv",
        help="Path to prompts CSV file (default: negative_role_prompts_all.csv)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Number of images per prompt (default: 3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cpu, cuda, mps) (default: auto)",
    )
    parser.add_argument(
        "--classifier-model", help="Path to pre-trained classifier model (optional)"
    )
    parser.add_argument(
        "--output",
        default="m2_FLUX_result",
        help="Output directory for results (default: m2_FLUX_result)",
    )

    args = parser.parse_args()

    # Check if prompts file exists
    if not os.path.exists(args.prompts):
        logger.error(f"Prompts file not found: {args.prompts}")
        logger.error(
            "Please ensure the prompts CSV file exists in the current directory."
        )
        sys.exit(1)

    # Initialize and run pipeline
    pipeline = FLUXPipeline(output_dir=args.output, device=args.device)

    # Run complete pipeline
    results = pipeline.run_complete_pipeline(
        prompts_file=args.prompts,
        num_images_per_prompt=args.num_images,
        classifier_model_path=args.classifier_model,
    )

    # Print summary
    if results["status"] == "success":
        print("\n" + "=" * 80)
        print("FLUX.1-dev ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Model: {results['model']}")
        print(f"Total time: {results['total_time']:.2f} seconds")
        print(f"Output directory: {results['output_directory']}")
        print(f"Generated images: {results['generation_metadata']['total_images']}")
        print(f"Analysis plots: {len(results['plots'])}")
        print(f"Report: {results['report_path']}")
        print("\nAll results saved to:", results["output_directory"])
        print("=" * 80)
    else:
        print(f"\nFLUX.1-dev ANALYSIS FAILED: {results['error']}")
        print(f"Output directory: {results.get('output_directory', 'N/A')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
