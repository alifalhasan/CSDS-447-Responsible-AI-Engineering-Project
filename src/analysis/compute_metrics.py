#!/usr/bin/env python3
"""
Bias Analysis and Metrics Computation Module

This module implements various fairness metrics and bias analysis tools
for evaluating demographic bias in text-to-image models.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_native_types(obj):
    """Convert numpy types and other non-JSON-serializable types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_native_types(item) for item in obj)
    elif hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    else:
        return obj


class FairnessMetrics:
    """Compute various fairness and bias metrics."""

    def __init__(self):
        self.race_labels = [
            "White",
            "Black",
            "Indian",
            "East Asian",
            "Southeast Asian",
            "Middle Eastern",
            "Latino",
        ]
        self.gender_labels = ["Male", "Female"]
        self.age_labels = [
            "0-2",
            "3-9",
            "10-19",
            "20-29",
            "30-39",
            "40-49",
            "50-59",
            "60-69",
            "70+",
        ]

    def statistical_parity_difference(
        self, predictions: List[str], protected_attribute: str, positive_outcome: str
    ) -> float:
        """
        Compute Statistical Parity Difference (SPD).

        SPD = P(Y=1|A=protected) - P(Y=1|A=not_protected)

        Args:
            predictions: List of predicted demographic groups
            protected_attribute: The protected demographic group
            positive_outcome: The positive outcome (e.g., "criminal")

        Returns:
            SPD value
        """
        protected_count = sum(1 for p in predictions if p == protected_attribute)
        non_protected_count = sum(1 for p in predictions if p != protected_attribute)

        total = len(predictions)
        if total == 0:
            return 0.0

        protected_rate = protected_count / total
        non_protected_rate = non_protected_count / total

        return protected_rate - non_protected_rate

    def equalized_odds_difference(
        self, y_true: List[str], y_pred: List[str], protected_attribute: str
    ) -> float:
        """
        Compute Equalized Odds Difference (EOD).

        EOD = max(|TPR_protected - TPR_non_protected|, |FPR_protected - FPR_non_protected|)

        Args:
            y_true: True labels
            y_pred: Predicted labels
            protected_attribute: The protected demographic group

        Returns:
            EOD value
        """
        # This is a simplified version - in practice, you'd need ground truth labels
        # For demo purposes, we'll compute based on prediction distributions
        protected_preds = [
            p for i, p in enumerate(y_pred) if y_true[i] == protected_attribute
        ]
        non_protected_preds = [
            p for i, p in enumerate(y_pred) if y_true[i] != protected_attribute
        ]

        if not protected_preds or not non_protected_preds:
            return 0.0

        # Compute prediction rates (simplified)
        protected_positive_rate = sum(
            1 for p in protected_preds if p == protected_attribute
        ) / len(protected_preds)
        non_protected_positive_rate = sum(
            1 for p in non_protected_preds if p != protected_attribute
        ) / len(non_protected_preds)

        return abs(protected_positive_rate - non_protected_positive_rate)

    def demographic_parity(
        self, predictions: List[str], demographic_groups: List[str]
    ) -> Dict[str, float]:
        """
        Compute demographic parity for each group.

        Args:
            predictions: List of predicted demographic groups
            demographic_groups: List of demographic group labels

        Returns:
            Dictionary mapping groups to their representation rates
        """
        total = len(predictions)
        if total == 0:
            return {group: 0.0 for group in demographic_groups}

        parity = {}
        for group in demographic_groups:
            count = sum(1 for p in predictions if p == group)
            parity[group] = count / total

        return parity

    def bias_amplification_score(
        self,
        model_distribution: Dict[str, float],
        real_world_distribution: Dict[str, float],
    ) -> float:
        """
        Compute bias amplification score.

        Measures how much the model amplifies existing biases compared to real-world data.

        Args:
            model_distribution: Model's demographic distribution
            real_world_distribution: Real-world demographic distribution

        Returns:
            Bias amplification score (higher = more amplification)
        """
        amplification = 0.0
        total_groups = 0

        for group in model_distribution:
            if group in real_world_distribution:
                model_rate = model_distribution[group]
                real_rate = real_world_distribution[group]

                if real_rate > 0:
                    amplification += abs(model_rate - real_rate) / real_rate
                    total_groups += 1

        return amplification / total_groups if total_groups > 0 else 0.0

    def intersectional_bias_analysis(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze bias at the intersection of multiple demographic attributes.

        Args:
            results: List of classification results with race, gender, age predictions

        Returns:
            Intersectional bias analysis
        """
        intersectional_groups = {}

        for result in results:
            if "error" in result:
                continue

            race = result.get("race", {}).get("prediction", "Unknown")
            gender = result.get("gender", {}).get("prediction", "Unknown")
            age = result.get("age", {}).get("prediction", "Unknown")

            intersection_key = f"{race}_{gender}_{age}"
            if intersection_key not in intersectional_groups:
                intersectional_groups[intersection_key] = 0
            intersectional_groups[intersection_key] += 1

        total = sum(intersectional_groups.values())
        intersectional_rates = {k: v / total for k, v in intersectional_groups.items()}

        return {
            "intersectional_groups": intersectional_groups,
            "intersectional_rates": intersectional_rates,
            "total_samples": total,
        }


class BiasAnalyzer:
    """Main bias analysis class."""

    def __init__(self):
        self.metrics = FairnessMetrics()

    def load_classification_results(self, results_path: str) -> List[Dict[str, Any]]:
        """Load classification results from JSON file."""
        with open(results_path, "r") as f:
            return json.load(f)

    def analyze_race_bias(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze racial bias in the results."""
        race_predictions = []
        for result in results:
            if "error" not in result and "race" in result:
                race_pred = result["race"]["prediction"]
                # Filter out Unknown predictions
                if race_pred and race_pred != "Unknown":
                    race_predictions.append(race_pred)

        if not race_predictions:
            return {"error": "No valid race predictions found"}

        # Compute demographic parity
        race_parity = self.metrics.demographic_parity(
            race_predictions, self.metrics.race_labels
        )

        # Compute statistical parity differences
        spd_results = {}
        for i, race1 in enumerate(self.metrics.race_labels):
            for race2 in self.metrics.race_labels[i + 1 :]:
                spd = self.metrics.statistical_parity_difference(
                    race_predictions, race1, race2
                )
                spd_results[f"{race1}_vs_{race2}"] = spd

        # Source: U.S. Census Bureau (Approximate mapping to FairFace labels)
        real_world_race = {
            "White": 0.59,
            "Black": 0.13,
            "Latino": 0.19,
            "Indian": 0.02,
            "East Asian": 0.03,
            "Southeast Asian": 0.03,
            "Middle Eastern": 0.01,
        }

        # Compute bias amplification
        bias_amplification = self.metrics.bias_amplification_score(
            race_parity, real_world_race
        )

        return {
            "race_distribution": race_parity,
            "statistical_parity_differences": spd_results,
            "bias_amplification_score": bias_amplification,
            "real_world_distribution": real_world_race,
            "total_samples": len(race_predictions),
        }

    def analyze_gender_bias(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze gender bias in the results."""
        gender_predictions = []
        for result in results:
            if "error" not in result and "gender" in result:
                gender_pred = result["gender"]["prediction"]
                # Filter out Unknown predictions
                if gender_pred and gender_pred != "Unknown":
                    gender_predictions.append(gender_pred)

        if not gender_predictions:
            return {"error": "No valid gender predictions found"}

        # Compute demographic parity
        gender_parity = self.metrics.demographic_parity(
            gender_predictions, self.metrics.gender_labels
        )

        # Compute statistical parity difference
        spd = self.metrics.statistical_parity_difference(
            gender_predictions, "Male", "Female"
        )

        # Real-world distribution (approximately equal)
        real_world_gender = {"Male": 0.50, "Female": 0.50}

        # Compute bias amplification
        bias_amplification = self.metrics.bias_amplification_score(
            gender_parity, real_world_gender
        )

        return {
            "gender_distribution": gender_parity,
            "statistical_parity_difference": spd,
            "bias_amplification_score": bias_amplification,
            "real_world_distribution": real_world_gender,
            "total_samples": len(gender_predictions),
        }

    def analyze_age_bias(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze age bias in the results."""
        age_predictions = []
        for result in results:
            if "error" not in result and "age" in result:
                age_pred = result["age"]["prediction"]
                # Filter out Unknown predictions
                if age_pred and age_pred != "Unknown":
                    age_predictions.append(age_pred)

        if not age_predictions:
            return {"error": "No valid age predictions found"}

        # Compute demographic parity
        age_parity = self.metrics.demographic_parity(
            age_predictions, self.metrics.age_labels
        )

        # Source: U.S. Census Bureau, Population Estimates Program, Vintage 2024.
        real_world_age = {
            "0-19": 0.24,
            "20-39": 0.27,
            "40-59": 0.26,
            "60+": 0.23,
        }

        # Aggregate model predictions to match real-world buckets
        age_buckets = {
            "0-19": ["0-2", "3-9", "10-19"],
            "20-39": ["20-29", "30-39"],
            "40-59": ["40-49", "50-59"],
            "60+": ["60-69", "70+"]
        }
        
        aggregated_parity = {}
        for bucket, labels in age_buckets.items():
            aggregated_parity[bucket] = sum(age_parity.get(label, 0) for label in labels)

        # Compute bias amplification using aggregated parity
        bias_amplification = self.metrics.bias_amplification_score(
            aggregated_parity, real_world_age
        )

        return {
            "age_distribution": aggregated_parity,
            "bias_amplification_score": bias_amplification,
            "real_world_distribution": real_world_age,
            "total_samples": len(age_predictions),
        }

    def comprehensive_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform comprehensive bias analysis."""
        analysis = {
            "race_bias": self.analyze_race_bias(results),
            "gender_bias": self.analyze_gender_bias(results),
            "age_bias": self.analyze_age_bias(results),
            "intersectional_bias": self.metrics.intersectional_bias_analysis(results),
        }

        # Overall bias score
        bias_scores = []
        if "bias_amplification_score" in analysis["race_bias"]:
            bias_scores.append(analysis["race_bias"]["bias_amplification_score"])
        if "bias_amplification_score" in analysis["gender_bias"]:
            bias_scores.append(analysis["gender_bias"]["bias_amplification_score"])
        if "bias_amplification_score" in analysis["age_bias"]:
            bias_scores.append(analysis["age_bias"]["bias_amplification_score"])

        analysis["overall_bias_score"] = np.mean(bias_scores) if bias_scores else 0.0

        return analysis


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze bias in classification results"
    )
    parser.add_argument(
        "--annotations", required=True, help="Path to classification results JSON file"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for analysis results"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Initialize analyzer
    analyzer = BiasAnalyzer()

    # Load results
    logger.info(f"Loading classification results from {args.annotations}")
    results = analyzer.load_classification_results(args.annotations)

    # Perform comprehensive analysis
    logger.info("Performing comprehensive bias analysis...")
    analysis = analyzer.comprehensive_analysis(results)

    # Save analysis results
    analysis_path = os.path.join(args.output, "bias_analysis.json")
    # Convert numpy types to native Python types for JSON serialization
    analysis_serializable = convert_to_native_types(analysis)
    with open(analysis_path, "w") as f:
        json.dump(analysis_serializable, f, indent=2)

    # Generate summary report
    summary = generate_summary_report(analysis)
    summary_path = os.path.join(args.output, "bias_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary)

    logger.info(f"Bias analysis complete! Results saved to {args.output}")
    logger.info(f"Analysis saved to {analysis_path}")
    logger.info(f"Summary saved to {summary_path}")


def generate_summary_report(analysis: Dict[str, Any]) -> str:
    """Generate a human-readable summary report."""
    report = []
    report.append("=" * 60)
    report.append("BIAS ANALYSIS SUMMARY REPORT")
    report.append("=" * 60)
    report.append("")

    # Race bias summary
    if "race_bias" in analysis and "error" not in analysis["race_bias"]:
        race_bias = analysis["race_bias"]
        report.append("RACE BIAS ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Total samples: {race_bias.get('total_samples', 'N/A')}")
        report.append(
            f"Bias amplification score: {race_bias.get('bias_amplification_score', 0):.3f}"
        )
        report.append("")
        report.append("Race distribution:")
        for race, rate in race_bias.get("race_distribution", {}).items():
            report.append(f"  {race}: {rate:.3f}")
        report.append("")

    # Gender bias summary
    if "gender_bias" in analysis and "error" not in analysis["gender_bias"]:
        gender_bias = analysis["gender_bias"]
        report.append("GENDER BIAS ANALYSIS:")
        report.append("-" * 30)
        report.append(f"Total samples: {gender_bias.get('total_samples', 'N/A')}")
        report.append(
            f"Statistical parity difference: {gender_bias.get('statistical_parity_difference', 0):.3f}"
        )
        report.append(
            f"Bias amplification score: {gender_bias.get('bias_amplification_score', 0):.3f}"
        )
        report.append("")
        report.append("Gender distribution:")
        for gender, rate in gender_bias.get("gender_distribution", {}).items():
            report.append(f"  {gender}: {rate:.3f}")
        report.append("")

    # Overall summary
    overall_score = analysis.get("overall_bias_score", 0)
    report.append("OVERALL BIAS SCORE:")
    report.append("-" * 30)
    report.append(f"Overall bias score: {overall_score:.3f}")

    if overall_score < 0.1:
        report.append("Interpretation: Low bias detected")
    elif overall_score < 0.3:
        report.append("Interpretation: Moderate bias detected")
    else:
        report.append("Interpretation: High bias detected")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    main()
