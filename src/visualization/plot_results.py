#!/usr/bin/env python3
"""
Visualization Module for Bias Analysis Results

This module creates various plots and visualizations to analyze and
present bias analysis results from text-to-image models.
"""

import os
import json
import argparse
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BiasVisualizer:
    """Create visualizations for bias analysis results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up color schemes
        self.race_colors = {
            'White': '#FF6B6B',
            'Black': '#4ECDC4', 
            'Indian': '#45B7D1',
            'East Asian': '#96CEB4',
            'Southeast Asian': '#FFEAA7',
            'Middle Eastern': '#DDA0DD',
            'Latino': '#98D8C8'
        }
        
        self.gender_colors = {
            'Male': '#3498DB',
            'Female': '#E74C3C'
        }
    
    def plot_demographic_distribution(self, analysis: Dict[str, Any], 
                                    attribute: str, title: str) -> str:
        """
        Create a bar plot of demographic distribution.
        
        Args:
            analysis: Bias analysis results
            attribute: Demographic attribute ('race', 'gender', 'age')
            title: Plot title
            
        Returns:
            Path to saved plot
        """
        bias_key = f"{attribute}_bias"
        if bias_key not in analysis or 'error' in analysis[bias_key]:
            logger.warning(f"No {attribute} bias data available")
            return None
        
        distribution = analysis[bias_key].get(f"{attribute}_distribution", {})
        real_world = analysis[bias_key].get('real_world_distribution', {})
        
        if not distribution:
            logger.warning(f"No {attribute} distribution data available")
            return None
        
        # Create DataFrame for plotting
        groups = list(distribution.keys())
        model_rates = [distribution[g] for g in groups]
        real_rates = [real_world.get(g, 0) for g in groups]
        
        df = pd.DataFrame({
            'Group': groups,
            'Model': model_rates,
            'Real World': real_rates
        })
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(groups))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, model_rates, width, label='Model', alpha=0.8)
        bars2 = ax.bar(x + width/2, real_rates, width, label='Real World', alpha=0.8)
        
        ax.set_xlabel(f'{attribute.title()} Groups')
        ax.set_ylabel('Proportion')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        filename = f"{attribute}_distribution.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {attribute} distribution plot to {filepath}")
        return filepath
    
    def plot_statistical_parity_differences(self, analysis: Dict[str, Any]) -> str:
        """Create a heatmap of statistical parity differences."""
        race_bias = analysis.get('race_bias', {})
        if 'error' in race_bias:
            logger.warning("No race bias data available for SPD plot")
            return None
        
        spd_data = race_bias.get('statistical_parity_differences', {})
        if not spd_data:
            logger.warning("No statistical parity difference data available")
            return None
        
        # Create SPD matrix
        race_labels = list(self.race_colors.keys())
        spd_matrix = np.zeros((len(race_labels), len(race_labels)))
        
        for i, race1 in enumerate(race_labels):
            for j, race2 in enumerate(race_labels):
                if i != j:
                    key = f"{race1}_vs_{race2}" if f"{race1}_vs_{race2}" in spd_data else f"{race2}_vs_{race1}"
                    if key in spd_data:
                        spd_matrix[i, j] = spd_data[key]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(spd_matrix, cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(race_labels)))
        ax.set_yticks(np.arange(len(race_labels)))
        ax.set_xticklabels(race_labels, rotation=45, ha='right')
        ax.set_yticklabels(race_labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Statistical Parity Difference', rotation=270, labelpad=20)
        
        # Add text annotations
        for i in range(len(race_labels)):
            for j in range(len(race_labels)):
                if i != j:
                    text = ax.text(j, i, f'{spd_matrix[i, j]:.3f}',
                                 ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Statistical Parity Differences Between Race Groups')
        plt.tight_layout()
        
        # Save plot
        filename = "statistical_parity_differences.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SPD heatmap to {filepath}")
        return filepath
    
    def plot_bias_amplification_comparison(self, analysis: Dict[str, Any]) -> str:
        """Create a comparison plot of bias amplification scores."""
        attributes = ['race', 'gender', 'age']
        bias_scores = []
        labels = []
        
        for attr in attributes:
            bias_key = f"{attr}_bias"
            if bias_key in analysis and 'error' not in analysis[bias_key]:
                score = analysis[bias_key].get('bias_amplification_score', 0)
                bias_scores.append(score)
                labels.append(attr.title())
        
        if not bias_scores:
            logger.warning("No bias amplification scores available")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(labels, bias_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
        
        ax.set_ylabel('Bias Amplification Score')
        ax.set_title('Bias Amplification Across Demographic Attributes')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, bias_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Add threshold lines
        ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, label='Low Bias Threshold')
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='High Bias Threshold')
        ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        filename = "bias_amplification_comparison.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bias amplification comparison to {filepath}")
        return filepath
    
    def plot_intersectional_analysis(self, analysis: Dict[str, Any]) -> str:
        """Create a visualization of intersectional bias analysis."""
        intersectional = analysis.get('intersectional_bias', {})
        if not intersectional:
            logger.warning("No intersectional bias data available")
            return None
        
        intersectional_rates = intersectional.get('intersectional_rates', {})
        if not intersectional_rates:
            logger.warning("No intersectional rates data available")
            return None
        
        # Sort by rate for better visualization
        sorted_rates = sorted(intersectional_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 20 for readability
        top_20 = sorted_rates[:20]
        groups, rates = zip(*top_20)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        bars = ax.barh(range(len(groups)), rates, color='skyblue', alpha=0.7)
        
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_xlabel('Representation Rate')
        ax.set_title('Top 20 Intersectional Groups by Representation Rate')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, rate) in enumerate(zip(bars, rates)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                   f'{rate:.3f}', ha='left', va='center', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        filename = "intersectional_analysis.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved intersectional analysis to {filepath}")
        return filepath
    
    def create_interactive_dashboard(self, analysis: Dict[str, Any]) -> str:
        """Create an interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Race Distribution', 'Gender Distribution', 
                          'Bias Amplification Scores', 'Statistical Parity Differences'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "heatmap"}]]
        )
        
        # Race distribution
        race_bias = analysis.get('race_bias', {})
        if 'error' not in race_bias:
            race_dist = race_bias.get('race_distribution', {})
            if race_dist:
                fig.add_trace(
                    go.Bar(x=list(race_dist.keys()), y=list(race_dist.values()),
                          name='Race Distribution', marker_color='lightblue'),
                    row=1, col=1
                )
        
        # Gender distribution
        gender_bias = analysis.get('gender_bias', {})
        if 'error' not in gender_bias:
            gender_dist = gender_bias.get('gender_distribution', {})
            if gender_dist:
                fig.add_trace(
                    go.Bar(x=list(gender_dist.keys()), y=list(gender_dist.values()),
                          name='Gender Distribution', marker_color='lightcoral'),
                    row=1, col=2
                )
        
        # Bias amplification scores
        attributes = ['race', 'gender', 'age']
        bias_scores = []
        labels = []
        
        for attr in attributes:
            bias_key = f"{attr}_bias"
            if bias_key in analysis and 'error' not in analysis[bias_key]:
                score = analysis[bias_key].get('bias_amplification_score', 0)
                bias_scores.append(score)
                labels.append(attr.title())
        
        if bias_scores:
            fig.add_trace(
                go.Bar(x=labels, y=bias_scores, name='Bias Amplification',
                      marker_color='lightgreen'),
                row=2, col=1
            )
        
        # Statistical parity differences (simplified)
        if 'error' not in race_bias:
            spd_data = race_bias.get('statistical_parity_differences', {})
            if spd_data:
                # Create a simple bar chart for SPD
                spd_values = list(spd_data.values())
                spd_labels = list(spd_data.keys())
                
                fig.add_trace(
                    go.Bar(x=spd_labels, y=spd_values, name='SPD',
                          marker_color='lightyellow'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            title_text="Bias Analysis Dashboard",
            showlegend=False,
            height=800
        )
        
        # Save interactive dashboard
        filename = "bias_analysis_dashboard.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)
        
        logger.info(f"Saved interactive dashboard to {filepath}")
        return filepath
    
    def create_all_visualizations(self, analysis: Dict[str, Any]) -> List[str]:
        """Create all available visualizations."""
        plots = []
        
        # Demographic distribution plots
        for attr in ['race', 'gender', 'age']:
            plot_path = self.plot_demographic_distribution(
                analysis, attr, f'{attr.title()} Distribution: Model vs Real World'
            )
            if plot_path:
                plots.append(plot_path)
        
        # Statistical parity differences
        spd_plot = self.plot_statistical_parity_differences(analysis)
        if spd_plot:
            plots.append(spd_plot)
        
        # Bias amplification comparison
        bias_plot = self.plot_bias_amplification_comparison(analysis)
        if bias_plot:
            plots.append(bias_plot)
        
        # Intersectional analysis
        intersectional_plot = self.plot_intersectional_analysis(analysis)
        if intersectional_plot:
            plots.append(intersectional_plot)
        
        # Interactive dashboard
        dashboard = self.create_interactive_dashboard(analysis)
        if dashboard:
            plots.append(dashboard)
        
        return plots

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Create visualizations for bias analysis")
    parser.add_argument("--results", required=True,
                       help="Path to bias analysis results JSON file")
    parser.add_argument("--output", required=True,
                       help="Output directory for visualization files")
    
    args = parser.parse_args()
    
    # Load analysis results
    with open(args.results, 'r') as f:
        analysis = json.load(f)
    
    # Create visualizer
    visualizer = BiasVisualizer(args.output)
    
    # Create all visualizations
    logger.info("Creating visualizations...")
    plots = visualizer.create_all_visualizations(analysis)
    
    logger.info(f"Created {len(plots)} visualizations in {args.output}")
    for plot in plots:
        logger.info(f"  - {os.path.basename(plot)}")

if __name__ == "__main__":
    main()
