"""
Visualization Module for Interpretability Analysis.

This module provides comprehensive visualization capabilities for SHAP analysis
and interpretability metrics, including publication-quality plots for both
baseline and transformer models.
"""

import logging
import numpy as np
import matplotlib
# Use non-interactive backend to prevent opening windows
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Ensure matplotlib is in non-interactive mode
plt.ioff()  # Turn off interactive mode

logger = logging.getLogger(__name__)


class InterpretabilityVisualizer:
    """
    Comprehensive visualization class for interpretability analysis.
    
    This class provides methods to create publication-quality visualizations
    for SHAP analysis, interpretability metrics, and cross-dataset comparisons.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the interpretability visualizer.
        
        Args:
            config: Configuration dictionary for visualization settings
        """
        self.config = config or {}
        
        # Default configuration
        self.visualization_config = self.config.get('visualization', {})
        self.max_features_display = self.visualization_config.get('max_features_display', 20)
        self.figure_dpi = self.visualization_config.get('figure_dpi', 300)
        self.figure_format = self.visualization_config.get('figure_format', 'png')
        self.color_scheme = self.visualization_config.get('color_scheme', 'viridis')
        self.save_plots = self.visualization_config.get('save_plots', True)
        self.max_tokens_per_text = self.visualization_config.get('max_tokens_per_text', 30)
        self.highlight_sample_idx = self.visualization_config.get('highlight_sample_idx', 0)
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
        
        # Configure seaborn for non-interactive use
        sns.set_style("whitegrid")
        sns.set_context("paper")
        
        logger.info("InterpretabilityVisualizer initialized")
    
    def _setup_matplotlib_style(self):
        """Set up matplotlib style for publication-quality plots."""
        # Set style parameters
        plt.style.use('default')
        
        # Configure matplotlib for publication quality
        plt.rcParams.update({
            'figure.dpi': self.figure_dpi,
            'savefig.dpi': self.figure_dpi,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'lines.linewidth': 1.5,
            'axes.linewidth': 0.8,
            'grid.alpha': 0.3,
            'text.usetex': False  # Set to True if LaTeX is available
        })
        
        # Set color scheme
        if self.color_scheme == 'viridis':
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.viridis(np.linspace(0, 1, 10)))
        elif self.color_scheme == 'plasma':
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.plasma(np.linspace(0, 1, 10)))
        elif self.color_scheme == 'inferno':
            plt.rcParams['axes.prop_cycle'] = plt.cycler('color', plt.cm.inferno(np.linspace(0, 1, 10)))
        else:
            # Use default color cycle
            pass
    
    def create_summary_plot(self, 
                           shap_values: np.ndarray,
                           feature_names: List[str],
                           max_features: Optional[int] = None,
                           title: str = "SHAP Summary Plot",
                           figsize: Tuple[int, int] = (10, 8),
                           output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a SHAP summary plot showing feature importance.
        
        Args:
            shap_values: SHAP values array (samples, features, classes)
            feature_names: List of feature names
            max_features: Maximum number of features to display
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating SHAP summary plot")
        
        if max_features is None:
            max_features = self.max_features_display
        
        # Flatten SHAP values if they have multiple classes
        if len(shap_values.shape) > 2:
            # For multi-class, use the positive class (index 1) or average
            if shap_values.shape[2] == 2:
                shap_values_flat = shap_values[:, :, 1]  # Positive class
            else:
                shap_values_flat = np.mean(shap_values, axis=2)  # Average across classes
        else:
            shap_values_flat = shap_values
        
        # Calculate feature importance (mean absolute SHAP values)
        feature_importance = np.mean(np.abs(shap_values_flat), axis=0)
        
        # Get top features
        top_indices = np.argsort(feature_importance)[-max_features:][::-1]
        top_importance = feature_importance[top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(top_names))
        bars = ax.barh(y_pos, top_importance, color=plt.cm.viridis(np.linspace(0, 1, len(top_names))))
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_importance)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', ha='left', fontsize=8)
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, "summary_plot")
        
        logger.info(f"Summary plot created with {len(top_names)} features")
        return fig
    
    def create_waterfall_plot(self,
                             shap_values: np.ndarray,
                             feature_names: List[str],
                             sample_idx: int = 0,
                             title: str = "SHAP Waterfall Plot",
                             figsize: Tuple[int, int] = (12, 8),
                             output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a waterfall plot for individual predictions.

        Args:
            shap_values: SHAP values array (samples, features, classes)
            feature_names: List of feature names
            sample_idx: Index of sample to visualize
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating SHAP waterfall plot for sample {sample_idx}")

        # Get SHAP values for the specific sample
        if len(shap_values.shape) > 2:
            if shap_values.shape[2] == 2:
                sample_shap = shap_values[sample_idx, :, 1]  # Positive class
            else:
                sample_shap = np.mean(shap_values[sample_idx, :, :], axis=1)  # Average across classes
        else:
            sample_shap = shap_values[sample_idx, :]

        # Sort features by absolute SHAP value and limit to top features
        abs_shap = np.abs(sample_shap)
        sorted_indices = np.argsort(abs_shap)[::-1]

        # Limit to max_tokens_per_text for readability
        max_features = min(self.max_tokens_per_text, len(sorted_indices))
        sorted_indices = sorted_indices[:max_features]

        sorted_shap = sample_shap[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Calculate cumulative values for waterfall
        cumulative = np.cumsum(sorted_shap)
        base_value = 0  # Starting point
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create waterfall bars
        x_pos = np.arange(len(sorted_names))
        colors = ['red' if val < 0 else 'blue' for val in sorted_shap]
        
        # Plot bars
        bars = ax.bar(x_pos, sorted_shap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add cumulative line
        ax.plot(x_pos, cumulative + base_value, 'k-', linewidth=2, alpha=0.8, label='Cumulative')
        
        # Add base value line
        ax.axhline(y=base_value, color='gray', linestyle='--', alpha=0.7, label='Base Value')
        
        # Add final prediction line
        final_prediction = base_value + cumulative[-1]
        ax.axhline(y=final_prediction, color='green', linestyle='-', linewidth=2, alpha=0.8, label='Final Prediction')
        
        # Customize the plot
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_ylabel('SHAP Value')
        ax.set_title(f"{title} - Sample {sample_idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_shap)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + (0.01 if height > 0 else -0.01),
                   f'{value:.3f}', ha='center', va='bottom' if height > 0 else 'top', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, f"waterfall_plot_sample_{sample_idx}")
        
        logger.info(f"Waterfall plot created for sample {sample_idx}")
        return fig
    
    def create_force_plot(self,
                         shap_values: np.ndarray,
                         feature_names: List[str],
                         sample_idx: int = 0,
                         title: str = "SHAP Force Plot",
                         figsize: Tuple[int, int] = (12, 8),
                         output_path: Optional[Union[str, Path]] = None,
                         top_k_each: int = 10) -> plt.Figure:
        """
        Create a force plot showing feature contributions.
        Shows top K positive and top K negative impact features.

        Args:
            shap_values: SHAP values array (samples, features, classes)
            feature_names: List of feature names
            sample_idx: Index of sample to visualize
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot
            top_k_each: Number of top positive and negative features to show (default 10)

        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating SHAP force plot for sample {sample_idx}")

        # Get SHAP values for the specific sample
        if len(shap_values.shape) > 2:
            if shap_values.shape[2] == 2:
                sample_shap = shap_values[sample_idx, :, 1]  # Positive class
            else:
                sample_shap = np.mean(shap_values[sample_idx, :, :], axis=1)  # Average across classes
        else:
            sample_shap = shap_values[sample_idx, :]

        # Separate positive and negative SHAP values
        positive_mask = sample_shap > 0
        negative_mask = sample_shap < 0

        positive_shap = sample_shap[positive_mask]
        positive_features = np.where(positive_mask)[0]

        negative_shap = sample_shap[negative_mask]
        negative_features = np.where(negative_mask)[0]

        # Get top K positive impacts (highest positive values)
        if len(positive_shap) > 0:
            top_positive_indices = np.argsort(positive_shap)[-top_k_each:][::-1]
            top_positive_shap = positive_shap[top_positive_indices]
            top_positive_names = [feature_names[positive_features[i]] for i in top_positive_indices]
        else:
            top_positive_shap = np.array([])
            top_positive_names = []

        # Get top K negative impacts (lowest negative values)
        if len(negative_shap) > 0:
            top_negative_indices = np.argsort(negative_shap)[:top_k_each]
            top_negative_shap = negative_shap[top_negative_indices]
            top_negative_names = [feature_names[negative_features[i]] for i in top_negative_indices]
        else:
            top_negative_shap = np.array([])
            top_negative_names = []

        # Combine top positive and negative features
        sorted_shap = np.concatenate([top_positive_shap, top_negative_shap])
        sorted_names = top_positive_names + top_negative_names
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)

        if len(sorted_shap) == 0:
            ax.text(0.5, 0.5, 'No SHAP values available for this sample',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(f"{title} - Sample {sample_idx}")
            return fig

        # Create horizontal bars showing force
        y_pos = np.arange(len(sorted_names))
        colors = ['#d62728' if val < 0 else '#1f77b4' for val in sorted_shap]  # Red for negative, blue for positive

        # Plot bars
        bars = ax.barh(y_pos, sorted_shap, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2, alpha=0.8)

        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names, fontsize=9)
        ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=10)
        ax.set_title(f"{title} - Sample {sample_idx}\n(Top {top_k_each} Positive & Top {top_k_each} Negative)",
                    fontsize=12, weight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_shap)):
            x_offset = 0.01 * (ax.get_xlim()[1] - ax.get_xlim()[0])
            ax.text(bar.get_width() + (x_offset if value > 0 else -x_offset),
                   bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', ha='left' if value > 0 else 'right',
                   fontsize=8, weight='bold')

        # Invert y-axis to show most important features at top
        ax.invert_yaxis()

        # Add section separators and labels if we have both positive and negative
        if len(top_positive_shap) > 0 and len(top_negative_shap) > 0:
            # Add a horizontal line separating positive from negative
            separator_y = len(top_positive_shap) - 0.5
            ax.axhline(y=separator_y, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Add section labels
            ax.text(0.02, 0.98, 'Positive Impact →', transform=ax.transAxes,
                   fontsize=10, weight='bold', color='#1f77b4', va='top')
            ax.text(0.02, 0.48, 'Negative Impact →', transform=ax.transAxes,
                   fontsize=10, weight='bold', color='#d62728', va='top')

        # Add legend
        red_patch = mpatches.Patch(color='#d62728', alpha=0.8, label='Negative Impact (pushes toward class 0)')
        blue_patch = mpatches.Patch(color='#1f77b4', alpha=0.8, label='Positive Impact (pushes toward class 1)')
        ax.legend(handles=[blue_patch, red_patch], loc='lower right', fontsize=9)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, f"force_plot_sample_{sample_idx}")
        
        logger.info(f"Force plot created for sample {sample_idx}")
        return fig
    
    def create_cross_dataset_comparison(self, 
                                       results_dict: Dict[str, Dict[str, Any]],
                                       metric: str = "sparsity_score",
                                       title: str = "Cross-Dataset Interpretability Comparison",
                                       figsize: Tuple[int, int] = (12, 8),
                                       output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create cross-dataset comparison visualization.
        
        Args:
            results_dict: Dictionary with dataset names as keys and results as values
            metric: Metric to compare across datasets
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info(f"Creating cross-dataset comparison for metric: {metric}")
        
        # Extract data for comparison
        datasets = list(results_dict.keys())
        metric_values = []
        model_names = []
        
        for dataset, results in results_dict.items():
            if 'metrics' in results and metric in results['metrics']:
                metric_value = results['metrics'][metric]
                if isinstance(metric_value, dict) and metric in metric_value:
                    metric_values.append(metric_value[metric])
                else:
                    metric_values.append(metric_value)
                model_names.append(dataset)
        
        if not metric_values:
            logger.warning(f"No data found for metric: {metric}")
            # Create empty plot
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, f'No data available for metric: {metric}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Bar plot
        bars = ax1.bar(model_names, metric_values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
        ax1.set_ylabel(metric.replace('_', ' ').title())
        ax1.set_title(f"{metric.replace('_', ' ').title()} by Dataset")
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Box plot (if we have multiple values per dataset)
        # For now, create a simple comparison
        ax2.bar(model_names, metric_values, color=plt.cm.plasma(np.linspace(0, 1, len(model_names))))
        ax2.set_ylabel(metric.replace('_', ' ').title())
        ax2.set_title(f"{metric.replace('_', ' ').title()} Comparison")
        ax2.tick_params(axis='x', rotation=45)
        
        # Add grid
        ax1.grid(True, alpha=0.3, axis='y')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Overall title
        fig.suptitle(title, fontsize=14, y=0.98)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, f"cross_dataset_comparison_{metric}")
        
        logger.info(f"Cross-dataset comparison created for {len(datasets)} datasets")
        return fig
    
    def create_metrics_comparison(self,
                                 metrics_data: Dict[str, Dict[str, float]],
                                 title: str = "Interpretability Metrics Comparison",
                                 figsize: Tuple[int, int] = (14, 10),
                                 output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a comprehensive comparison of interpretability metrics.

        Args:
            metrics_data: Dictionary with model names as keys and metrics as values
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot

        Returns:
            matplotlib Figure object
        """
        logger.info("Creating interpretability metrics comparison")

        # Extract metrics - use the actual metric names from the flattened data
        models = list(metrics_data.keys())

        # Determine available metrics across all models
        all_metrics = set()
        for model_data in metrics_data.values():
            all_metrics.update(model_data.keys())

        # Prioritize key metrics if they exist
        priority_metrics = ['faithfulness_mean_drop', 'saas_alignment', 'mean_entropy', 'explanation_similarity']
        metrics = [m for m in priority_metrics if m in all_metrics]

        # Add any other metrics not in priority list
        for m in sorted(all_metrics):
            if m not in metrics and not m.startswith('error'):
                metrics.append(m)

        # Limit to 3 main metrics for the bar plots
        metrics = metrics[:3] if len(metrics) > 3 else metrics

        if not metrics:
            logger.warning("No metrics available for comparison plot")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No metrics available for comparison',
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            return fig

        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        # Plot each metric
        for i, metric in enumerate(metrics):
            if i >= len(axes) - 1:  # Reserve last subplot for heatmap
                break

            ax = axes[i]
            values = []
            model_names = []

            for model, data in metrics_data.items():
                if metric in data and data[metric] is not None:
                    values.append(float(data[metric]))
                    model_names.append(model)

            if values:
                bars = ax.bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')

                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(bar.get_height() * 0.01, 0.001),
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
            else:
                ax.text(0.5, 0.5, f'No data for {metric}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{metric.replace('_', ' ').title()}")

        # Overall comparison heatmap in the last subplot
        ax = axes[-1]
        metric_matrix = []
        for model in models:
            row = []
            for metric in metrics:
                value = metrics_data.get(model, {}).get(metric)
                if value is not None:
                    row.append(float(value))
                else:
                    row.append(0)
            metric_matrix.append(row)

        if metric_matrix and any(any(row) for row in metric_matrix):
            im = ax.imshow(metric_matrix, cmap='viridis', aspect='auto')
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45, ha='right')
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels(models)
            ax.set_title("Metrics Heatmap")

            # Add colorbar
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, 'No data for heatmap',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        # Overall title
        fig.suptitle(title, fontsize=14, y=0.98)

        # Adjust layout
        plt.tight_layout()

        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, "metrics_comparison")

        logger.info("Metrics comparison plot created")
        return fig
    
    def create_feature_importance_heatmap(self, 
                                         shap_values: np.ndarray,
                                         feature_names: List[str],
                                         max_features: Optional[int] = None,
                                         title: str = "Feature Importance Heatmap",
                                         figsize: Tuple[int, int] = (12, 8),
                                         output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a heatmap showing feature importance across samples.
        
        Args:
            shap_values: SHAP values array (samples, features, classes)
            feature_names: List of feature names
            max_features: Maximum number of features to display
            title: Plot title
            figsize: Figure size (width, height)
            output_path: Path to save the plot
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating feature importance heatmap")
        
        if max_features is None:
            max_features = self.max_features_display
        
        # Flatten SHAP values if they have multiple classes
        if len(shap_values.shape) > 2:
            if shap_values.shape[2] == 2:
                shap_values_flat = shap_values[:, :, 1]  # Positive class
            else:
                shap_values_flat = np.mean(shap_values, axis=2)  # Average across classes
        else:
            shap_values_flat = shap_values
        
        # Get top features by mean absolute importance
        feature_importance = np.mean(np.abs(shap_values_flat), axis=0)
        top_indices = np.argsort(feature_importance)[-max_features:][::-1]
        
        # Select top features
        top_shap = shap_values_flat[:, top_indices]
        top_names = [feature_names[i] for i in top_indices]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap
        im = ax.imshow(top_shap.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
        
        # Customize the plot
        ax.set_xticks(range(0, top_shap.shape[0], max(1, top_shap.shape[0]//10)))
        ax.set_xticklabels([f'Sample {i}' for i in range(0, top_shap.shape[0], max(1, top_shap.shape[0]//10))])
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Samples')
        ax.set_ylabel('Features')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('SHAP Value')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot if requested
        if self.save_plots and output_path:
            self._save_plot(fig, output_path, "feature_importance_heatmap")
        
        logger.info(f"Feature importance heatmap created with {len(top_names)} features")
        return fig

    def create_explanation_quality_dashboard(
            self,
            metrics_results: Dict[str, Any],
            output_path: Union[str, Path],
            model_name: str,
            dataset_name: str
        ) -> Optional[plt.Figure]:
        """
        Create a dashboard summarizing explanation quality metrics.
        """
        if not metrics_results:
            logger.warning("No metrics provided for dashboard")
            return None

        saas = metrics_results.get('saas', {})
        entropy = metrics_results.get('attribution_entropy', {})
        problematic = metrics_results.get('problematic_attributions', {})

        alignment_score = saas.get('alignment_score')
        alignment_samples = saas.get('samples_evaluated', 0)
        entropies = entropy.get('entropies', [])
        mean_entropy = entropy.get('mean_entropy')
        problematic_list = problematic.get('items', [])

        if alignment_score is None and mean_entropy is None and not problematic_list:
            logger.warning("Insufficient metrics for dashboard")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. SAAS bar
        ax = axes[0, 0]
        if alignment_score is not None:
            color = 'green' if alignment_score >= 0.3 else 'orange' if alignment_score >= 0 else 'red'
            ax.bar(['SAAS'], [alignment_score], color=color, width=0.5)
            ax.axhline(0, linestyle='--', color='gray', alpha=0.7)
            ax.set_ylim(-1, 1)
            ax.set_title('Sentiment-Attribution Alignment', fontsize=10, weight='bold')
            ax.text(0, alignment_score + 0.05 * (1 if alignment_score >= 0 else -1),
                    f'{alignment_score:.3f}', ha='center', va='bottom' if alignment_score >= 0 else 'top', fontsize=9)
            ax.set_ylabel('Alignment Score', fontsize=9)
            ax.text(0, -0.9, f'Samples: {alignment_samples}', ha='center', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Alignment score\nunavailable\n(requires vaderSentiment)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title('Sentiment-Attribution Alignment', fontsize=10, weight='bold')
            ax.grid(True, alpha=0.3)

        # 2. Entropy distribution
        ax = axes[0, 1]
        if entropies:
            # Limit to first 30 samples for readability
            display_entropies = entropies[:30]
            colors = ['green' if e <= 3 else 'orange' if e <= 4 else 'red' for e in display_entropies]
            ax.bar(range(len(display_entropies)), display_entropies, color=colors, width=0.8)
            ax.set_title('Attribution Entropy by Sample', fontsize=10, weight='bold')
            ax.set_xlabel('Sample Index', fontsize=9)
            ax.set_ylabel('Entropy', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            if mean_entropy is not None:
                ax.axhline(mean_entropy, linestyle='--', color='black', alpha=0.7, linewidth=2)
                ax.text(len(display_entropies) * 0.95, mean_entropy + 0.1,
                        f'Mean: {mean_entropy:.2f}', ha='right', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            if len(entropies) > 30:
                ax.text(0.5, -0.15, f'Showing first 30 of {len(entropies)} samples',
                       ha='center', va='top', transform=ax.transAxes, fontsize=8, style='italic')
        else:
            ax.text(0.5, 0.5, 'Entropy data\nunavailable', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title('Attribution Entropy by Sample', fontsize=10, weight='bold')
            ax.grid(True, alpha=0.3)

        # 3. SHAP score histogram
        ax = axes[1, 0]
        shap_distribution = metrics_results.get('shap_distribution', {})
        shap_values = shap_distribution.get('values')
        if shap_values is not None and len(shap_values) > 0:
            ax.hist(shap_values, bins=30, color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.axvline(0, linestyle='--', color='red', alpha=0.7, linewidth=2)
            ax.set_title('Distribution of SHAP Values', fontsize=10, weight='bold')
            ax.set_xlabel('SHAP value', fontsize=9)
            ax.set_ylabel('Frequency', fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
            # Add mean and median lines
            mean_val = np.mean(shap_values)
            median_val = np.median(shap_values)
            ax.axvline(mean_val, linestyle=':', color='green', alpha=0.7, linewidth=2, label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, linestyle='-.', color='orange', alpha=0.7, linewidth=2, label=f'Median: {median_val:.3f}')
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'SHAP distribution\nunavailable', transform=ax.transAxes,
                    ha='center', va='center', fontsize=10)
            ax.set_title('Distribution of SHAP Values', fontsize=10, weight='bold')
            ax.grid(True, alpha=0.3)

        # 4. Problematic attribution pie or bar chart
        ax = axes[1, 1]
        if problematic_list:
            issue_counts = {}
            for item in problematic_list:
                issue = item.get('issue', 'unknown')
                issue_counts[issue] = issue_counts.get(issue, 0) + 1

            if len(issue_counts) > 0:
                labels = [label.replace('_', ' ').title() for label in issue_counts.keys()]
                sizes = list(issue_counts.values())

                # Use pie chart if 2-4 categories, bar chart otherwise
                if len(labels) <= 4:
                    colors = plt.cm.Paired(np.linspace(0, 1, len(labels)))
                    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                                       startangle=90, colors=colors)
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontsize(8)
                        autotext.set_weight('bold')
                    ax.set_title('Problematic Attributions', fontsize=10, weight='bold')
                else:
                    ax.barh(labels, sizes, color=plt.cm.Set3(np.linspace(0, 1, len(labels))))
                    ax.set_xlabel('Count', fontsize=9)
                    ax.set_title('Problematic Attributions', fontsize=10, weight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
            else:
                ax.text(0.5, 0.5, 'No problematic\nattributions found',
                        transform=ax.transAxes, ha='center', va='center', fontsize=10)
                ax.set_title('Problematic Attributions', fontsize=10, weight='bold')
        else:
            ax.text(0.5, 0.5, 'No problematic\nattributions found\n(requires vaderSentiment)',
                    transform=ax.transAxes, ha='center', va='center', fontsize=10)
            ax.set_title('Problematic Attributions', fontsize=10, weight='bold')
            ax.grid(True, alpha=0.3)

        fig.suptitle(f'Explanation Quality Dashboard - {model_name} on {dataset_name}', fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        if self.save_plots:
            self._save_plot(fig, output_path, "explanation_quality_dashboard")

        return fig

    def create_token_highlight_html(
            self,
            shap_results: Dict[str, Any],
            output_path: Union[str, Path],
            model_name: str,
            dataset_name: str
        ) -> Optional[Path]:
        """
        Create an HTML file highlighting tokens with SHAP contributions.
        """
        samples = shap_results.get('samples')
        texts = shap_results.get('texts')

        if not samples or texts is None or len(samples) == 0:
            logger.warning("No sample token data available for highlight visualization")
            return None

        sample_idx = min(self.highlight_sample_idx, len(samples) - 1)
        sample_data = samples[sample_idx]
        tokens = sample_data.get('tokens', [])
        shap_vals = sample_data.get('shap_values', [])

        if not tokens or not shap_vals:
            logger.warning("Selected sample has no token information")
            return None

        abs_vals = np.abs(shap_vals)
        if abs_vals.size == 0:
            logger.warning("No SHAP values available for highlight")
            return None

        top_k = min(self.max_tokens_per_text, len(tokens))
        top_indices = np.argsort(abs_vals)[-top_k:]
        max_abs = abs_vals[top_indices].max() if top_indices.size > 0 else abs_vals.max()
        top_set = set(top_indices.tolist())

        def token_to_span(token: str, value: float, include: bool) -> str:
            clean_token = token
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                return ''
            if token.startswith('##'):
                clean_token = token[2:]
                prefix = ''
            else:
                prefix = ' '
            if not include or max_abs == 0:
                return prefix + clean_token
            intensity = min(abs(value) / max_abs, 1.0)
            hue = 0 if value >= 0 else 220  # red for positive, blue for negative
            alpha = 0.15 + 0.75 * intensity
            style = (
                f"background-color: hsla({hue}, 85%, 60%, {alpha});"
                "padding: 2px 4px; border-radius: 3px;"
            )
            return f"{prefix}<span style=\"{style}\">{clean_token}</span>"

        highlighted_tokens: List[str] = []
        for idx, (token, value) in enumerate(zip(tokens, shap_vals)):
            include = idx in top_set
            span = token_to_span(token, value, include)
            if span:
                highlighted_tokens.append(span)

        highlighted_text = ''.join(highlighted_tokens).strip()
        original_text = texts[sample_idx]

        html_content = f"""
        <html>
        <head>
            <meta charset="utf-8">
            <title>SHAP Token Highlight - {model_name} on {dataset_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 900px; margin: auto; padding: 20px; }}
                .original {{ margin-bottom: 20px; }}
                .highlighted {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .legend {{ margin-top: 15px; font-size: 0.9em; color: #555; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h2>SHAP Token Highlight</h2>
                <h3>{model_name} on {dataset_name} (Sample {sample_idx})</h3>
                <div class="original">
                    <strong>Original Text:</strong>
                    <p>{original_text}</p>
                </div>
                <div class="highlighted">
                    <strong>Highlighted Explanation:</strong>
                    <p>{highlighted_text}</p>
                </div>
                <div class="legend">
                    Positive contributions are shown in red, negative in blue. Only the top {top_k} tokens by |SHAP| are highlighted.
                </div>
            </div>
        </body>
        </html>
        """

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        filename = f"token_highlight_sample_{sample_idx}.html"
        filepath = output_path / filename
        filepath.write_text(html_content, encoding='utf-8')
        logger.info(f"Token highlight HTML saved to {filepath}")
        return filepath
    
    def _save_plot(self, fig: plt.Figure, output_path: Union[str, Path], plot_name: str):
        """
        Save a plot to the specified path.
        
        Args:
            fig: matplotlib Figure object
            output_path: Base output path
            plot_name: Name of the plot file
        """
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            filename = f"{plot_name}.{self.figure_format}"
            filepath = output_path / filename
            
            fig.savefig(filepath, format=self.figure_format, dpi=self.figure_dpi, bbox_inches='tight')
            logger.info(f"Plot saved to: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving plot {plot_name}: {e}")
    
    def create_all_visualizations(self, 
                                 shap_results: Dict[str, Any],
                                 model_name: str,
                                 dataset_name: str,
                                 output_dir: Union[str, Path],
                                 metrics_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create all standard visualizations for a model and dataset.
        
        Args:
            shap_results: Results from SHAP analysis
            model_name: Name of the model
            dataset_name: Name of the dataset
            output_dir: Directory to save plots
            metrics_results: Optional interpretability metrics results
            
        Returns:
            Dictionary mapping plot names to Figure objects
        """
        logger.info(f"Creating all visualizations for {model_name} on {dataset_name}")
        
        output_path = Path(output_dir) / "interpretability" / "visualizations" / f"{model_name}_{dataset_name}"
        output_path.mkdir(parents=True, exist_ok=True)
        
        figures: Dict[str, Any] = {}
        
        # Extract data
        shap_values = shap_results['shap_values']
        feature_names = shap_results['feature_names']
        samples = shap_results.get('samples')
        texts = shap_results.get('texts')
        
        # Create summary plot
        try:
            fig_summary = self.create_summary_plot(
                shap_values, feature_names, 
                title=f"SHAP Summary - {model_name} on {dataset_name}",
                output_path=output_path
            )
            figures['summary'] = fig_summary
        except Exception as e:
            logger.error(f"Error creating summary plot: {e}")
        
        # Create waterfall plot for first sample
        try:
            fig_waterfall = self.create_waterfall_plot(
                shap_values, feature_names, sample_idx=0,
                title=f"SHAP Waterfall - {model_name} on {dataset_name}",
                output_path=output_path
            )
            figures['waterfall'] = fig_waterfall
        except Exception as e:
            logger.error(f"Error creating waterfall plot: {e}")
        
        # Create force plot for first sample
        try:
            fig_force = self.create_force_plot(
                shap_values, feature_names, sample_idx=0,
                title=f"SHAP Force - {model_name} on {dataset_name}",
                output_path=output_path
            )
            figures['force'] = fig_force
        except Exception as e:
            logger.error(f"Error creating force plot: {e}")
        
        # Create feature importance heatmap
        try:
            fig_heatmap = self.create_feature_importance_heatmap(
                shap_values, feature_names,
                title=f"Feature Importance Heatmap - {model_name} on {dataset_name}",
                output_path=output_path
            )
            figures['heatmap'] = fig_heatmap
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
        
        # Metrics comparison plot removed - redundant with explanation quality dashboard

        # Create explanation quality dashboard
        if metrics_results:
            try:
                dashboard_fig = self.create_explanation_quality_dashboard(
                    metrics_results, output_path, model_name, dataset_name
                )
                if dashboard_fig is not None:
                    figures['explanation_dashboard'] = dashboard_fig
            except Exception as e:
                logger.error(f"Error creating explanation quality dashboard: {e}")

        # Create token highlight HTML
        try:
            highlight_path = self.create_token_highlight_html(
                shap_results, output_path, model_name, dataset_name
            )
            if highlight_path:
                figures['token_highlight_html'] = highlight_path
        except Exception as e:
            logger.error(f"Error creating token highlight HTML: {e}")
        
        logger.info(f"Created {len(figures)} visualizations for {model_name} on {dataset_name}")
        return figures

    @staticmethod
    def _flatten_metrics(metrics_results: Dict[str, Any]) -> Dict[str, float]:
        flat: Dict[str, float] = {}
        if not metrics_results:
            return flat

        for key, value in metrics_results.items():
            if isinstance(value, dict):
                if 'value' in value and isinstance(value['value'], (int, float)):
                    flat[key] = float(value['value'])
                elif key == 'faithfulness':
                    score = value.get('mean_drop_total')
                    if score is not None:
                        flat['faithfulness_mean_drop'] = float(score)
                elif key == 'saas':
                    alignment = value.get('alignment_score')
                    if alignment is not None:
                        flat['saas_alignment'] = float(alignment)
                elif key == 'attribution_entropy':
                    mean_entropy = value.get('mean_entropy')
                    if mean_entropy is not None:
                        flat['mean_entropy'] = float(mean_entropy)
                elif key == 'explanation_consistency':
                    mean_sim = value.get('mean_similarity')
                    if mean_sim is not None:
                        flat['explanation_similarity'] = float(mean_sim)
            elif isinstance(value, (int, float)):
                flat[key] = float(value)
        return flat


def create_interpretability_visualizer(config: Dict[str, Any]) -> InterpretabilityVisualizer:
    """
    Factory function to create an InterpretabilityVisualizer instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        InterpretabilityVisualizer instance
    """
    return InterpretabilityVisualizer(config)


if __name__ == "__main__":
    # Test the visualization module
    print("Interpretability Visualization module loaded successfully")
    print("Use create_interpretability_visualizer() to create a visualizer instance")
