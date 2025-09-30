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
        
        # Sort features by SHAP value
        sorted_indices = np.argsort(sample_shap)[::-1]
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
                         figsize: Tuple[int, int] = (12, 6),
                         output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create a force plot showing feature contributions.
        
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
        logger.info(f"Creating SHAP force plot for sample {sample_idx}")
        
        # Get SHAP values for the specific sample
        if len(shap_values.shape) > 2:
            if shap_values.shape[2] == 2:
                sample_shap = shap_values[sample_idx, :, 1]  # Positive class
            else:
                sample_shap = np.mean(shap_values[sample_idx, :, :], axis=1)  # Average across classes
        else:
            sample_shap = shap_values[sample_idx, :]
        
        # Sort features by absolute SHAP value
        sorted_indices = np.argsort(np.abs(sample_shap))[::-1]
        sorted_shap = sample_shap[sorted_indices]
        sorted_names = [feature_names[i] for i in sorted_indices]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bars showing force
        y_pos = np.arange(len(sorted_names))
        colors = ['red' if val < 0 else 'blue' for val in sorted_shap]
        
        # Plot bars
        bars = ax.barh(y_pos, sorted_shap, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add zero line
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.8)
        
        # Customize the plot
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('SHAP Value (Force)')
        ax.set_title(f"{title} - Sample {sample_idx}")
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_shap)):
            ax.text(bar.get_width() + (0.001 if value > 0 else -0.001), bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', ha='left' if value > 0 else 'right', fontsize=8)
        
        # Invert y-axis to show most important features at top
        ax.invert_yaxis()
        
        # Add legend
        red_patch = mpatches.Patch(color='red', alpha=0.7, label='Negative Impact')
        blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Positive Impact')
        ax.legend(handles=[red_patch, blue_patch])
        
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
        
        # Extract metrics
        models = list(metrics_data.keys())
        metrics = ['faithfulness_score', 'sparsity_score', 'intuitiveness_score']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            ax = axes[i]
            values = []
            model_names = []
            
            for model, data in metrics_data.items():
                if metric in data:
                    values.append(data[metric])
                    model_names.append(model)
            
            if values:
                bars = ax.bar(model_names, values, color=plt.cm.viridis(np.linspace(0, 1, len(model_names))))
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f"{metric.replace('_', ' ').title()}")
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Overall comparison heatmap
        if len(axes) > len(metrics):
            ax = axes[-1]
            # Create a simple heatmap
            metric_matrix = []
            for model in models:
                row = []
                for metric in metrics:
                    if metric in metrics_data.get(model, {}):
                        row.append(metrics_data[model][metric])
                    else:
                        row.append(0)
                metric_matrix.append(row)
            
            if metric_matrix:
                im = ax.imshow(metric_matrix, cmap='viridis', aspect='auto')
                ax.set_xticks(range(len(metrics)))
                ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], rotation=45)
                ax.set_yticks(range(len(models)))
                ax.set_yticklabels(models)
                ax.set_title("Metrics Heatmap")
                
                # Add colorbar
                plt.colorbar(im, ax=ax)
        
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
                                 metrics_results: Optional[Dict[str, Any]] = None) -> Dict[str, plt.Figure]:
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
        
        figures = {}
        
        # Extract data
        shap_values = shap_results['shap_values']
        feature_names = shap_results['feature_names']
        
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
        
        # Create metrics comparison if available
        if metrics_results:
            try:
                fig_metrics = self.create_metrics_comparison(
                    {f"{model_name}_{dataset_name}": metrics_results},
                    title=f"Interpretability Metrics - {model_name} on {dataset_name}",
                    output_path=output_path
                )
                figures['metrics'] = fig_metrics
            except Exception as e:
                logger.error(f"Error creating metrics plot: {e}")
        
        logger.info(f"Created {len(figures)} visualizations for {model_name} on {dataset_name}")
        return figures


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
