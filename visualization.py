"""
Visualization module for MELoRA experiments.
Generates plots for training curves, memory usage, and performance comparisons.
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import utils


class Visualizer:
    """Handles visualization of experimental results."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config['visualization']
        self.save_path = self.viz_config['save_path']
        
        # Create save directory
        os.makedirs(self.save_path, exist_ok=True)
        
        # Set style
        plt.style.use(self.viz_config['style'])
        sns.set_palette("husl")
        
        # Set default figure size
        self.figsize = tuple(self.viz_config['figure_size'])
        self.fontsize = self.viz_config['font_size']
        
        # Set font sizes
        plt.rcParams.update({
            'font.size': self.fontsize,
            'axes.titlesize': self.fontsize + 2,
            'axes.labelsize': self.fontsize,
            'xtick.labelsize': self.fontsize - 2,
            'ytick.labelsize': self.fontsize - 2,
            'legend.fontsize': self.fontsize - 2,
        })
        
        self.logger = utils.get_logger()
        
    def plot_learning_curves(self, 
                           metrics_history: Dict[str, List[float]],
                           title: str = "MELoRA Learning Curves",
                           save_name: Optional[str] = None):
        """Plot learning curves for training and validation metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]*1.5))
        axes = axes.flatten()
        
        # Define metrics to plot
        metrics_to_plot = [
            ('train/meta_train_loss', 'Meta-Training Loss', 'loss'),
            ('val/meta_val_loss', 'Meta-Validation Loss', 'loss'),
            ('val/meta_val_accuracy', 'Meta-Validation Accuracy', 'accuracy'),
            ('train/memory/allocated_mb', 'GPU Memory Usage (MB)', 'memory')
        ]
        
        for idx, (metric_key, metric_title, metric_type) in enumerate(metrics_to_plot):
            if idx < len(axes) and metric_key in metrics_history:
                ax = axes[idx]
                values = metrics_history[metric_key]
                iterations = range(len(values))
                
                # Plot with smoothing
                ax.plot(iterations, values, alpha=0.3, label='Raw')
                
                # Add smoothed line
                if len(values) > 10:
                    window = min(len(values) // 10, 50)
                    smoothed = pd.Series(values).rolling(window, center=True).mean()
                    ax.plot(iterations, smoothed, label='Smoothed', linewidth=2)
                    
                ax.set_xlabel('Iteration')
                ax.set_ylabel(metric_title)
                ax.set_title(metric_title)
                ax.grid(True, alpha=0.3)
                
                if metric_type == 'accuracy':
                    ax.set_ylim([0, 1])
                    
                if len(values) > 10:
                    ax.legend()
                    
        # Remove empty subplots
        for idx in range(len(metrics_to_plot), len(axes)):
            fig.delaxes(axes[idx])
            
        plt.suptitle(title, fontsize=self.fontsize + 4)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_memory_usage(self,
                         memory_profile: List[Dict[str, float]],
                         title: str = "Memory Usage Profile",
                         save_name: Optional[str] = None):
        """Plot detailed memory usage over time."""
        if not memory_profile:
            self.logger.warning("No memory profile data to plot")
            return None
            
        # Extract data
        timestamps = [m['timestamp'] for m in memory_profile]
        timestamps = [(t - timestamps[0]) for t in timestamps]  # Relative time
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)
        
        # GPU Memory
        if 'allocated_mb' in memory_profile[0]:
            allocated = [m.get('allocated_mb', 0) for m in memory_profile]
            reserved = [m.get('reserved_mb', 0) for m in memory_profile]
            
            ax1.plot(timestamps, allocated, label='Allocated', linewidth=2)
            ax1.plot(timestamps, reserved, label='Reserved', linewidth=2, linestyle='--')
            ax1.fill_between(timestamps, 0, allocated, alpha=0.3)
            
            ax1.set_ylabel('GPU Memory (MB)')
            ax1.set_title('GPU Memory Usage')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Add memory limit line
            if 'memory_limits' in self.config['memory_optimization']:
                limit = self.config['memory_optimization']['memory_limits']['max_gpu_memory_mb']
                ax1.axhline(y=limit, color='r', linestyle=':', label='Memory Limit')
                
        # CPU Memory
        if 'cpu_rss_mb' in memory_profile[0]:
            cpu_rss = [m.get('cpu_rss_mb', 0) for m in memory_profile]
            
            ax2.plot(timestamps, cpu_rss, label='RSS', linewidth=2, color='green')
            ax2.fill_between(timestamps, 0, cpu_rss, alpha=0.3, color='green')
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('CPU Memory (MB)')
            ax2.set_title('CPU Memory Usage')
            ax2.grid(True, alpha=0.3)
            
        plt.suptitle(title, fontsize=self.fontsize + 2)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_parameter_comparison(self,
                                results_dict: Dict[str, Dict[str, Any]],
                                metric: str = 'accuracy',
                                title: Optional[str] = None,
                                save_name: Optional[str] = None):
        """Plot comparison of different methods or configurations."""
        if not title:
            title = f"Method Comparison - {metric.capitalize()}"
            
        methods = list(results_dict.keys())
        means = []
        stds = []
        
        for method in methods:
            if f'{metric}_mean' in results_dict[method]:
                means.append(results_dict[method][f'{metric}_mean'])
                stds.append(results_dict[method][f'{metric}_std'])
            else:
                means.append(0)
                stds.append(0)
                
        # Create bar plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(methods))
        width = 0.6
        
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      error_kw={'linewidth': 2, 'capthick': 2})
        
        # Color bars based on performance
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
            
        ax.set_xlabel('Method')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                   ha='center', va='bottom', fontsize=self.fontsize-2)
            
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_pareto_frontier(self,
                           results_list: List[Dict[str, Any]],
                           x_metric: str = 'memory',
                           y_metric: str = 'accuracy',
                           labels: Optional[List[str]] = None,
                           title: str = "Memory-Performance Trade-off",
                           save_name: Optional[str] = None):
        """Plot Pareto frontier for multi-objective optimization."""
        # Extract data points
        x_values = []
        y_values = []
        
        for result in results_list:
            if x_metric == 'memory' and 'memory' in result:
                x_val = result['memory'].get('peak_gpu_mb', 0)
            else:
                x_val = result.get(f'{x_metric}_mean', 0)
                
            y_val = result.get(f'{y_metric}_mean', 0)
            
            x_values.append(x_val)
            y_values.append(y_val)
            
        # Find Pareto frontier
        pareto_points = self._find_pareto_frontier(x_values, y_values, minimize_x=True)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all points
        scatter = ax.scatter(x_values, y_values, s=100, alpha=0.6, 
                           c=range(len(x_values)), cmap='viridis')
        
        # Highlight Pareto frontier
        pareto_x = [x_values[i] for i in pareto_points]
        pareto_y = [y_values[i] for i in pareto_points]
        ax.plot(pareto_x, pareto_y, 'r--', linewidth=2, label='Pareto Frontier')
        ax.scatter(pareto_x, pareto_y, s=150, c='red', marker='*', 
                  edgecolors='black', linewidth=2)
        
        # Add labels if provided
        if labels:
            for i, label in enumerate(labels):
                ax.annotate(label, (x_values[i], y_values[i]), 
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=self.fontsize-3)
                          
        ax.set_xlabel(f'{x_metric.capitalize()} (MB)' if x_metric == 'memory' else x_metric.capitalize())
        ax.set_ylabel(y_metric.capitalize())
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_heatmap(self,
                    data: Union[np.ndarray, pd.DataFrame],
                    title: str = "Heatmap",
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    save_name: Optional[str] = None,
                    cmap: str = 'coolwarm',
                    annotate: bool = True):
        """Plot a heatmap of 2D data."""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to DataFrame if numpy array
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
            
        # Create heatmap
        sns.heatmap(data, annot=annotate, fmt='.2f', cmap=cmap,
                   cbar_kws={'label': 'Value'}, ax=ax,
                   annot_kws={'fontsize': self.fontsize-4})
        
        ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_ablation_study(self,
                          ablation_results: Dict[str, Dict[str, float]],
                          baseline_name: str = 'full',
                          metric: str = 'accuracy',
                          title: str = "Ablation Study Results",
                          save_name: Optional[str] = None):
        """Visualize ablation study results."""
        # Get baseline performance
        baseline_value = ablation_results[baseline_name][f'{metric}_mean']
        
        # Calculate relative changes
        components = []
        changes = []
        
        for component, results in ablation_results.items():
            if component != baseline_name:
                components.append(component)
                value = results[f'{metric}_mean']
                change = (value - baseline_value) / baseline_value * 100
                changes.append(change)
                
        # Sort by impact
        sorted_idx = np.argsort(changes)
        components = [components[i] for i in sorted_idx]
        changes = [changes[i] for i in sorted_idx]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1]*0.8))
        
        colors = ['red' if c < 0 else 'green' for c in changes]
        bars = ax.barh(components, changes, color=colors, alpha=0.7)
        
        # Add value labels
        for bar, change in zip(bars, changes):
            x = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(x + (1 if x > 0 else -1), y, f'{change:.1f}%',
                   ha='left' if x > 0 else 'right', va='center',
                   fontsize=self.fontsize-2)
                   
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.set_xlabel(f'Relative Change in {metric.capitalize()} (%)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def plot_interactive_3d(self,
                          results_df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          z_col: str,
                          color_col: Optional[str] = None,
                          title: str = "3D Performance Visualization",
                          save_name: Optional[str] = None):
        """Create interactive 3D scatter plot using Plotly."""
        fig = px.scatter_3d(
            results_df,
            x=x_col,
            y=y_col,
            z=z_col,
            color=color_col,
            title=title,
            labels={x_col: x_col.replace('_', ' ').title(),
                   y_col: y_col.replace('_', ' ').title(),
                   z_col: z_col.replace('_', ' ').title()}
        )
        
        fig.update_traces(marker=dict(size=8))
        fig.update_layout(
            scene=dict(
                xaxis_title=x_col.replace('_', ' ').title(),
                yaxis_title=y_col.replace('_', ' ').title(),
                zaxis_title=z_col.replace('_', ' ').title(),
            ),
            width=800,
            height=600
        )
        
        if save_name:
            fig.write_html(os.path.join(self.save_path, f"{save_name}.html"))
            
        return fig
    
    def plot_task_distribution(self,
                             task_results: List[Dict[str, Any]],
                             metric: str = 'accuracy',
                             title: str = "Task Performance Distribution",
                             save_name: Optional[str] = None):
        """Plot distribution of performance across tasks."""
        values = [t[metric] for t in task_results if metric in t]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0]*1.5, self.figsize[1]))
        
        # Histogram
        n, bins, patches = ax1.hist(values, bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(values):.3f}')
        ax1.axvline(np.median(values), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {np.median(values):.3f}')
        ax1.set_xlabel(metric.capitalize())
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot with violin plot overlay
        parts = ax2.violinplot([values], positions=[0], showmeans=True, 
                              showextrema=True, showmedians=True)
        ax2.boxplot([values], positions=[0], widths=0.1)
        ax2.set_ylabel(metric.capitalize())
        ax2.set_xticklabels(['All Tasks'])
        ax2.set_title('Box Plot')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(title, fontsize=self.fontsize + 2)
        plt.tight_layout()
        
        if save_name:
            self._save_figure(fig, save_name)
            
        return fig
    
    def _find_pareto_frontier(self, x_values: List[float], 
                            y_values: List[float],
                            minimize_x: bool = True) -> List[int]:
        """Find indices of points on Pareto frontier."""
        points = list(zip(x_values, y_values, range(len(x_values))))
        
        # Sort by x value
        points.sort(key=lambda p: p[0], reverse=not minimize_x)
        
        pareto_indices = []
        best_y = float('-inf')
        
        for x, y, idx in points:
            if y >= best_y:
                pareto_indices.append(idx)
                best_y = y
                
        return pareto_indices
    
    def _save_figure(self, fig: plt.Figure, name: str):
        """Save figure in multiple formats."""
        for fmt in self.viz_config['plots'][0]['save_format']:
            filepath = os.path.join(self.save_path, f"{name}.{fmt}")
            fig.savefig(filepath, dpi=self.viz_config['plots'][0]['dpi'], 
                       bbox_inches='tight')
            self.logger.info(f"Figure saved: {filepath}")
        plt.close(fig) 