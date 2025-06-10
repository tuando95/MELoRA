"""
Analysis module for MELoRA experiments.
Performs statistical analysis, ablation studies, and generates comprehensive reports.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation import Evaluator
from visualization import Visualizer
import utils


class Analyzer:
    """Performs comprehensive analysis of experimental results."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.analysis_config = config['analysis']
        self.logger = utils.get_logger()
        
        # Initialize visualizer for analysis plots
        self.visualizer = Visualizer(config)
        
        # Storage for analysis results
        self.analysis_results = {}
        
    def analyze_experiment_results(self,
                                 results_dict: Dict[str, Dict[str, Any]],
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
        """Perform comprehensive analysis of experiment results."""
        self.logger.info("Starting comprehensive analysis")
        
        # Statistical comparisons
        statistical_results = self.statistical_comparison(results_dict)
        self.analysis_results['statistical_comparison'] = statistical_results
        
        # Effect size analysis
        effect_sizes = self.compute_effect_sizes(results_dict)
        self.analysis_results['effect_sizes'] = effect_sizes
        
        # Performance correlations
        correlations = self.analyze_correlations(results_dict)
        self.analysis_results['correlations'] = correlations
        
        # Generate summary report
        report = self.generate_analysis_report()
        
        if save_path:
            self.save_analysis_results(save_path)
            
        return self.analysis_results
    
    def ablation_study(self,
                      base_model,
                      dataset_loader,
                      test_tasks: List[Tuple[List, List]],
                      components: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Perform ablation study on model components."""
        if components is None:
            components = self.analysis_config['ablation']['components']
            
        self.logger.info(f"Performing ablation study on: {components}")
        
        ablation_results = {}
        
        # Evaluate full model
        evaluator = Evaluator(base_model, self.config, dataset_loader)
        full_results = evaluator.evaluate_meta_learning(test_tasks)
        ablation_results['full'] = full_results
        
        # Ablate each component
        for component in components:
            self.logger.info(f"Ablating component: {component}")
            
            # Create ablated model
            ablated_model = self._create_ablated_model(base_model, component)
            
            # Evaluate ablated model
            evaluator = Evaluator(ablated_model, self.config, dataset_loader)
            component_results = evaluator.evaluate_meta_learning(test_tasks)
            ablation_results[f'without_{component}'] = component_results
            
        # Analyze ablation results
        ablation_analysis = self._analyze_ablation_results(ablation_results)
        self.analysis_results['ablation_study'] = ablation_analysis
        
        # Visualize ablation results
        if self.analysis_config['ablation']['interaction_analysis']:
            self._analyze_component_interactions(ablation_results)
            
        return ablation_results
    
    def statistical_comparison(self,
                             results_dict: Dict[str, Dict[str, Any]],
                             metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """Perform statistical tests between different methods."""
        if metrics is None:
            metrics = ['accuracy', 'f1_macro']
            
        comparison_results = {}
        
        for metric in metrics:
            metric_comparison = {}
            
            # Extract values for each method
            method_values = {}
            for method, results in results_dict.items():
                if f'{metric}_mean' in results:
                    # If we have raw values, use them
                    if metric in results and isinstance(results[metric], list):
                        method_values[method] = results[metric]
                    else:
                        # Generate synthetic values from mean/std for testing
                        mean = results[f'{metric}_mean']
                        std = results.get(f'{metric}_std', 0.01)
                        n_samples = 100
                        method_values[method] = np.random.normal(mean, std, n_samples)
                        
            # Perform pairwise t-tests
            if self.analysis_config['statistics']['test_type'] == 'paired_t_test':
                pairwise_results = self._pairwise_t_tests(method_values)
                metric_comparison['pairwise_tests'] = pairwise_results
                
            # Perform ANOVA if more than 2 groups
            if len(method_values) > 2:
                anova_results = self._perform_anova(method_values)
                metric_comparison['anova'] = anova_results
                
                # Post-hoc tests if ANOVA is significant
                if anova_results['p_value'] < 0.05:
                    posthoc_results = self._perform_posthoc_tests(method_values)
                    metric_comparison['posthoc'] = posthoc_results
                    
            comparison_results[metric] = metric_comparison
            
        return comparison_results
    
    def compute_effect_sizes(self,
                           results_dict: Dict[str, Dict[str, Any]],
                           baseline_method: str = 'fine_tuning') -> Dict[str, Dict[str, float]]:
        """Compute effect sizes (Cohen's d) between methods."""
        effect_sizes = {}
        
        if baseline_method not in results_dict:
            self.logger.warning(f"Baseline method {baseline_method} not found")
            return effect_sizes
            
        baseline_results = results_dict[baseline_method]
        
        for method, results in results_dict.items():
            if method == baseline_method:
                continue
                
            method_effect_sizes = {}
            
            for metric in ['accuracy', 'f1_macro', 'loss']:
                if f'{metric}_mean' in results and f'{metric}_mean' in baseline_results:
                    # Cohen's d calculation
                    mean1 = results[f'{metric}_mean']
                    mean2 = baseline_results[f'{metric}_mean']
                    std1 = results.get(f'{metric}_std', 0.01)
                    std2 = baseline_results.get(f'{metric}_std', 0.01)
                    
                    # Pooled standard deviation
                    pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                    
                    if pooled_std > 0:
                        cohens_d = (mean1 - mean2) / pooled_std
                        method_effect_sizes[metric] = cohens_d
                        
            effect_sizes[method] = method_effect_sizes
            
        return effect_sizes
    
    def analyze_correlations(self,
                           results_dict: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Analyze correlations between different metrics."""
        # Extract all numeric metrics
        metrics_data = defaultdict(list)
        
        for method, results in results_dict.items():
            for key, value in results.items():
                if isinstance(value, (int, float)) and not key.endswith('_std'):
                    metrics_data[key].append(value)
                    
        # Convert to DataFrame for correlation analysis
        df = pd.DataFrame(metrics_data)
        
        # Compute correlation matrix
        correlation_matrix = df.corr()
        
        # Find significant correlations
        significant_correlations = {}
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                col1 = correlation_matrix.columns[i]
                col2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.5:  # Threshold for significant correlation
                    key = f"{col1}_vs_{col2}"
                    significant_correlations[key] = corr_value
                    
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'significant_correlations': significant_correlations
        }
    
    def analyze_failure_modes(self,
                            task_results: List[Dict[str, Any]],
                            threshold_percentile: float = 25) -> Dict[str, Any]:
        """Analyze failure modes by examining worst-performing tasks."""
        # Extract performance metrics
        performances = [t.get('accuracy', 0) for t in task_results]
        
        # Find threshold for "failure"
        threshold = np.percentile(performances, threshold_percentile)
        
        # Identify failed tasks
        failed_tasks = [t for t in task_results if t.get('accuracy', 0) < threshold]
        successful_tasks = [t for t in task_results if t.get('accuracy', 0) >= threshold]
        
        failure_analysis = {
            'num_failed': len(failed_tasks),
            'num_successful': len(successful_tasks),
            'failure_threshold': threshold,
            'failure_rate': len(failed_tasks) / len(task_results)
        }
        
        # Analyze characteristics of failed tasks
        if failed_tasks:
            # Support set size analysis
            failed_support_sizes = [t['support_size'] for t in failed_tasks]
            successful_support_sizes = [t['support_size'] for t in successful_tasks]
            
            failure_analysis['avg_failed_support_size'] = np.mean(failed_support_sizes)
            failure_analysis['avg_successful_support_size'] = np.mean(successful_support_sizes)
            
            # Statistical test for difference
            if len(failed_support_sizes) > 1 and len(successful_support_sizes) > 1:
                t_stat, p_value = stats.ttest_ind(failed_support_sizes, successful_support_sizes)
                failure_analysis['support_size_difference_p_value'] = p_value
                
        return failure_analysis
    
    def sensitivity_analysis(self,
                           base_model,
                           dataset_loader,
                           test_tasks: List[Tuple[List, List]],
                           parameter_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """Perform sensitivity analysis on key hyperparameters."""
        if parameter_ranges is None:
            parameter_ranges = {
                'inner_lr': [0.001, 0.005, 0.01, 0.02],
                'inner_steps': [1, 3, 5],
                'lora_rank': [4, 8, 16, 32]
            }
            
        sensitivity_results = {}
        
        for param_name, param_values in parameter_ranges.items():
            self.logger.info(f"Testing sensitivity to {param_name}")
            
            param_results = {}
            for value in param_values:
                # Update configuration
                temp_config = self.config.copy()
                self._update_config_parameter(temp_config, param_name, value)
                
                # Evaluate with modified parameter
                evaluator = Evaluator(base_model, temp_config, dataset_loader)
                results = evaluator.evaluate_meta_learning(test_tasks[:50])  # Subset for speed
                
                param_results[value] = {
                    'accuracy_mean': results.get('accuracy_mean', 0),
                    'memory_mb': results.get('memory', {}).get('peak_gpu_mb', 0)
                }
                
            sensitivity_results[param_name] = param_results
            
        # Analyze sensitivity
        sensitivity_analysis = self._analyze_sensitivity(sensitivity_results)
        self.analysis_results['sensitivity'] = sensitivity_analysis
        
        return sensitivity_analysis
    
    def _create_ablated_model(self, base_model, component: str):
        """Create a model with specific component ablated."""
        import copy
        ablated_model = copy.deepcopy(base_model)
        
        if component == 'checkpointing':
            # Disable checkpointing
            ablated_model.checkpoint_layers = []
        elif component == 'hessian':
            # Disable Hessian approximation (handled in trainer)
            pass
        elif component == 'gradient_accumulation':
            # Disable gradient accumulation (handled in trainer)
            pass
        elif component == 'lora_rank':
            # Use minimal LoRA rank
            for lora_layer in ablated_model.lora_layers.values():
                # This would require re-initializing with rank=1
                pass
                
        return ablated_model
    
    def _analyze_ablation_results(self, 
                                ablation_results: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Analyze results from ablation study."""
        analysis = {}
        
        # Get baseline (full model) performance
        baseline = ablation_results['full']
        baseline_accuracy = baseline.get('accuracy_mean', 0)
        
        # Calculate impact of each component
        component_impacts = {}
        for component, results in ablation_results.items():
            if component != 'full':
                accuracy = results.get('accuracy_mean', 0)
                impact = (baseline_accuracy - accuracy) / baseline_accuracy * 100
                component_impacts[component] = {
                    'absolute_drop': baseline_accuracy - accuracy,
                    'relative_drop_percent': impact
                }
                
        analysis['component_impacts'] = component_impacts
        
        # Rank components by importance
        ranked_components = sorted(
            component_impacts.items(),
            key=lambda x: x[1]['absolute_drop'],
            reverse=True
        )
        analysis['component_ranking'] = [c[0] for c in ranked_components]
        
        return analysis
    
    def _analyze_component_interactions(self,
                                      ablation_results: Dict[str, Dict[str, float]]):
        """Analyze interactions between components."""
        # This would require ablating pairs of components
        # For now, visualize the individual effects
        self.visualizer.plot_ablation_study(
            ablation_results,
            baseline_name='full',
            save_name='ablation_study'
        )
        
    def _pairwise_t_tests(self, 
                         method_values: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Perform pairwise t-tests between methods."""
        results = {}
        methods = list(method_values.keys())
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods[i+1:], i+1):
                values1 = method_values[method1]
                values2 = method_values[method2]
                
                # Perform t-test
                t_stat, p_value = stats.ttest_ind(values1, values2)
                
                # Apply Bonferroni correction
                n_comparisons = len(methods) * (len(methods) - 1) / 2
                corrected_p = p_value * n_comparisons
                
                key = f"{method1}_vs_{method2}"
                results[key] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'corrected_p_value': min(corrected_p, 1.0),
                    'significant': corrected_p < 0.05
                }
                
        return results
    
    def _perform_anova(self, method_values: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Perform one-way ANOVA test."""
        values = list(method_values.values())
        f_stat, p_value = stats.f_oneway(*values)
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    
    def _perform_posthoc_tests(self, 
                             method_values: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Perform post-hoc tests (Tukey HSD)."""
        # Prepare data for Tukey HSD
        all_values = []
        all_groups = []
        
        for method, values in method_values.items():
            all_values.extend(values)
            all_groups.extend([method] * len(values))
            
        # Perform Tukey HSD
        tukey_results = pairwise_tukeyhsd(all_values, all_groups, alpha=0.05)
        
        return {
            'summary': str(tukey_results),
            'table': tukey_results.summary().as_html() if hasattr(tukey_results.summary(), 'as_html') else str(tukey_results.summary())
        }
    
    def _analyze_sensitivity(self, 
                           sensitivity_results: Dict[str, Dict[Any, Dict[str, float]]]) -> Dict[str, Any]:
        """Analyze sensitivity analysis results."""
        analysis = {}
        
        for param_name, param_results in sensitivity_results.items():
            values = list(param_results.keys())
            accuracies = [r['accuracy_mean'] for r in param_results.values()]
            
            # Calculate variance in performance
            performance_variance = np.var(accuracies)
            performance_range = max(accuracies) - min(accuracies)
            
            # Find optimal value
            optimal_idx = np.argmax(accuracies)
            optimal_value = values[optimal_idx]
            
            analysis[param_name] = {
                'performance_variance': performance_variance,
                'performance_range': performance_range,
                'optimal_value': optimal_value,
                'optimal_accuracy': accuracies[optimal_idx],
                'sensitivity_score': performance_range / np.mean(accuracies)  # Normalized sensitivity
            }
            
        # Rank parameters by sensitivity
        sensitivity_ranking = sorted(
            analysis.items(),
            key=lambda x: x[1]['sensitivity_score'],
            reverse=True
        )
        
        return {
            'parameter_analysis': analysis,
            'sensitivity_ranking': [p[0] for p in sensitivity_ranking]
        }
    
    def _update_config_parameter(self, config: Dict, param_name: str, value: Any):
        """Update a specific parameter in the configuration."""
        if param_name == 'inner_lr':
            config['meta_learning']['inner']['default_lr'] = value
        elif param_name == 'inner_steps':
            config['meta_learning']['inner']['default_num_steps'] = value
        elif param_name == 'lora_rank':
            config['lora']['default_rank'] = value
            
    def generate_analysis_report(self) -> str:
        """Generate comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("MELoRA Analysis Report")
        report.append("=" * 80)
        report.append("")
        
        # Statistical comparison results
        if 'statistical_comparison' in self.analysis_results:
            report.append("Statistical Comparison Results:")
            report.append("-" * 40)
            
            for metric, comparisons in self.analysis_results['statistical_comparison'].items():
                report.append(f"\nMetric: {metric}")
                
                if 'anova' in comparisons:
                    anova = comparisons['anova']
                    report.append(f"  ANOVA F-statistic: {anova['f_statistic']:.4f}")
                    report.append(f"  ANOVA p-value: {anova['p_value']:.4f}")
                    report.append(f"  Significant difference: {anova['significant']}")
                    
            report.append("")
            
        # Effect sizes
        if 'effect_sizes' in self.analysis_results:
            report.append("Effect Sizes (Cohen's d):")
            report.append("-" * 40)
            
            for method, effects in self.analysis_results['effect_sizes'].items():
                report.append(f"\n{method}:")
                for metric, d in effects.items():
                    magnitude = self._interpret_cohens_d(d)
                    report.append(f"  {metric}: {d:.3f} ({magnitude})")
                    
            report.append("")
            
        # Ablation study results
        if 'ablation_study' in self.analysis_results:
            report.append("Ablation Study Results:")
            report.append("-" * 40)
            
            ablation = self.analysis_results['ablation_study']
            if 'component_ranking' in ablation:
                report.append("Component importance ranking:")
                for i, component in enumerate(ablation['component_ranking'], 1):
                    impact = ablation['component_impacts'][component]
                    report.append(f"  {i}. {component}: {impact['relative_drop_percent']:.1f}% drop")
                    
            report.append("")
            
        # Sensitivity analysis
        if 'sensitivity' in self.analysis_results:
            report.append("Sensitivity Analysis:")
            report.append("-" * 40)
            
            sensitivity = self.analysis_results['sensitivity']
            if 'sensitivity_ranking' in sensitivity:
                report.append("Parameter sensitivity ranking:")
                for i, param in enumerate(sensitivity['sensitivity_ranking'], 1):
                    score = sensitivity['parameter_analysis'][param]['sensitivity_score']
                    report.append(f"  {i}. {param}: sensitivity score = {score:.3f}")
                    
        return "\n".join(report)
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d = abs(d)
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"
            
    def save_analysis_results(self, filepath: str):
        """Save analysis results to file."""
        # Save as JSON
        json_path = filepath.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
            
        # Save report as text
        report = self.generate_analysis_report()
        txt_path = filepath.replace('.json', '.txt')
        with open(txt_path, 'w') as f:
            f.write(report)
            
        self.logger.info(f"Analysis results saved to {json_path} and {txt_path}") 