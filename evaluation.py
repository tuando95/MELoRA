"""
Evaluation module for MELoRA experiments.
Handles model evaluation, metrics computation, and result analysis.
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm
import pandas as pd

from model import MELoRAModel
from dataset_loader import DatasetLoader
import utils


class Evaluator:
    """Handles evaluation procedures and metrics computation."""
    
    def __init__(self,
                 model: Union[MELoRAModel, nn.Module],
                 config: Dict,
                 dataset_loader: DatasetLoader,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.dataset_loader = dataset_loader
        self.device = device or utils.get_device(config)
        self.model.to(self.device)
        
        # Evaluation configuration
        self.eval_config = config['evaluation']
        self.metrics_config = self.eval_config['metrics']
        self.memory_config = self.eval_config['memory_profiling']
        self.protocol_config = self.eval_config['protocol']
        self.efficiency_config = self.eval_config['efficiency_metrics']
        
        # Initialize memory profiler
        self.memory_profiler = utils.MemoryProfiler(config)
        
        # Results storage
        self.results = defaultdict(list)
        self.task_results = []
        
        self.logger = utils.get_logger()
        
    def evaluate_meta_learning(self,
                             test_tasks: List[Tuple[List, List]],
                             adaptation_steps: Optional[int] = None,
                             adaptation_lr: Optional[float] = None) -> Dict[str, Any]:
        """Evaluate meta-learning model on test tasks."""
        if adaptation_steps is None:
            adaptation_steps = self.config['meta_learning']['inner']['default_num_steps']
        if adaptation_lr is None:
            adaptation_lr = self.config['meta_learning']['inner']['default_lr']
            
        self.logger.info(f"Evaluating on {len(test_tasks)} test tasks")
        self.logger.info(f"Adaptation steps: {adaptation_steps}, LR: {adaptation_lr}")
        
        # Reset results
        self.results.clear()
        self.task_results.clear()
        
        # Profile initial memory
        if self.memory_config['enabled']:
            initial_memory = self.memory_profiler.profile_memory('eval_initial')
            
        # Evaluate each task
        for task_idx, (support_set, query_set) in enumerate(
            tqdm(test_tasks, desc="Evaluating tasks")
        ):
            task_metrics = self._evaluate_single_task(
                support_set, query_set, 
                adaptation_steps, adaptation_lr,
                task_idx
            )
            self.task_results.append(task_metrics)
            
            # Aggregate metrics
            for key, value in task_metrics.items():
                if isinstance(value, (int, float)):
                    self.results[key].append(value)
                    
        # Compute aggregate statistics
        aggregate_results = self._compute_aggregate_metrics()
        
        # Add memory profiling results
        if self.memory_config['enabled']:
            memory_results = self._compute_memory_metrics()
            aggregate_results['memory'] = memory_results
            
        # Add efficiency metrics
        if self.efficiency_config['measure_latency']:
            efficiency_results = self._compute_efficiency_metrics()
            aggregate_results['efficiency'] = efficiency_results
            
        return aggregate_results
    
    def _evaluate_single_task(self,
                            support_set: List[Dict],
                            query_set: List[Dict],
                            adaptation_steps: int,
                            adaptation_lr: float,
                            task_idx: int) -> Dict[str, Any]:
        """Evaluate model on a single task."""
        start_time = time.time()
        
        # Adapt model to support set
        adapted_params = self._adapt_to_task(
            support_set, adaptation_steps, adaptation_lr
        )
        
        adaptation_time = time.time() - start_time
        
        # Evaluate on query set
        query_metrics = self._evaluate_on_query_set(
            query_set, adapted_params, task_idx
        )
        
        query_metrics['adaptation_time'] = adaptation_time
        query_metrics['support_size'] = len(support_set)
        query_metrics['query_size'] = len(query_set)
        
        # Profile memory if enabled
        if self.memory_config['enabled'] and task_idx % self.memory_config['profile_interval'] == 0:
            memory_stats = self.memory_profiler.profile_memory(f'task_{task_idx}')
            query_metrics['memory'] = memory_stats
            
        return query_metrics
    
    def _get_num_classes_from_data(self, data: List[Dict]) -> int:
        """Extract number of classes from task data."""
        if not data:
            return 3  # Default fallback
        
        labels = [example['label'] for example in data]
        unique_labels = set(labels)
        num_classes = len(unique_labels)
        
        # Validate labels are in expected range
        min_label, max_label = min(unique_labels), max(unique_labels)
        if min_label < 0 or max_label >= num_classes:
            # Assume labels are 0-indexed and max_label + 1 is the number of classes
            num_classes = max_label + 1
        
        return num_classes

    def _adapt_to_task(self,
                      support_set: List[Dict],
                      adaptation_steps: int,
                      adaptation_lr: float) -> List[torch.Tensor]:
        """Adapt model parameters to a specific task."""
        # Set the number of classes for this task
        if hasattr(self.model, 'set_num_classes'):
            num_classes = self._get_num_classes_from_data(support_set)
            self.model.set_num_classes(num_classes)
            
        # Get LoRA parameters
        if hasattr(self.model, 'get_lora_parameters'):
            adapted_params = [p.clone() for p in self.model.get_lora_parameters()]
        else:
            adapted_params = [p.clone() for p in self.model.parameters() if p.requires_grad]
            
        # Create data loader
        support_loader = self.dataset_loader.get_data_loader(
            support_set, batch_size=len(support_set), shuffle=True
        )
        
        # Adaptation loop
        for step in range(adaptation_steps):
            for batch in support_loader:
                batch = utils.move_to_device(batch, self.device)
                
                # Temporarily replace parameters
                original_params = self._replace_parameters(adapted_params)
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Compute gradients
                grads = torch.autograd.grad(
                    loss, adapted_params, create_graph=False, allow_unused=True
                )
                
                # Handle None gradients for unused parameters
                grads = [g if g is not None else torch.zeros_like(p) 
                        for g, p in zip(grads, adapted_params)]
                
                # Restore original parameters
                self._replace_parameters(original_params)
                
                # Update adapted parameters
                adapted_params = [
                    p - adaptation_lr * g for p, g in zip(adapted_params, grads)
                ]
                
        return adapted_params
    
    def _evaluate_on_query_set(self,
                             query_set: List[Dict],
                             adapted_params: List[torch.Tensor],
                             task_idx: int) -> Dict[str, Any]:
        """Evaluate adapted model on query set."""
        # Set the number of classes for this task
        if hasattr(self.model, 'set_num_classes'):
            num_classes = self._get_num_classes_from_data(query_set)
            self.model.set_num_classes(num_classes)
            
        query_loader = self.dataset_loader.get_data_loader(
            query_set, batch_size=len(query_set), shuffle=False
        )
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        
        self.model.eval()
        with torch.no_grad():
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                
                # Use adapted parameters
                original_params = self._replace_parameters(adapted_params)
                
                outputs = self.model(**batch)
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Restore parameters
                self._replace_parameters(original_params)
                
                # Collect predictions
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                total_loss += loss.item()
                
        # Compute metrics
        metrics = self._compute_classification_metrics(
            all_predictions, all_labels, all_logits
        )
        metrics['loss'] = total_loss / len(query_loader)
        
        return metrics
    
    def _compute_classification_metrics(self,
                                      predictions: List[int],
                                      labels: List[int],
                                      logits: List[np.ndarray]) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        
        # Precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_macro'] = f1
        
        # Micro F1
        _, _, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average='micro', zero_division=0
        )
        metrics['f1_micro'] = f1_micro
        
        # Per-class metrics
        per_class_metrics = classification_report(
            labels, predictions, output_dict=True, zero_division=0
        )
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Confidence metrics
        if logits:
            all_logits = np.vstack(logits)
            probs = F.softmax(torch.tensor(all_logits), dim=1).numpy()
            
            # Average confidence
            max_probs = np.max(probs, axis=1)
            metrics['avg_confidence'] = np.mean(max_probs)
            
            # Calibration error (simplified)
            correct = np.array(predictions) == np.array(labels)
            metrics['avg_confidence_correct'] = np.mean(max_probs[correct])
            metrics['avg_confidence_incorrect'] = np.mean(max_probs[~correct])
            
        return metrics
    
    def _compute_aggregate_metrics(self) -> Dict[str, Any]:
        """Compute aggregate statistics over all tasks."""
        aggregate = {}
        
        # Compute mean and std for each metric
        for metric_name, values in self.results.items():
            if values and isinstance(values[0], (int, float)):
                aggregate[f'{metric_name}_mean'] = np.mean(values)
                aggregate[f'{metric_name}_std'] = np.std(values)
                aggregate[f'{metric_name}_min'] = np.min(values)
                aggregate[f'{metric_name}_max'] = np.max(values)
                
                # Confidence intervals
                if self.protocol_config['bootstrap_samples'] > 0:
                    ci_lower, ci_upper = self._bootstrap_confidence_interval(
                        values, 
                        self.protocol_config['confidence_level'],
                        self.protocol_config['bootstrap_samples']
                    )
                    aggregate[f'{metric_name}_ci_lower'] = ci_lower
                    aggregate[f'{metric_name}_ci_upper'] = ci_upper
                    
        # Task-level statistics
        aggregate['num_tasks'] = len(self.task_results)
        aggregate['avg_support_size'] = np.mean([t['support_size'] for t in self.task_results])
        aggregate['avg_query_size'] = np.mean([t['query_size'] for t in self.task_results])
        
        return aggregate
    
    def _bootstrap_confidence_interval(self,
                                     values: List[float],
                                     confidence_level: float,
                                     n_samples: int) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        bootstrap_means = []
        n = len(values)
        
        for _ in range(n_samples):
            # Resample with replacement
            sample = np.random.choice(values, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))
            
        # Compute percentiles
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return ci_lower, ci_upper
    
    def _compute_memory_metrics(self) -> Dict[str, Any]:
        """Compute memory usage metrics."""
        memory_metrics = {}
        
        # Get all memory measurements
        measurements = self.memory_profiler.measurements
        
        if measurements:
            # Peak memory usage
            gpu_measurements = [m.get('allocated_mb', 0) for m in measurements]
            if gpu_measurements:
                memory_metrics['peak_gpu_mb'] = np.max(gpu_measurements)
                memory_metrics['avg_gpu_mb'] = np.mean(gpu_measurements)
                
            # Memory efficiency ratio
            if hasattr(self.model, 'get_memory_usage'):
                model_memory = self.model.get_memory_usage()
                memory_metrics.update(model_memory)
                
        # Memory breakdown
        if self.memory_config['detailed_breakdown']:
            breakdown = self.memory_profiler.get_memory_breakdown(self.model)
            memory_metrics['breakdown'] = breakdown
            
        return memory_metrics
    
    def _compute_efficiency_metrics(self) -> Dict[str, Any]:
        """Compute computational efficiency metrics."""
        efficiency_metrics = {}
        
        # Latency statistics
        if 'adaptation_time' in self.results:
            times = self.results['adaptation_time']
            efficiency_metrics['avg_adaptation_time'] = np.mean(times)
            efficiency_metrics['std_adaptation_time'] = np.std(times)
            
        # Throughput
        total_examples = sum(t['support_size'] + t['query_size'] for t in self.task_results)
        total_time = sum(self.results.get('adaptation_time', []))
        if total_time > 0:
            efficiency_metrics['throughput_examples_per_sec'] = total_examples / total_time
            
        # FLOPs estimation (if available)
        if self.efficiency_config['measure_flops']:
            # This would require model-specific implementation
            pass
            
        return efficiency_metrics
    
    def _replace_parameters(self, new_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Temporarily replace model parameters."""
        original_params = []
        
        if hasattr(self.model, 'lora_layers'):
            # MELoRA model
            param_idx = 0
            for lora_layer in self.model.lora_layers.values():
                original_params.append(lora_layer.lora_A.data.clone())
                lora_layer.lora_A.data = new_params[param_idx]
                param_idx += 1
                
                original_params.append(lora_layer.lora_B.data.clone())
                lora_layer.lora_B.data = new_params[param_idx]
                param_idx += 1
                
            if hasattr(self.model, 'classifier'):
                for param in self.model.classifier.parameters():
                    original_params.append(param.data.clone())
                    param.data = new_params[param_idx]
                    param_idx += 1
        else:
            # Generic model
            param_idx = 0
            for param in self.model.parameters():
                if param.requires_grad:
                    original_params.append(param.data.clone())
                    param.data = new_params[param_idx]
                    param_idx += 1
                    
        return original_params
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj

    def save_results(self, filepath: str, format: str = 'json'):
        """Save evaluation results."""
        results_data = {
            'aggregate_metrics': self._compute_aggregate_metrics(),
            'task_results': self.task_results,
            'config': self.config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if format == 'json':
            # Convert numpy types to serializable types
            serializable_data = self._convert_to_serializable(results_data)
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
        elif format == 'csv':
            # Save aggregate metrics as CSV
            df = pd.DataFrame([results_data['aggregate_metrics']])
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self.logger.info(f"Results saved to: {filepath}")
        
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """Generate a comprehensive evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("MELoRA Evaluation Report")
        report.append("=" * 80)
        report.append("")
        
        # Summary statistics
        aggregate = self._compute_aggregate_metrics()
        report.append("Summary Statistics:")
        report.append("-" * 40)
        
        for metric in ['accuracy', 'f1_macro', 'loss']:
            if f'{metric}_mean' in aggregate:
                mean = aggregate[f'{metric}_mean']
                std = aggregate[f'{metric}_std']
                report.append(f"{metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")
                
                if f'{metric}_ci_lower' in aggregate:
                    ci_lower = aggregate[f'{metric}_ci_lower']
                    ci_upper = aggregate[f'{metric}_ci_upper']
                    report.append(f"{'':15s}  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
                    
        report.append("")
        
        # Memory usage
        if 'memory' in aggregate:
            report.append("Memory Usage:")
            report.append("-" * 40)
            memory = aggregate['memory']
            if 'peak_gpu_mb' in memory:
                report.append(f"Peak GPU Memory: {memory['peak_gpu_mb']:.2f} MB")
            if 'lora_memory_mb' in memory:
                report.append(f"LoRA Parameters: {memory['lora_memory_mb']:.2f} MB")
            if 'lora_ratio' in memory:
                report.append(f"LoRA/Total Ratio: {memory['lora_ratio']:.4f}")
            report.append("")
            
        # Efficiency metrics
        if 'efficiency' in aggregate:
            report.append("Efficiency Metrics:")
            report.append("-" * 40)
            efficiency = aggregate['efficiency']
            if 'avg_adaptation_time' in efficiency:
                report.append(f"Avg Adaptation Time: {efficiency['avg_adaptation_time']:.3f}s")
            if 'throughput_examples_per_sec' in efficiency:
                report.append(f"Throughput: {efficiency['throughput_examples_per_sec']:.1f} ex/s")
            report.append("")
            
        # Task statistics
        report.append("Task Statistics:")
        report.append("-" * 40)
        report.append(f"Number of Tasks: {aggregate['num_tasks']}")
        report.append(f"Avg Support Size: {aggregate['avg_support_size']:.1f}")
        report.append(f"Avg Query Size: {aggregate['avg_query_size']:.1f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Report saved to: {save_path}")
            
        return report_text
    
    def compare_methods(self, 
                       results_dict: Dict[str, Dict[str, Any]],
                       metrics: List[str] = ['accuracy', 'f1_macro']) -> pd.DataFrame:
        """Compare results from multiple methods."""
        comparison_data = []
        
        for method_name, results in results_dict.items():
            row = {'Method': method_name}
            
            for metric in metrics:
                if f'{metric}_mean' in results:
                    mean = results[f'{metric}_mean']
                    std = results[f'{metric}_std']
                    row[f'{metric}_mean'] = mean
                    row[f'{metric}_std'] = std
                    row[f'{metric}_str'] = f"{mean:.4f} ± {std:.4f}"
                    
            if 'memory' in results and 'peak_gpu_mb' in results['memory']:
                row['Peak GPU (MB)'] = results['memory']['peak_gpu_mb']
                
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        return df 