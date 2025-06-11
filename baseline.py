"""
Baseline implementations for comparative evaluation.
Includes MAML, FOMAML, Reptile, and standard fine-tuning baselines.

All baselines use the same training/validation/test data as MELoRA for fair comparison:
- Meta-learning baselines (MAML, FOMAML, Reptile) use full training data
- All methods use identical test tasks for evaluation
- Validation during training uses subsets (50 tasks) for computational efficiency
- All methods use the unified Evaluator protocol for consistent metrics

Memory Requirements Summary:
- FullMAML: HIGH (16GB+ GPU) - Second-order gradients with computation graph retention
- FOMAML: MODERATE (8GB+ GPU) - First-order approximation, 50-70% less memory than FullMAML
- Reptile: LOW (4GB+ GPU) - Most memory efficient, only stores initial/final parameters
- Standard Fine-tuning: LOW (4GB+ GPU) - No meta-learning overhead

See memory_requirements.md for detailed analysis and optimization strategies.
"""

import copy
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod
from collections import defaultdict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import MELoRAModel
from dataset_loader import DatasetLoader
import utils


class BaselineMethod(ABC):
    """Abstract base class for baseline methods."""
    
    def __init__(self, 
                 model: MELoRAModel,
                 config: Dict,
                 dataset_loader: DatasetLoader,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.dataset_loader = dataset_loader
        self.device = device or utils.get_device(config)
        self.model.to(self.device)
        self.logger = utils.get_logger()
        
        # Initialize memory profiler for tracking
        self.memory_profiler = utils.MemoryProfiler(config)
        
        # Training state tracking
        self.global_step = 0
        self.metrics_history = defaultdict(list)
        
    @abstractmethod
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Train the baseline method."""
        pass
    
    @abstractmethod
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate the baseline method."""
        pass
    
    def save_checkpoint(self, filepath: str, metrics: Optional[Dict] = None):
        """Save model checkpoint."""
        self.model.save_checkpoint(filepath, metrics=metrics)
        
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        return self.model.load_checkpoint(filepath)
    
    def _get_method_config(self, method_name: str) -> Dict:
        """Get configuration for a specific method by name."""
        for method in self.config['baselines']['methods']:
            if method['name'] == method_name:
                return method['config']
        raise ValueError(f"Method {method_name} not found in baseline configuration")
    
    def _log_training_metrics(self, iteration: int, loss: float, method_name: str):
        """Log training metrics with memory profiling."""
        metrics = {
            'meta_train_loss': loss,
            'learning_rate': getattr(self, 'meta_optimizer', None) and self.meta_optimizer.param_groups[0]['lr'] or 0.0
        }
        
        # Add memory metrics
        memory_stats = self.memory_profiler.profile_memory(f'{method_name}_iter_{iteration}')
        # Only add numeric memory stats to metrics (exclude 'tag' and 'timestamp')
        numeric_memory_stats = {k: v for k, v in memory_stats.items() 
                              if isinstance(v, (int, float)) and k not in ['tag', 'timestamp']}
        metrics.update({f'memory/{k}': v for k, v in numeric_memory_stats.items()})
        
        # Store in history for later analysis
        for key, value in metrics.items():
            self.metrics_history[f'train/{key}'].append(value)
        
        utils.log_metrics(metrics, iteration, prefix='train')
        
    def generate_summary_statistics(self, results: Dict[str, Any]) -> str:
        """Generate summary statistics similar to MELoRA evaluation."""
        summary = []
        summary.append("Summary Statistics:")
        summary.append("-" * 40)
        
        # Extract key metrics with confidence intervals if available
        for metric in ['accuracy', 'f1_macro', 'loss']:
            if f'{metric}_mean' in results:
                mean = results[f'{metric}_mean']
                std = results[f'{metric}_std']
                summary.append(f"{metric.capitalize():15s}: {mean:.4f} ± {std:.4f}")
                
                if f'{metric}_ci_lower' in results:
                    ci_lower = results[f'{metric}_ci_lower']
                    ci_upper = results[f'{metric}_ci_upper']
                    summary.append(f"{'':15s}  95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return "\n".join(summary)


class FullMAML(BaselineMethod):
    """Full second-order MAML baseline (memory-intensive)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_config = self._get_method_config('full_maml')
        self.inner_lr = self.meta_config['inner_lr']
        self.outer_lr = self.meta_config['outer_lr']
        self.inner_steps = self.meta_config['inner_steps']
        self.meta_batch_size = self.config['meta_learning']['default_meta_batch_size']
        
        # Initialize optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),  # All parameters, not just LoRA
            lr=self.outer_lr
        )
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Train with full MAML (second-order)."""
        if not meta_train_tasks:
            raise ValueError("Empty meta_train_tasks provided")
            
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Full MAML training")
        self.logger.warning("This method is memory-intensive and may not fit on consumer GPUs")
        
        # Log initial memory usage
        initial_memory = self.memory_profiler.profile_memory('fullmaml_initial')
        self.logger.info(f"Initial memory: {initial_memory}")
        
        # Check GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if gpu_memory < 8:
                self.logger.warning(f"Only {gpu_memory:.1f}GB GPU memory available. FullMAML may run out of memory.")
        
        pbar = tqdm(range(num_iterations), desc="FullMAML Training")
        for iteration in pbar:
            # Sample task batch
            task_indices = np.random.choice(len(meta_train_tasks), 
                                          self.meta_batch_size, replace=True)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Meta-training step
            meta_loss = self._meta_train_step(task_batch)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{meta_loss:.4f}'})
            
            # Logging with memory tracking
            log_interval = self.config.get('training', {}).get('log_interval', 100)
            if iteration % log_interval == 0:
                self._log_training_metrics(iteration, meta_loss, 'FullMAML')
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks[:50])  # Subset for speed
                self.logger.info(f"Validation: {val_metrics}")
                
            self.global_step = iteration
                
    def _meta_train_step(self, task_batch: List[Tuple[List, List]]) -> float:
        """Perform one meta-training step with full second-order gradients."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        
        try:
        
        for support_set, query_set in task_batch:
            # Set the number of classes for this task
            num_classes = self._get_num_classes_from_data(support_set + query_set)
            self.model.set_num_classes(num_classes)
            # Get initial LoRA parameters
            lora_params = self.model.get_lora_parameters()
            fast_weights = [p.clone() for p in lora_params]
            
            # Create parameter mapping for functional forward
            param_dict = self._create_param_dict(fast_weights)
            
            # Inner loop adaptation
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    
                    # Forward with fast weights
                    outputs = self._functional_forward(batch, param_dict)
                    loss = outputs['loss']
                    
                    # Compute gradients
                    grads = torch.autograd.grad(loss, fast_weights, create_graph=True, allow_unused=True)
                    
                    # Handle None gradients for unused parameters
                    grads = [g if g is not None else torch.zeros_like(w) 
                            for g, w in zip(grads, fast_weights)]
                    
                    # Update fast weights
                    fast_weights = [w - self.inner_lr * g 
                                  for w, g in zip(fast_weights, grads)]
                    
                    # Update parameter dictionary
                    param_dict = self._create_param_dict(fast_weights)
                    
            # Compute query loss
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self._functional_forward(batch, param_dict)
                query_loss = outputs['loss']
                total_loss += query_loss
                
        # Backward through everything (second-order)
        total_loss = total_loss / len(task_batch)
        total_loss.backward()
        
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.meta_optimizer.step()
            
            return total_loss.item()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                self.logger.error("GPU out of memory during FullMAML training. Try reducing batch size or using FOMAML.")
                torch.cuda.empty_cache()  # Clear GPU cache
            raise
        except Exception as e:
            self.logger.error(f"Error during meta-training step: {e}")
            raise
    
    def _get_num_classes_from_data(self, data: List[Dict]) -> int:
        """Extract number of classes from task data."""
        if not data:
            self.logger.warning("Empty dataset provided, using default number of classes")
            return self.model.max_num_labels  # Default fallback
        
        try:
            labels = [example['label'] for example in data]
            unique_labels = set(labels)
            num_classes = len(unique_labels)
            
            # Validate labels are in expected range
            min_label, max_label = min(unique_labels), max(unique_labels)
            if min_label < 0:
                raise ValueError(f"Found negative label: {min_label}")
            
            # Use max_label + 1 as number of classes to handle non-contiguous labels
            num_classes = max_label + 1
            
            if num_classes > self.model.max_num_labels:
                self.logger.warning(f"Task has {num_classes} classes, but model supports max {self.model.max_num_labels}")
                
            return min(num_classes, self.model.max_num_labels)
            
        except KeyError as e:
            self.logger.error(f"Missing 'label' key in data: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error extracting number of classes: {e}")
            raise
    
    def _create_param_dict(self, fast_weights: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create parameter dictionary from flat weight list."""
        if len(fast_weights) == 0:
            raise ValueError("Empty fast_weights list provided")
            
        param_dict = {}
        idx = 0
        expected_params = len(self.model.lora_layers) * 2 + 1  # 2 per LoRA layer + classifier weight
        if self.model.classifier.bias is not None:
            expected_params += 1  # classifier bias
            
        if len(fast_weights) != expected_params:
            raise ValueError(f"Expected {expected_params} parameters, got {len(fast_weights)}")
        
        # Map LoRA parameters
        for name, layer in self.model.lora_layers.items():
            # lora_A
            param_shape = layer.lora_A.shape
            param_dict[f"{name}.lora_A"] = fast_weights[idx].view(param_shape)
            idx += 1
            
            # lora_B
            param_shape = layer.lora_B.shape
            param_dict[f"{name}.lora_B"] = fast_weights[idx].view(param_shape)
            idx += 1
        
        # Map classifier parameters
        # Weight
        param_shape = self.model.classifier.weight.shape
        param_dict["classifier.weight"] = fast_weights[idx].view(param_shape)
        idx += 1
        
        # Bias
        if self.model.classifier.bias is not None:
            param_shape = self.model.classifier.bias.shape
            param_dict["classifier.bias"] = fast_weights[idx].view(param_shape)
            idx += 1
        
        return param_dict
    
    def _functional_forward(self, batch: Dict, param_dict: Dict[str, torch.Tensor]) -> Dict:
        """Functional forward pass with given parameters."""
        try:
            # Temporarily replace model parameters with fast weights
            original_params = {}
            
            # Replace LoRA parameters
            for name, layer in self.model.lora_layers.items():
                # Save original parameters
                original_params[f"{name}.lora_A"] = layer.lora_A.data.clone()
                original_params[f"{name}.lora_B"] = layer.lora_B.data.clone()
                
                # Set new parameters
                layer.lora_A.data = param_dict[f"{name}.lora_A"]
                layer.lora_B.data = param_dict[f"{name}.lora_B"]
            
            # Replace classifier parameters
            original_params["classifier.weight"] = self.model.classifier.weight.data.clone()
            self.model.classifier.weight.data = param_dict["classifier.weight"]
            
            if self.model.classifier.bias is not None:
                original_params["classifier.bias"] = self.model.classifier.bias.data.clone()
                self.model.classifier.bias.data = param_dict["classifier.bias"]
            
            # Forward pass with new parameters
            outputs = self.model(**batch)
            
            return outputs
            
        except Exception as e:
            self.logger.error(f"Error in functional forward pass: {e}")
            raise
        finally:
            # Always restore original parameters
            if 'original_params' in locals():
                for name, layer in self.model.lora_layers.items():
                    if f"{name}.lora_A" in original_params:
                        layer.lora_A.data = original_params[f"{name}.lora_A"]
                    if f"{name}.lora_B" in original_params:
                        layer.lora_B.data = original_params[f"{name}.lora_B"]
                
                if "classifier.weight" in original_params:
                    self.model.classifier.weight.data = original_params["classifier.weight"]
                if "classifier.bias" in original_params and self.model.classifier.bias is not None:
                    self.model.classifier.bias.data = original_params["classifier.bias"]
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks using same protocol as MELoRA."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for fair comparison
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(test_tasks)
        
        # Generate and log summary statistics
        summary = self.generate_summary_statistics(results)
        self.logger.info(f"FullMAML Evaluation Results:\n{summary}")
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0),
            'full_results': results  # Include full results for analysis
        }
    
    def _adapt_to_task(self, support_set: List[Dict]) -> nn.Module:
        """Adapt model to a specific task."""
        adapted_model = copy.deepcopy(self.model)
        optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            support_loader = self.dataset_loader.get_data_loader(
                support_set, batch_size=len(support_set)
            )
            
            for batch in support_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = adapted_model(**batch)
                loss = outputs['loss']
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        return adapted_model


class FOMAML(BaselineMethod):
    """First-Order MAML baseline."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_config = self._get_method_config('fomaml')
        self.inner_lr = self.meta_config['inner_lr']
        self.outer_lr = self.meta_config['outer_lr']
        self.inner_steps = self.meta_config['inner_steps']
        self.meta_batch_size = self.config['meta_learning']['default_meta_batch_size']
        
        # Only optimize LoRA parameters
        self.meta_optimizer = torch.optim.Adam(
            self.model.get_lora_parameters(),
            lr=self.outer_lr
        )
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Train with FOMAML (first-order approximation)."""
        if not meta_train_tasks:
            raise ValueError("Empty meta_train_tasks provided")
            
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting FOMAML training")
        
        # Log initial memory usage
        initial_memory = self.memory_profiler.profile_memory('fomaml_initial')
        self.logger.info(f"Initial memory: {initial_memory}")
        
        pbar = tqdm(range(num_iterations), desc="FOMAML Training")
        for iteration in pbar:
            # Sample task batch
            task_indices = np.random.choice(len(meta_train_tasks), 
                                          self.meta_batch_size, replace=True)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Meta-training step
            meta_loss = self._meta_train_step(task_batch)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{meta_loss:.4f}'})
            
            # Logging with memory tracking
            log_interval = self.config.get('training', {}).get('log_interval', 100)
            if iteration % log_interval == 0:
                self._log_training_metrics(iteration, meta_loss, 'FOMAML')
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")
                
            self.global_step = iteration
                
    def _meta_train_step(self, task_batch: List[Tuple[List, List]]) -> float:
        """Perform one meta-training step with first-order approximation."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        accumulated_grads = []
        
        # Save original parameters once
        original_params = [p.clone() for p in self.model.get_lora_parameters()]
        
        for support_set, query_set in task_batch:
            # Set the number of classes for this task
            num_classes = self._get_num_classes_from_data(support_set + query_set)
            self.model.set_num_classes(num_classes)
            
            # Reset to original parameters for each task
            for p, orig_p in zip(self.model.get_lora_parameters(), original_params):
                p.data = orig_p.data.clone()
            
            # Inner loop adaptation (SGD on current task)
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    # Manual gradient computation and update (no optimizer needed)
                    grads = torch.autograd.grad(
                        loss, self.model.get_lora_parameters(), 
                        create_graph=False, retain_graph=False, allow_unused=True
                    )
                    
                    # Update parameters manually
                    with torch.no_grad():
                        for p, g in zip(self.model.get_lora_parameters(), grads):
                            if g is not None:
                                p.data = p.data - self.inner_lr * g
                    
            # Compute query loss at adapted parameters
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            task_query_loss = 0.0
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self.model(**batch)
                query_loss = outputs['loss']
                task_query_loss += query_loss.item()
                
                # Compute gradients w.r.t. adapted parameters
                task_grads = torch.autograd.grad(
                    query_loss, self.model.get_lora_parameters(),
                    retain_graph=False, allow_unused=True
                )
                
                # Accumulate gradients (FOMAML: use gradients from adapted params)
                if len(accumulated_grads) == 0:
                    accumulated_grads = [g.clone() if g is not None else torch.zeros_like(p) 
                                       for g, p in zip(task_grads, self.model.get_lora_parameters())]
                else:
                    for i, g in enumerate(task_grads):
                        if g is not None:
                            accumulated_grads[i] += g
                        
            total_loss += task_query_loss
                
        # Restore original parameters before applying meta-gradients
        for p, orig_p in zip(self.model.get_lora_parameters(), original_params):
            p.data = orig_p.data
            
        # Apply accumulated gradients to original parameters
        for p, grad in zip(self.model.get_lora_parameters(), accumulated_grads):
            p.grad = grad / len(task_batch)
                
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.get_lora_parameters(), max_norm=1.0)
        
        self.meta_optimizer.step()
        
        return total_loss / len(task_batch)
    
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks."""
        return self._evaluate_common(test_tasks)
    
    def _evaluate_common(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Common evaluation logic using same protocol as MELoRA."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for fair comparison
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(test_tasks)
        
        # Generate and log summary statistics
        summary = self.generate_summary_statistics(results)
        self.logger.info(f"FOMAML Evaluation Results:\n{summary}")
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0),
            'full_results': results  # Include full results for analysis
        }


class Reptile(BaselineMethod):
    """Reptile meta-learning baseline."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_config = self._get_method_config('reptile')
        self.lr = self.meta_config['lr']
        self.inner_steps = self.meta_config['inner_steps']
        self.epsilon = self.meta_config['epsilon']
        self.meta_batch_size = self.config['meta_learning']['default_meta_batch_size']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Train with Reptile algorithm."""
        if not meta_train_tasks:
            raise ValueError("Empty meta_train_tasks provided")
            
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Reptile training")
        
        # Log initial memory usage
        initial_memory = self.memory_profiler.profile_memory('reptile_initial')
        self.logger.info(f"Initial memory: {initial_memory}")
        
        pbar = tqdm(range(num_iterations), desc="Reptile Training")
        for iteration in pbar:
            # Sample task
            task_idx = np.random.choice(len(meta_train_tasks))
            support_set, query_set = meta_train_tasks[task_idx]
            
            # Set the number of classes for this task
            num_classes = self._get_num_classes_from_data(support_set + query_set)
            self.model.set_num_classes(num_classes)
            
            # Store initial parameters
            initial_params = [p.clone() for p in self.model.get_lora_parameters()]
            
            # Inner loop optimization
            optimizer = torch.optim.SGD(self.model.get_lora_parameters(), lr=self.lr)
            
            total_loss = 0.0
            num_batches = 0
            
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            # Reptile update: move towards adapted parameters
            for p, init_p in zip(self.model.get_lora_parameters(), initial_params):
                p.data = init_p + self.epsilon * (p.data - init_p)
                
            # Update progress bar with average loss
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
            # Logging with memory tracking
            log_interval = self.config.get('training', {}).get('log_interval', 100)
            if iteration % log_interval == 0:
                self._log_training_metrics(iteration, avg_loss, 'Reptile')
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks[:50])  # Subset for speed
                self.logger.info(f"Validation: {val_metrics}")
                
            self.global_step = iteration
                
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks using the unified evaluation protocol."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for a fair comparison, 
        # using Reptile's own learning rate for adaptation.
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(
            test_tasks,
            adaptation_steps=self.inner_steps,
            adaptation_lr=self.lr
        )
        
        # Generate and log summary statistics
        summary = self.generate_summary_statistics(results)
        self.logger.info(f"Reptile Evaluation Results:\n{summary}")
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0),
            'full_results': results  # Include full results for analysis
        }


class StandardFineTuning(BaselineMethod):
    """Standard fine-tuning baseline (no meta-learning)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_config = self._get_method_config('fine_tuning')
        self.lr = self.ft_config['lr']
        self.epochs = self.ft_config['epochs']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Standard training on all tasks (no meta-learning)."""
        self.logger.info("Standard fine-tuning baseline doesn't use meta-training")
        self.logger.info("Model will be adapted from scratch for each test task")
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate by fine-tuning from scratch using same protocol as MELoRA."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for fair comparison
        # But with more epochs for fine-tuning instead of few-shot adaptation
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(
            test_tasks, 
            adaptation_steps=self.epochs,  # Use epochs for adaptation
            adaptation_lr=self.lr
        )
        
        # Generate and log summary statistics
        summary = self.generate_summary_statistics(results)
        self.logger.info(f"Standard Fine-tuning Evaluation Results:\n{summary}")
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0),
            'full_results': results  # Include full results for analysis
        }


class LoRAFineTuning(BaselineMethod):
    """LoRA fine-tuning baseline (no meta-learning)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_config = self._get_method_config('lora_fine_tuning')
        self.lr = self.ft_config['lr']
        self.epochs = self.ft_config['epochs']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """LoRA fine-tuning doesn't use meta-training."""
        self.logger.info("LoRA fine-tuning baseline doesn't use meta-training")
        self.logger.info("LoRA parameters will be adapted from scratch for each test task")
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate by fine-tuning LoRA parameters on each task."""
        from evaluation import Evaluator
        
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(
            test_tasks, 
            adaptation_steps=self.epochs,  # Use epochs for adaptation
            adaptation_lr=self.lr
        )
        
        # Generate and log summary statistics
        summary = self.generate_summary_statistics(results)
        self.logger.info(f"LoRA Fine-tuning Evaluation Results:\n{summary}")
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0),
            'full_results': results  # Include full results for analysis
        }


def create_baseline(baseline_name: str, 
                   model: MELoRAModel,
                   config: Dict,
                   dataset_loader: DatasetLoader) -> BaselineMethod:
    """Factory function to create baseline instances."""
    baselines = {
        'full_maml': FullMAML,
        'fomaml': FOMAML,
        'reptile': Reptile,
        'fine_tuning': StandardFineTuning,
        'lora_fine_tuning': LoRAFineTuning
    }
    
    if baseline_name not in baselines:
        raise ValueError(f"Unknown baseline: {baseline_name}")
        
    return baselines[baseline_name](model, config, dataset_loader) 