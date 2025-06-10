"""
Baseline implementations for comparative evaluation.
Includes MAML, FOMAML, Reptile, and standard fine-tuning baselines.
"""

import copy
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

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


class FullMAML(BaselineMethod):
    """Full second-order MAML baseline (memory-intensive)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_config = self.config['baselines']['methods'][0]['config']
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
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Full MAML training")
        self.logger.warning("This method is memory-intensive and may not fit on consumer GPUs")
        
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
            
            # Logging
            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}")
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")
                
    def _meta_train_step(self, task_batch: List[Tuple[List, List]]) -> float:
        """Perform one meta-training step with full second-order gradients."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        
        for support_set, query_set in task_batch:
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
    
    def _create_param_dict(self, fast_weights: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create parameter dictionary from flat weight list."""
        param_dict = {}
        idx = 0
        
        # Map LoRA parameters
        for name, layer in self.model.lora_layers.items():
            # lora_A
            param_shape = layer.lora_A.shape
            param_size = layer.lora_A.numel()
            param_dict[f"{name}.lora_A"] = fast_weights[idx].view(param_shape)
            idx += 1
            
            # lora_B
            param_shape = layer.lora_B.shape
            param_size = layer.lora_B.numel()
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
        
        return param_dict
    
    def _functional_forward(self, batch: Dict, param_dict: Dict[str, torch.Tensor]) -> Dict:
        """Functional forward pass with given parameters."""
        # Temporarily replace model parameters with fast weights
        original_params = {}
        
        # Replace LoRA parameters
        for name, layer in self.model.lora_layers.items():
            # Save original parameters
            original_params[f"{name}.lora_A"] = layer.lora_A.data
            original_params[f"{name}.lora_B"] = layer.lora_B.data
            
            # Set new parameters
            layer.lora_A.data = param_dict[f"{name}.lora_A"]
            layer.lora_B.data = param_dict[f"{name}.lora_B"]
        
        # Replace classifier parameters
        original_params["classifier.weight"] = self.model.classifier.weight.data
        self.model.classifier.weight.data = param_dict["classifier.weight"]
        
        if self.model.classifier.bias is not None:
            original_params["classifier.bias"] = self.model.classifier.bias.data
            self.model.classifier.bias.data = param_dict["classifier.bias"]
        
        # Forward pass with new parameters
        outputs = self.model(**batch)
        
        # Restore original parameters
        for name, layer in self.model.lora_layers.items():
            layer.lora_A.data = original_params[f"{name}.lora_A"]
            layer.lora_B.data = original_params[f"{name}.lora_B"]
        
        self.model.classifier.weight.data = original_params["classifier.weight"]
        if self.model.classifier.bias is not None:
            self.model.classifier.bias.data = original_params["classifier.bias"]
        
        return outputs
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        
        for support_set, query_set in test_tasks:
            # Adapt to support set
            adapted_model = self._adapt_to_task(support_set)
            
            # Evaluate on query set
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                
                with torch.no_grad():
                    outputs = adapted_model(**batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == batch['labels']).float().mean()
                total_accuracy += accuracy.item()
                
        return {
            'test_loss': total_loss / len(test_tasks),
            'test_accuracy': total_accuracy / len(test_tasks)
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
        self.meta_config = self.config['baselines']['methods'][1]['config']
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
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting FOMAML training")
        
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
            
            # Logging
            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}")
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")
                
    def _meta_train_step(self, task_batch: List[Tuple[List, List]]) -> float:
        """Perform one meta-training step with first-order approximation."""
        self.model.train()
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        accumulated_grads = []
        
        # Save original parameters once
        original_params = [p.clone() for p in self.model.get_lora_parameters()]
        
        for support_set, query_set in task_batch:
            # Reset to original parameters for each task
            for p, orig_p in zip(self.model.get_lora_parameters(), original_params):
                p.data = orig_p.data.clone()
            
            # Inner loop adaptation
            inner_optimizer = torch.optim.SGD(
                self.model.get_lora_parameters(), lr=self.inner_lr
            )
            
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    inner_optimizer.zero_grad()
                    loss.backward()
                    inner_optimizer.step()
                    
            # Compute query loss at adapted parameters
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            # Zero grad before computing query loss
            self.meta_optimizer.zero_grad()
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self.model(**batch)
                query_loss = outputs['loss']
                
                # First-order approximation: compute gradients at adapted params
                query_loss.backward()
                total_loss += query_loss.item()
                
            # Store gradients computed at adapted parameters
            if len(accumulated_grads) == 0:
                accumulated_grads = [p.grad.clone() if p.grad is not None else None 
                                   for p in self.model.get_lora_parameters()]
            else:
                for i, p in enumerate(self.model.get_lora_parameters()):
                    if p.grad is not None:
                        if accumulated_grads[i] is None:
                            accumulated_grads[i] = p.grad.clone()
                        else:
                            accumulated_grads[i] += p.grad
                
        # Restore original parameters before applying gradients
        for p, orig_p in zip(self.model.get_lora_parameters(), original_params):
            p.data = orig_p.data
            
        # Apply accumulated gradients
        for p, grad in zip(self.model.get_lora_parameters(), accumulated_grads):
            if grad is not None:
                p.grad = grad / len(task_batch)
                
        self.meta_optimizer.step()
        
        return total_loss / len(task_batch)
    
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks."""
        return self._evaluate_common(test_tasks)
    
    def _evaluate_common(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Common evaluation logic."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        
        for support_set, query_set in tqdm(test_tasks, desc="Evaluating"):
            # Clone model for adaptation
            adapted_model = copy.deepcopy(self.model)
            optimizer = torch.optim.SGD(
                adapted_model.get_lora_parameters(), lr=self.inner_lr
            )
            
            # Adapt to support set
            adapted_model.train()
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
                    
            # Evaluate on query set
            adapted_model.eval()
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                
                with torch.no_grad():
                    outputs = adapted_model(**batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == batch['labels']).float().mean()
                total_accuracy += accuracy.item()
                
        return {
            'test_loss': total_loss / len(test_tasks),
            'test_accuracy': total_accuracy / len(test_tasks)
        }


class Reptile(BaselineMethod):
    """Reptile meta-learning baseline."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_config = self.config['baselines']['methods'][2]['config']
        self.lr = self.meta_config['lr']
        self.inner_steps = self.meta_config['inner_steps']
        self.epsilon = self.meta_config['epsilon']
        self.meta_batch_size = self.config['meta_learning']['default_meta_batch_size']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Train with Reptile algorithm."""
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Reptile training")
        
        pbar = tqdm(range(num_iterations), desc="Reptile Training")
        for iteration in pbar:
            # Sample task
            task_idx = np.random.choice(len(meta_train_tasks))
            support_set, _ = meta_train_tasks[task_idx]
            
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
                
            # Logging
            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Loss = {avg_loss:.4f}")
                
            # Validation
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks[:50])  # Subset for speed
                self.logger.info(f"Validation: {val_metrics}")
                
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks."""
        return FOMAML._evaluate_common(self, test_tasks)


class StandardFineTuning(BaselineMethod):
    """Standard fine-tuning baseline (no meta-learning)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_config = self.config['baselines']['methods'][3]['config']
        self.lr = self.ft_config['lr']
        self.epochs = self.ft_config['epochs']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """Standard training on all tasks (no meta-learning)."""
        self.logger.info("Standard fine-tuning baseline doesn't use meta-training")
        self.logger.info("Model will be adapted from scratch for each test task")
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate by fine-tuning from scratch on each task."""
        total_loss = 0.0
        total_accuracy = 0.0
        
        for support_set, query_set in tqdm(test_tasks, desc="Fine-tuning"):
            # Reset model to pre-trained weights
            adapted_model = copy.deepcopy(self.model)
            optimizer = torch.optim.Adam(adapted_model.get_lora_parameters(), lr=self.lr)
            
            # Fine-tune on support set
            adapted_model.train()
            for epoch in range(self.epochs):
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
                    
            # Evaluate on query set
            adapted_model.eval()
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                
                with torch.no_grad():
                    outputs = adapted_model(**batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == batch['labels']).float().mean()
                total_accuracy += accuracy.item()
                
        return {
            'test_loss': total_loss / len(test_tasks),
            'test_accuracy': total_accuracy / len(test_tasks)
        }


class LoRAFineTuning(BaselineMethod):
    """LoRA fine-tuning baseline (no meta-learning)."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ft_config = self.config['baselines']['methods'][4]['config']
        self.lr = self.ft_config['lr']
        self.epochs = self.ft_config['epochs']
        
    def train(self, meta_train_tasks: List[Tuple[List, List]], 
             meta_val_tasks: Optional[List[Tuple[List, List]]] = None):
        """LoRA fine-tuning doesn't use meta-training."""
        self.logger.info("LoRA fine-tuning baseline doesn't use meta-training")
        self.logger.info("LoRA parameters will be adapted from scratch for each test task")
        
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate by fine-tuning LoRA parameters on each task."""
        # Same as standard fine-tuning but already using LoRA
        return StandardFineTuning.evaluate(self, test_tasks)


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