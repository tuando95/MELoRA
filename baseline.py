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
        
        for iteration in range(num_iterations):
            # Sample task batch
            task_indices = np.random.choice(len(meta_train_tasks), 
                                          self.meta_batch_size, replace=True)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Meta-training step
            meta_loss = self._meta_train_step(task_batch)
            
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
            # Clone model for inner loop
            fast_weights = []
            for param in self.model.parameters():
                fast_weights.append(param.clone())
                
            # Inner loop adaptation
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    
                    # Forward with fast weights
                    outputs = self._functional_forward(batch, fast_weights)
                    loss = outputs['loss']
                    
                    # Compute gradients
                    grads = torch.autograd.grad(loss, fast_weights, create_graph=True)
                    
                    # Update fast weights
                    fast_weights = [w - self.inner_lr * g 
                                  for w, g in zip(fast_weights, grads)]
                    
            # Compute query loss
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self._functional_forward(batch, fast_weights)
                query_loss = outputs['loss']
                total_loss += query_loss
                
        # Backward through everything (second-order)
        total_loss = total_loss / len(task_batch)
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.meta_optimizer.step()
        
        return total_loss.item()
    
    def _functional_forward(self, batch: Dict, params: List[torch.Tensor]) -> Dict:
        """Functional forward pass with given parameters."""
        # This is a simplified version - full implementation would need
        # to properly handle all model parameters
        raise NotImplementedError("Full functional forward not implemented")
        
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
        
        for iteration in range(num_iterations):
            # Sample task batch
            task_indices = np.random.choice(len(meta_train_tasks), 
                                          self.meta_batch_size, replace=True)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Meta-training step
            meta_loss = self._meta_train_step(task_batch)
            
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
        
        for support_set, query_set in task_batch:
            # Clone current parameters
            original_params = [p.clone() for p in self.model.get_lora_parameters()]
            
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
                    
            # Compute query loss
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set)
            )
            
            for batch in query_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self.model(**batch)
                query_loss = outputs['loss']
                
                # First-order approximation: treat adapted params as constants
                query_loss.backward()
                total_loss += query_loss.item()
                
            # Restore original parameters
            for p, orig_p in zip(self.model.get_lora_parameters(), original_params):
                p.data = orig_p
                
        # Apply accumulated gradients
        for p in self.model.get_lora_parameters():
            if p.grad is not None:
                p.grad = p.grad / len(task_batch)
                
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
        
        for iteration in range(num_iterations):
            # Sample task
            task_idx = np.random.choice(len(meta_train_tasks))
            support_set, _ = meta_train_tasks[task_idx]
            
            # Store initial parameters
            initial_params = [p.clone() for p in self.model.get_lora_parameters()]
            
            # Inner loop optimization
            optimizer = torch.optim.SGD(self.model.get_lora_parameters(), lr=self.lr)
            
            for _ in range(self.inner_steps):
                support_loader = self.dataset_loader.get_data_loader(
                    support_set, batch_size=len(support_set)
                )
                
                for batch in support_loader:
                    batch = utils.move_to_device(batch, self.device)
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
            # Reptile update: move towards adapted parameters
            for p, init_p in zip(self.model.get_lora_parameters(), initial_params):
                p.data = init_p + self.epsilon * (p.data - init_p)
                
            # Logging
            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}")
                
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