"""
MELoRA trainer implementation.
Handles meta-learning training loop with memory-efficient optimizations.
"""

import os
import copy
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from model import MELoRAModel
from dataset_loader import DatasetLoader
import utils


class MELoRATrainer:
    """Trainer for Memory-Efficient LoRA Meta-Learning."""
    
    def __init__(self, 
                 model: MELoRAModel,
                 config: Dict,
                 dataset_loader: DatasetLoader,
                 device: Optional[torch.device] = None):
        self.model = model
        self.config = config
        self.dataset_loader = dataset_loader
        self.device = device or utils.get_device(config)
        
        # Move model to device
        self.model.to(self.device)
        
        # Meta-learning configuration
        self.meta_config = config['meta_learning']
        self.inner_lr = self.meta_config['inner']['default_lr']
        self.outer_lr = self.meta_config['outer']['default_lr']
        self.inner_steps = self.meta_config['inner']['default_num_steps']
        self.meta_batch_size = self.meta_config['default_meta_batch_size']
        
        # Memory optimization configuration
        self.memory_config = config['memory_optimization']
        self.gradient_accumulation_steps = self.memory_config['gradient_accumulation']['default_micro_batch']
        self.use_hessian_approx = self.memory_config['hessian_approximation']['enabled']
        self.hessian_method = self.memory_config['hessian_approximation']['method']
        
        # Initialize optimizers
        self._init_optimizers()
        
        # Initialize memory profiler
        self.memory_profiler = utils.MemoryProfiler(config)
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.logger = utils.get_logger()
        
    def _init_optimizers(self):
        """Initialize inner and outer loop optimizers."""
        # Get LoRA parameters only
        lora_params = self.model.get_lora_parameters()
        
        # Outer loop optimizer (meta-optimizer)
        optimizer_config = self.meta_config['outer']['optimizer_params']
        if self.meta_config['outer']['optimizer'] == 'adamw':
            self.meta_optimizer = torch.optim.AdamW(
                lora_params,
                lr=self.outer_lr,
                betas=optimizer_config['betas'],
                eps=optimizer_config['eps'],
                weight_decay=optimizer_config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.meta_config['outer']['optimizer']}")
            
        # Inner loop optimizer is created per task
        
    def train(self, 
              meta_train_tasks: List[Tuple[List, List]],
              meta_val_tasks: Optional[List[Tuple[List, List]]] = None,
              num_iterations: Optional[int] = None):
        """Main training loop for meta-learning."""
        if num_iterations is None:
            num_iterations = self.meta_config['num_meta_iterations']
            
        self.logger.info(f"Starting MELoRA training for {num_iterations} iterations")
        self.logger.info(f"Meta-batch size: {self.meta_batch_size}")
        self.logger.info(f"Inner steps: {self.inner_steps}, Inner LR: {self.inner_lr}")
        
        # Log initial memory usage
        initial_memory = self.memory_profiler.profile_memory('initial')
        self.logger.info(f"Initial memory: {initial_memory}")
        
        # Create progress bar for training iterations
        pbar = tqdm(range(num_iterations), desc="MELoRA Training", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
        
        for iteration in pbar:
            # Sample meta-batch of tasks
            task_batch = self._sample_task_batch(meta_train_tasks)
            
            # Meta-training step
            meta_train_loss = self._meta_train_step(task_batch)
            
            # Update progress bar with current metrics
            pbar.set_postfix({
                'Loss': f'{meta_train_loss:.4f}',
                'LR': f'{self.meta_optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Logging
            if iteration % self.config['training']['log_interval'] == 0:
                metrics = {
                    'meta_train_loss': meta_train_loss,
                    'learning_rate': self.meta_optimizer.param_groups[0]['lr']
                }
                
                # Add memory metrics
                memory_stats = self.memory_profiler.profile_memory(f'iter_{iteration}')
                # Only add numeric memory stats to metrics (exclude 'tag' and 'timestamp')
                numeric_memory_stats = {k: v for k, v in memory_stats.items() 
                                      if isinstance(v, (int, float)) and k not in ['tag', 'timestamp']}
                metrics.update({f'memory/{k}': v for k, v in numeric_memory_stats.items()})
                
                utils.log_metrics(metrics, iteration, prefix='train')
                
            # Validation
            if meta_val_tasks and iteration % self.meta_config['validation_frequency'] == 0:
                pbar.set_description("Validating...")
                val_metrics = self.validate(meta_val_tasks)
                utils.log_metrics(val_metrics, iteration, prefix='val')
                
                # Update progress bar with validation metrics
                pbar.set_postfix({
                    'Loss': f'{meta_train_loss:.4f}',
                    'Val Loss': f'{val_metrics["meta_val_loss"]:.4f}',
                    'Val Acc': f'{val_metrics["meta_val_accuracy"]:.3f}',
                    'LR': f'{self.meta_optimizer.param_groups[0]["lr"]:.2e}'
                })
                pbar.set_description("MELoRA Training")
                
                # Early stopping check
                if val_metrics['meta_val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['meta_val_loss']
                    self.save_checkpoint(iteration, val_metrics)
                    
            # Checkpoint saving
            if iteration % self.meta_config['checkpoint_frequency'] == 0:
                self.save_checkpoint(iteration)
                
            self.global_step = iteration
        
        pbar.close()
        self.logger.info("Training completed")
        
    def _sample_task_batch(self, tasks: List[Tuple[List, List]]) -> List[Tuple[List, List]]:
        """Sample a batch of tasks for meta-training."""
        indices = np.random.choice(len(tasks), self.meta_batch_size, replace=True)
        return [tasks[i] for i in indices]
    
    def _meta_train_step(self, task_batch: List[Tuple[List, List]]) -> float:
        """Perform one meta-training step."""
        self.model.train()
        
        # Initialize meta-gradients
        meta_grads = None
        total_query_loss = 0.0
        
        # Process tasks with gradient accumulation
        num_micro_batches = len(task_batch) // self.gradient_accumulation_steps
        
        # Add progress bar for task processing (only if batch is large enough)
        if len(task_batch) > 4:
            task_pbar = tqdm(range(len(task_batch)), desc="Processing tasks", 
                           leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}')
        else:
            task_pbar = None
        
        for micro_batch_idx in range(num_micro_batches):
            micro_batch_start = micro_batch_idx * self.gradient_accumulation_steps
            micro_batch_end = min(
                micro_batch_start + self.gradient_accumulation_steps,
                len(task_batch)
            )
            
            micro_batch_grads = None
            micro_batch_loss = 0.0
            
            for task_idx in range(micro_batch_start, micro_batch_end):
                support_set, query_set = task_batch[task_idx]
                
                # Inner loop adaptation
                adapted_params = self._inner_loop_adaptation(support_set)
                
                # Compute query loss with adapted parameters
                query_loss = self._compute_query_loss(query_set, adapted_params)
                
                # Compute meta-gradients
                if self.use_hessian_approx and self.hessian_method != 'none':
                    task_meta_grads = self._compute_second_order_gradients(
                        support_set, query_set, adapted_params
                    )
                else:
                    # First-order approximation (FOMAML)
                    task_meta_grads = torch.autograd.grad(
                        query_loss, 
                        self.model.get_lora_parameters(),
                        retain_graph=True,
                        allow_unused=True
                    )
                    
                    # Handle None gradients for unused parameters
                    task_meta_grads = [g if g is not None else torch.zeros_like(p) 
                                     for g, p in zip(task_meta_grads, self.model.get_lora_parameters())]
                
                # Accumulate gradients
                if micro_batch_grads is None:
                    micro_batch_grads = [g.clone() for g in task_meta_grads]
                else:
                    for i, g in enumerate(task_meta_grads):
                        micro_batch_grads[i] += g
                
                micro_batch_loss += query_loss.item()
                
                # Update task progress bar
                if task_pbar is not None:
                    task_pbar.update(1)
                    task_pbar.set_postfix({'Loss': f'{query_loss.item():.4f}'})
                
            # Average micro-batch gradients
            for g in micro_batch_grads:
                g /= (micro_batch_end - micro_batch_start)
            
            # Accumulate to meta-gradients
            if meta_grads is None:
                meta_grads = micro_batch_grads
            else:
                for i, g in enumerate(micro_batch_grads):
                    meta_grads[i] += g
                    
            total_query_loss += micro_batch_loss
        
        if task_pbar is not None:
            task_pbar.close()
            
        # Average meta-gradients
        for g in meta_grads:
            g /= num_micro_batches
            
        # Apply meta-gradients
        self._apply_meta_gradients(meta_grads)
        
        return total_query_loss / len(task_batch)
    
    def _get_num_classes_from_data(self, data: List[Dict]) -> int:
        """Extract number of classes from task data."""
        if not data:
            return self.model.max_num_labels  # Default fallback
        
        labels = [example['label'] for example in data]
        unique_labels = set(labels)
        num_classes = len(unique_labels)
        
        # Validate labels are in expected range
        min_label, max_label = min(unique_labels), max(unique_labels)
        if min_label < 0 or max_label >= num_classes:
            self.logger.warning(f"Labels not in expected range [0, {num_classes-1}]: found range [{min_label}, {max_label}]")
            # Assume labels are 0-indexed and max_label + 1 is the number of classes
            num_classes = max_label + 1
        
        return num_classes
    
    def _inner_loop_adaptation(self, support_set: List[Dict]) -> List[torch.Tensor]:
        """Perform inner loop adaptation on support set."""
        # Set the number of classes for this task
        num_classes = self._get_num_classes_from_data(support_set)
        self.model.set_num_classes(num_classes)
        
        # Create a copy of current parameters with gradient tracking
        adapted_params = []
        for p in self.model.get_lora_parameters():
            # Clone and ensure requires_grad is True
            cloned_param = p.clone().detach().requires_grad_(True)
            adapted_params.append(cloned_param)
        
        # Create data loader for support set
        support_loader = self.dataset_loader.get_data_loader(
            support_set, 
            batch_size=len(support_set),  # Full batch for few-shot
            shuffle=True
        )
        
        # Inner loop optimization
        for step in range(self.inner_steps):
            for batch in support_loader:
                # Move batch to device
                batch = utils.move_to_device(batch, self.device)
                
                # Forward pass with current adapted parameters
                with torch.enable_grad():
                    # Temporarily replace model parameters
                    original_params = self._replace_parameters(adapted_params)
                    
                    # Compute loss
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    
                    # Compute gradients w.r.t. adapted parameters
                    grads = torch.autograd.grad(
                        loss, adapted_params, create_graph=True, allow_unused=True
                    )
                    
                    # Handle None gradients for unused parameters
                    grads = [g if g is not None else torch.zeros_like(p) 
                            for g, p in zip(grads, adapted_params)]
                    
                    # Restore original parameters
                    self._replace_parameters(original_params)
                
                # Update adapted parameters (ensure they keep requires_grad=True)
                new_adapted_params = []
                for p, g in zip(adapted_params, grads):
                    new_param = (p - self.inner_lr * g).requires_grad_(True)
                    new_adapted_params.append(new_param)
                adapted_params = new_adapted_params
                
        return adapted_params
    
    def _compute_query_loss(self, 
                           query_set: List[Dict],
                           adapted_params: List[torch.Tensor]) -> torch.Tensor:
        """Compute loss on query set with adapted parameters."""
        # Set the number of classes for this task
        num_classes = self._get_num_classes_from_data(query_set)
        self.model.set_num_classes(num_classes)
        
        # Create data loader for query set
        query_loader = self.dataset_loader.get_data_loader(
            query_set,
            batch_size=len(query_set),  # Full batch
            shuffle=False
        )
        
        total_loss = 0.0
        
        for batch in query_loader:
            batch = utils.move_to_device(batch, self.device)
            
            # Temporarily replace model parameters
            original_params = self._replace_parameters(adapted_params)
            
            # Forward pass
            outputs = self.model(**batch)
            total_loss += outputs['loss']
            
            # Restore original parameters
            self._replace_parameters(original_params)
            
        return total_loss
    
    def _compute_second_order_gradients(self,
                                      support_set: List[Dict],
                                      query_set: List[Dict],
                                      adapted_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute second-order meta-gradients."""
        # Compute query loss
        query_loss = self._compute_query_loss(query_set, adapted_params)
        
        # BUG FIX: Compute gradients w.r.t. ORIGINAL parameters, not adapted ones
        query_grads = torch.autograd.grad(
            query_loss, self.model.get_lora_parameters(), retain_graph=True, allow_unused=True
        )
        
        # Handle None gradients for unused parameters
        query_grads = [g if g is not None else torch.zeros_like(p) 
                      for g, p in zip(query_grads, self.model.get_lora_parameters())]
        
        if self.hessian_method == 'diagonal':
            # BUG FIX: Compute Hessian of SUPPORT loss w.r.t. ORIGINAL parameters
            support_loader = self.dataset_loader.get_data_loader(
                support_set, batch_size=len(support_set), shuffle=False
            )
            
            # Temporarily restore original parameters to compute support loss
            current_adapted_params = [p.data.clone() for p in self.model.get_lora_parameters()]
            
            for batch in support_loader:
                batch = utils.move_to_device(batch, self.device)
                # Use current (original) parameters for support loss
                outputs = self.model(**batch)
                support_loss = outputs['loss']
                
            # Compute diagonal Hessian w.r.t. original parameters
            # Increase samples for better estimate and add regularization
            n_samples = max(20, self.memory_config['hessian_approximation']['hutchinson_samples'])
            diag_hessian = self.model.compute_diagonal_hessian(support_loss, n_samples=n_samples)
            
            # Add regularization to prevent numerical issues
            regularization = 1e-6
            diag_hessian = diag_hessian + regularization
            
            # Apply Hessian correction to query gradients  
            # Formula: meta_grad = grad - α * (I + α * H)^(-1) * H * grad
            # Simplified to: meta_grad = grad / (1 + α * H) for diagonal case
            meta_grads = []
            param_idx = 0
            for param, grad in zip(self.model.get_lora_parameters(), query_grads):
                param_size = param.numel()
                param_hessian = diag_hessian[param_idx:param_idx + param_size]
                param_hessian = param_hessian.reshape(param.shape)
                
                # FIXED: Proper Newton-like step (no abs, better regularization)
                # Newton update: grad - α * H^(-1) * grad ≈ grad / (1 + α * H)
                denominator = 1.0 + self.inner_lr * param_hessian
                # Better regularization for numerical stability
                meta_grad = grad / torch.clamp(denominator, min=0.1, max=10.0)
                
                meta_grads.append(meta_grad)
                param_idx += param_size
                
        else:
            # First-order approximation
            meta_grads = query_grads
            
        return meta_grads
    
    def _replace_parameters(self, new_params: List[torch.Tensor]) -> List[torch.Tensor]:
        """Temporarily replace model parameters and return original."""
        original_params = []
        param_idx = 0
        
        # Replace LoRA parameters
        for lora_layer in self.model.lora_layers.values():
            # lora_A
            original_params.append(lora_layer.lora_A.data.clone())
            lora_layer.lora_A.data = new_params[param_idx]
            param_idx += 1
            
            # lora_B
            original_params.append(lora_layer.lora_B.data.clone())
            lora_layer.lora_B.data = new_params[param_idx]
            param_idx += 1
            
        # Replace classifier parameters
        for param in self.model.classifier.parameters():
            original_params.append(param.data.clone())
            param.data = new_params[param_idx]
            param_idx += 1
            
        return original_params
    
    def _apply_meta_gradients(self, meta_grads: List[torch.Tensor]):
        """Apply meta-gradients to model parameters."""
        # Set gradients
        for param, grad in zip(self.model.get_lora_parameters(), meta_grads):
            param.grad = grad
            
        # Clip gradients
        clip_value = self.config['training']['regularization']['gradient_clipping']
        utils.clip_gradients(self.model, clip_value)
        
        # Optimizer step
        self.meta_optimizer.step()
        self.meta_optimizer.zero_grad()
        
    def validate(self, val_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Validate model on validation tasks."""
        self.model.eval()
        
        total_loss = 0.0
        total_accuracy = 0.0
        
        with torch.no_grad():
            for support_set, query_set in tqdm(val_tasks, desc="Validation"):
                # Inner loop adaptation
                adapted_params = self._inner_loop_adaptation(support_set)
                
                # Set the number of classes for the query evaluation
                num_classes = self._get_num_classes_from_data(query_set)
                self.model.set_num_classes(num_classes)
                
                # Evaluate on query set
                query_loader = self.dataset_loader.get_data_loader(
                    query_set, batch_size=len(query_set), shuffle=False
                )
                
                for batch in query_loader:
                    batch = utils.move_to_device(batch, self.device)
                    
                    # Temporarily use adapted parameters
                    original_params = self._replace_parameters(adapted_params)
                    
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                    logits = outputs['logits']
                    
                    # Restore parameters
                    self._replace_parameters(original_params)
                    
                    # Compute metrics
                    total_loss += loss.item()
                    predictions = torch.argmax(logits, dim=1)
                    accuracy = (predictions == batch['labels']).float().mean()
                    total_accuracy += accuracy.item()
                    
        avg_loss = total_loss / len(val_tasks)
        avg_accuracy = total_accuracy / len(val_tasks)
        
        return {
            'meta_val_loss': avg_loss,
            'meta_val_accuracy': avg_accuracy
        }
    
    def save_checkpoint(self, iteration: int, metrics: Optional[Dict] = None):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.config['experiment']['checkpoint_dir'],
            f'checkpoint_iter_{iteration}.pt'
        )
        
        self.model.save_checkpoint(
            checkpoint_path,
            optimizer=self.meta_optimizer,
            epoch=iteration,
            metrics=metrics
        )
        
        # Save memory profile
        memory_profile_path = os.path.join(
            self.config['experiment']['output_dir'],
            f'memory_profile_iter_{iteration}.json'
        )
        self.memory_profiler.save_profile(memory_profile_path)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = self.model.load_checkpoint(
            checkpoint_path,
            optimizer=self.meta_optimizer
        )
        
        self.global_step = checkpoint.get('epoch', 0)
        self.logger.info(f"Loaded checkpoint from iteration {self.global_step}") 