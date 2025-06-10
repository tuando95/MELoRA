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
from torch.utils.checkpoint import checkpoint
import torch.func as func

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
        
    def _pad_and_collate(self, tensor_list: List[Dict], max_size: int) -> Dict[str, torch.Tensor]:
        """Pads tensors in a list to a max size and stacks them."""
        padded_tensors = []
        pad_token_id = self.dataset_loader.tokenizer.pad_token_id
        for tensors in tensor_list:
            padded_dict = {}
            current_size = tensors['input_ids'].shape[0]
            pad_size = max_size - current_size

            if pad_size > 0:
                padded_dict['input_ids'] = F.pad(
                    tensors['input_ids'], (0, 0, 0, pad_size), value=pad_token_id)
                padded_dict['attention_mask'] = F.pad(
                    tensors['attention_mask'], (0, 0, 0, pad_size), value=0)
                padded_dict['labels'] = F.pad(
                    tensors['labels'], (0, pad_size), value=-100)
            else:
                padded_dict = tensors
            
            padded_tensors.append(padded_dict)

        return {k: torch.stack([s[k] for s in padded_tensors]) for k in padded_tensors[0]}

    def _get_task_tensors(self, task_list):
        """Collates and pads tasks into batched tensors for vmap."""
        support_tensors, query_tensors = [], []
        for support_set, query_set in task_list:
            support_loader = self.dataset_loader.get_data_loader(
                support_set, batch_size=len(support_set), shuffle=False)
            query_loader = self.dataset_loader.get_data_loader(
                query_set, batch_size=len(query_set), shuffle=False)
            support_tensors.append(next(iter(support_loader)))
            query_tensors.append(next(iter(query_loader)))

        max_support_size = max(s['input_ids'].shape[0] for s in support_tensors)
        max_query_size = max(q['input_ids'].shape[0] for q in query_tensors)

        collated_support = self._pad_and_collate(support_tensors, max_support_size)
        collated_query = self._pad_and_collate(query_tensors, max_query_size)
        
        return collated_support, collated_query

    def train(self, meta_train_tasks: List, meta_val_tasks: Optional[List] = None, num_iterations: Optional[int] = None):
        """Train the MELoRA model using a parallelized vmap approach."""
        num_iterations = num_iterations or self.meta_config['num_meta_iterations']
        
        self.logger.info("Starting Parallelized MELoRA training (vmap)")
        pbar = tqdm(range(num_iterations), desc="MELoRA Training (vmap)")

        for iteration in pbar:
            self.model.train()
            task_indices = np.random.choice(len(meta_train_tasks), self.meta_batch_size, replace=False)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Perform one parallel meta-training step
            meta_loss, query_loss, lora_reg_loss = self._meta_train_step_parallel(task_batch)
            
            pbar.set_postfix({
                'meta_loss': f'{meta_loss:.4f}',
                'query_loss': f'{query_loss:.4f}',
                'reg_loss': f'{lora_reg_loss:.4f}'
            })

            if iteration % self.config['training']['log_interval'] == 0:
                self._log_metrics(iteration, meta_loss, query_loss, lora_reg_loss)
            
            if meta_val_tasks and iteration % self.config['training']['eval_interval'] == 0:
                self._run_validation(meta_val_tasks, iteration)
            
            if iteration % self.config['training']['save_interval'] == 0 and iteration > 0:
                self.save_checkpoint(f"iter_{iteration}.pt")
    
    def _meta_train_step_parallel(self, task_batch: List) -> Tuple[float, float, float]:
        """A single parallel meta-update step using vmap."""
        self.meta_optimizer.zero_grad()

        collated_support, collated_query = self._get_task_tensors(task_batch)
        params = {name: p for name, p in self.model.named_parameters() if p.requires_grad}
        buffers = {name: b for name, b in self.model.named_buffers()}

        def melora_single_task_loss(params, buffers, support_batch, query_batch):
            # Inner loop adaptation
            fast_params = params
            for _ in range(self.inner_steps):
                support_batch_device = utils.move_to_device(support_batch, self.device)
                support_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=support_batch_device)
                
                inner_loss = support_outputs['loss']
                if self.lora_config['regularization_weight'] > 0:
                    lora_params = [p for name, p in fast_params.items() if 'lora' in name]
                    reg_loss = self._lora_regularization(lora_params)
                    inner_loss += self.lora_config['regularization_weight'] * reg_loss

                grads = torch.autograd.grad(inner_loss, list(fast_params.values()), allow_unused=True)
                fast_params = {
                    name: p - self.inner_lr * g if g is not None else p
                    for (name, p), g in zip(fast_params.items(), grads)
                }
            
            # Evaluate on query set
            query_batch_device = utils.move_to_device(query_batch, self.device)
            query_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=query_batch_device)
            task_query_loss = query_outputs['loss']

            task_lora_reg_loss = torch.tensor(0.0, device=self.device)
            if self.lora_config['regularization_weight'] > 0:
                lora_params = [p for name, p in fast_params.items() if 'lora' in name]
                task_lora_reg_loss = self._lora_regularization(lora_params)
            
            return task_query_loss, task_lora_reg_loss

        # Vectorize over the meta-batch dimension
        in_dims = (None, None, 0, 0)
        query_losses, lora_reg_losses = func.vmap(melora_single_task_loss, in_dims=in_dims)(
            params, buffers, collated_support, collated_query
        )

        # Aggregate losses and perform the meta-update
        query_loss = torch.mean(query_losses)
        lora_reg_loss = torch.mean(lora_reg_losses)
        meta_loss = query_loss + self.lora_config['regularization_weight'] * lora_reg_loss

        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip_norm'])
        self.meta_optimizer.step()

        return meta_loss.item(), query_loss.item(), lora_reg_loss.item()
    
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
        
        # Compute first-order gradients
        query_grads = torch.autograd.grad(
            query_loss, adapted_params, retain_graph=True, allow_unused=True
        )
        
        # Handle None gradients for unused parameters
        query_grads = [g if g is not None else torch.zeros_like(p) 
                      for g, p in zip(query_grads, adapted_params)]
        
        if self.hessian_method == 'diagonal':
            # Diagonal Hessian approximation
            support_loader = self.dataset_loader.get_data_loader(
                support_set, batch_size=len(support_set), shuffle=False
            )
            
            for batch in support_loader:
                batch = utils.move_to_device(batch, self.device)
                outputs = self.model(**batch)
                support_loss = outputs['loss']
                
            # Compute diagonal Hessian
            diag_hessian = self.model.compute_diagonal_hessian(
                support_loss,
                n_samples=self.memory_config['hessian_approximation']['hutchinson_samples']
            )
            
            # Apply Hessian to query gradients
            meta_grads = []
            param_idx = 0
            for param, grad in zip(self.model.get_lora_parameters(), query_grads):
                param_size = param.numel()
                param_hessian = diag_hessian[param_idx:param_idx + param_size]
                param_hessian = param_hessian.reshape(param.shape)
                
                # Meta-gradient = I - α * H
                meta_grad = grad - self.inner_lr * param_hessian * grad
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