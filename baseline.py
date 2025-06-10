"""
Baseline implementations for comparative evaluation.
Includes MAML, FOMAML, Reptile, and standard fine-tuning baselines.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import torch.func as func

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

    def _pad_and_collate(self, tensor_list: List[Dict], max_size: int) -> Dict[str, torch.Tensor]:
        """Pads tensors in a list to a max size and stacks them."""
        padded_tensors = []
        # Use a default pad_token_id if tokenizer is not directly on dataset_loader
        pad_token_id = getattr(self.dataset_loader.tokenizer, 'pad_token_id', 0)
        
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
        
        has_query = task_list and len(task_list[0]) > 1 and task_list[0][1] is not None
        
        for item in task_list:
            support_set = item[0]
            query_set = item[1] if has_query else None

            support_loader = self.dataset_loader.get_data_loader(
                support_set, batch_size=len(support_set), shuffle=False)
            support_tensors.append(next(iter(support_loader)))

            if query_set:
                query_loader = self.dataset_loader.get_data_loader(
                    query_set, batch_size=len(query_set), shuffle=False)
                query_tensors.append(next(iter(query_loader)))

        max_support_size = max(s['input_ids'].shape[0] for s in support_tensors)
        collated_support = self._pad_and_collate(support_tensors, max_support_size)

        collated_query = None
        if query_tensors:
            max_query_size = max(q['input_ids'].shape[0] for q in query_tensors)
            collated_query = self._pad_and_collate(query_tensors, max_query_size)
        
        return collated_support, collated_query


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
        """Train with full MAML using a parallelized vmap approach."""
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Parallelized FullMAML training (vmap)")
        
        pbar = tqdm(range(num_iterations), desc="FullMAML Training (vmap)")
        for iteration in pbar:
            task_indices = np.random.choice(len(meta_train_tasks), self.meta_batch_size, replace=False)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            meta_loss = self._meta_train_step_parallel(task_batch)
            pbar.set_postfix({'loss': f'{meta_loss:.4f}'})

            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}")
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")

    def _meta_train_step_parallel(self, task_batch: List[Tuple[List, List]]) -> float:
        """A single parallel meta-update step for FullMAML using vmap."""
        self.meta_optimizer.zero_grad()
        
        collated_support, collated_query = self._get_task_tensors(task_batch)
        params = {name: p for name, p in self.model.named_parameters()}
        buffers = {name: b for name, b in self.model.named_buffers()}

        def full_maml_single_task_loss(params, buffers, support_batch, query_batch):
            fast_params = params
            for _ in range(self.inner_steps):
                support_batch_device = utils.move_to_device(support_batch, self.device)
                support_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=support_batch_device)
                grads = torch.autograd.grad(support_outputs['loss'], list(fast_params.values()), create_graph=True, allow_unused=True)
                fast_params = {
                    name: p - self.inner_lr * g if g is not None else p
                    for (name, p), g in zip(fast_params.items(), grads)
                }
            
            query_batch_device = utils.move_to_device(query_batch, self.device)
            query_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=query_batch_device)
            return query_outputs['loss']

        in_dims = (None, None, 0, 0)
        query_losses = func.vmap(
            full_maml_single_task_loss, in_dims=in_dims, randomness='different'
        )(
            params, buffers, collated_support, collated_query
        )

        meta_loss = torch.mean(query_losses)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()
        return meta_loss.item()
    
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
        """Evaluate on test tasks using same protocol as MELoRA."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for fair comparison
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(test_tasks)
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0)
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
        """Train with FOMAML using a parallelized vmap approach."""
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Parallelized FOMAML training (vmap)")
        
        pbar = tqdm(range(num_iterations), desc="FOMAML Training (vmap)")
        for iteration in pbar:
            # Sample a batch of tasks
            task_indices = np.random.choice(len(meta_train_tasks), 
                                          self.meta_batch_size, replace=False)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            # Perform one parallel meta-training step
            meta_loss = self._meta_train_step_parallel(task_batch)
            
            pbar.set_postfix({'loss': f'{meta_loss:.4f}'})
            
            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Meta Loss = {meta_loss:.4f}")
                
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")
    
    def _meta_train_step_parallel(self, task_batch: List[Tuple[List, List]]) -> float:
        """A single parallel meta-update step using vmap."""
        self.meta_optimizer.zero_grad()

        # Collate list of tasks into batched tensors
        collated_support, collated_query = self._get_task_tensors(task_batch)

        # Get functional representations of the model's parameters and buffers
        params = {name: p for name, p in self.model.named_parameters()}
        buffers = {name: b for name, b in self.model.named_buffers()}

        # Define the function that processes a single task
        def fomaml_single_task_loss(params, buffers, support_batch, query_batch):
            # Inner loop adaptation
            fast_params = params
            for _ in range(self.inner_steps):
                # Ensure all tensors in the batch are on the correct device
                support_batch_device = utils.move_to_device(support_batch, self.device)
                
                # Compute loss on support set
                support_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=support_batch_device)
                support_loss = support_outputs['loss']
                
                # Compute gradients for the inner loop
                grads = torch.autograd.grad(support_loss, fast_params.values(), allow_unused=True)
                
                # Update parameters
                fast_params = {
                    name: p - self.inner_lr * g if g is not None else p
                    for (name, p), g in zip(fast_params.items(), grads)
                }

            # Evaluate on the query set with the adapted parameters
            query_batch_device = utils.move_to_device(query_batch, self.device)
            query_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=query_batch_device)
            return query_outputs['loss']

        # Vectorize the single-task function over the meta-batch dimension (dim 0)
        # We broadcast the params and buffers to each task (in_dims=None)
        in_dims = (None, None, 0, 0)
        query_losses = func.vmap(
            fomaml_single_task_loss, in_dims=in_dims, randomness='different'
        )(
            params, buffers, collated_support, collated_query
        )

        # Compute the final meta-loss and update the model
        meta_loss = torch.mean(query_losses)
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.meta_optimizer.step()

        return meta_loss.item()
    
    def evaluate(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Evaluate on test tasks."""
        return self._evaluate_common(test_tasks)
    
    def _evaluate_common(self, test_tasks: List[Tuple[List, List]]) -> Dict[str, float]:
        """Common evaluation logic using same protocol as MELoRA."""
        from evaluation import Evaluator
        
        # Use MELoRA's evaluation protocol for fair comparison
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        results = evaluator.evaluate_meta_learning(test_tasks)
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0)
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
        """Train with Reptile using a parallelized vmap approach."""
        num_iterations = self.config['meta_learning']['num_meta_iterations']
        
        self.logger.info("Starting Parallelized Reptile training (vmap)")
        
        pbar = tqdm(range(num_iterations), desc="Reptile Training (vmap)")
        for iteration in pbar:
            task_indices = np.random.choice(len(meta_train_tasks), self.meta_batch_size, replace=False)
            task_batch = [meta_train_tasks[i] for i in task_indices]
            
            avg_loss = self._meta_train_step_parallel(task_batch)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})

            if iteration % self.config['training']['log_interval'] == 0:
                self.logger.info(f"Iteration {iteration}: Avg Inner-Loop Loss = {avg_loss:.4f}")
            if meta_val_tasks and iteration % 100 == 0:
                val_metrics = self.evaluate(meta_val_tasks)
                self.logger.info(f"Validation: {val_metrics}")

    def _meta_train_step_parallel(self, task_batch: List[Tuple[List, List]]) -> float:
        """A single parallel meta-update step for Reptile using vmap."""
        collated_support, _ = self._get_task_tensors(task_batch)
        params = {name: p for name, p in self.model.named_parameters()}
        buffers = {name: b for name, b in self.model.named_buffers()}

        def reptile_single_task_adapt(params, buffers, support_batch):
            fast_params = params
            total_loss = 0.0
            for _ in range(self.inner_steps):
                support_batch_device = utils.move_to_device(support_batch, self.device)
                support_outputs = func.functional_call(self.model, (fast_params, buffers), args=(), kwargs=support_batch_device)
                loss = support_outputs['loss']
                grads = torch.autograd.grad(loss, list(fast_params.values()), allow_unused=True)
                fast_params = {
                    name: p - self.lr * g if g is not None else p
                    for (name, p), g in zip(fast_params.items(), grads)
                }
                total_loss += loss
            # vmap requires that all returned tensors have a batch dimension.
            # We return the total loss and the final parameters for this task.
            return total_loss / self.inner_steps, fast_params

        in_dims = (None, None, 0)
        # vmap returns adapted_params for each task, stacked along dim 0
        avg_losses, adapted_params_batch = func.vmap(
            reptile_single_task_adapt, in_dims=in_dims, randomness='different'
        )(
            params, buffers, collated_support
        )
        
        # Average the adapted parameters across the meta-batch
        avg_adapted_params = {
            name: torch.mean(p_batch, dim=0)
            for name, p_batch in adapted_params_batch.items()
        }

        # Perform the Reptile update
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    p.data = p.data + self.epsilon * (avg_adapted_params[name] - p.data)

        return torch.mean(avg_losses).item()

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
        
        # Extract the metrics that baselines expect
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0)
        }


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
        """Evaluate by fine-tuning from scratch using the unified Evaluator."""
        from evaluation import Evaluator
        
        # This evaluator uses the model passed to StandardFineTuning, which main.py
        # now ensures is a fresh, pre-trained model instance.
        evaluator = Evaluator(self.model, self.config, self.dataset_loader)
        
        # Call the evaluator with reset_model_per_task=True to ensure
        # that for each task, we start from the initial pre-trained weights.
        # Pass the specific epochs and lr for fine-tuning as adaptation parameters.
        results = evaluator.evaluate_meta_learning(
            test_tasks, 
            adaptation_steps=self.epochs,
            adaptation_lr=self.lr,
            reset_model_per_task=True
        )
        
        # Extract and return the primary metrics for comparison
        return {
            'test_loss': results.get('loss_mean', 0.0),
            'test_accuracy': results.get('accuracy_mean', 0.0)
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