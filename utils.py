"""
Utility functions for MELoRA experiments.
Includes reproducibility, logging, memory profiling, and metrics utilities.
"""

import os
import json
import time
import random
import logging
import subprocess
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
import psutil
import mlflow
from torch.utils.tensorboard import SummaryWriter


# Global variables for logging and tracking
_tensorboard_writer = None
_mlflow_run = None
_logger = None


def load_config(config_path: str = 'config.yml') -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int, config: Optional[Dict] = None):
    """Set random seeds for reproducibility."""
    if config is None:
        config = load_config()
    
    # Set all random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior if configured
    if config['reproducibility']['cuda_deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def setup_logging(config: Dict) -> logging.Logger:
    """Set up logging configuration."""
    global _logger
    
    log_config = config['logging']
    
    # Create logger
    logger = logging.getLogger('melora')
    logger.setLevel(getattr(logging, log_config['level']))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(log_config['format'])
    
    # Console handler
    if log_config['console']['enabled']:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_config['file']['enabled']:
        os.makedirs(os.path.dirname(log_config['file']['path']), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_config['file']['path'],
            maxBytes=log_config['file']['max_size_mb'] * 1024 * 1024,
            backupCount=log_config['file']['backup_count']
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger() -> logging.Logger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        config = load_config()
        _logger = setup_logging(config)
    return _logger


def setup_tensorboard(config: Dict) -> Optional[SummaryWriter]:
    """Set up TensorBoard writer."""
    global _tensorboard_writer
    
    if config['logging']['tensorboard']['enabled']:
        log_dir = os.path.join(
            config['logging']['tensorboard']['log_dir'],
            datetime.now().strftime('%Y%m%d_%H%M%S')
        )
        _tensorboard_writer = SummaryWriter(log_dir)
        get_logger().info(f"TensorBoard logging to: {log_dir}")
    
    return _tensorboard_writer


def setup_mlflow(config: Dict) -> Optional[Any]:
    """Set up MLflow tracking."""
    global _mlflow_run
    
    if config['experiment']['use_mlflow']:
        mlflow.set_tracking_uri(config['experiment']['mlflow_uri'])
        mlflow.set_experiment(config['experiment']['name'])
        
        # Start MLflow run
        _mlflow_run = mlflow.start_run()
        
        # Log configuration
        mlflow.log_params({
            'model': config['models']['selected_model'],
            'lora_rank': config['lora']['default_rank'],
            'inner_lr': config['meta_learning']['inner']['default_lr'],
            'outer_lr': config['meta_learning']['outer']['default_lr'],
            'meta_batch_size': config['meta_learning']['default_meta_batch_size']
        })
        
        get_logger().info(f"MLflow run started: {_mlflow_run.info.run_id}")
    
    return _mlflow_run


def log_metrics(metrics: Dict[str, float], 
                step: int,
                prefix: str = '') -> None:
    """Log metrics to all configured backends."""
    global _tensorboard_writer, _mlflow_run
    
    # Add prefix if provided
    if prefix:
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
    
    # Log to console
    logger = get_logger()
    logger.info(f"Step {step}: {metrics}")
    
    # Log to TensorBoard
    if _tensorboard_writer is not None:
        for key, value in metrics.items():
            # Only log numeric values to TensorBoard
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _tensorboard_writer.add_scalar(key, value, step)
            else:
                # Skip non-numeric values but log a warning
                logger = get_logger()
                logger.debug(f"Skipping non-numeric metric for TensorBoard: {key}={value}")
    
    # Log to MLflow
    if _mlflow_run is not None:
        # Filter numeric metrics for MLflow as well
        numeric_metrics = {k: v for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and not isinstance(v, bool)}
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics, step)


def save_results(results: Dict[str, Any], 
                filepath: str,
                format: str = 'json') -> None:
    """Save experiment results to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if format == 'json':
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == 'yaml':
        with open(filepath, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    get_logger().info(f"Results saved to: {filepath}")


def load_results(filepath: str) -> Dict[str, Any]:
    """Load experiment results from file."""
    ext = os.path.splitext(filepath)[1]
    
    if ext == '.json':
        with open(filepath, 'r') as f:
            results = json.load(f)
    elif ext in ['.yml', '.yaml']:
        with open(filepath, 'r') as f:
            results = yaml.safe_load(f)
    else:
        raise ValueError(f"Unknown file extension: {ext}")
    
    return results


# Memory profiling utilities
class MemoryProfiler:
    """Utility class for memory profiling."""
    
    def __init__(self, config: Dict):
        self.config = config['evaluation']['memory_profiling']
        self.enabled = self.config['enabled']
        self.use_nvidia_smi = self.config['use_nvidia_smi']
        self.use_torch_profiler = self.config['use_torch_profiler']
        self.measurements = []
        
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        
        # PyTorch memory stats
        if self.use_torch_profiler:
            memory_stats['allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            memory_stats['reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            memory_stats['max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024**2
        
        # NVIDIA-SMI stats
        if self.use_nvidia_smi:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used,memory.total', 
                     '--format=csv,nounits,noheader'],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    used, total = map(int, result.stdout.strip().split(','))
                    memory_stats['nvidia_smi_used_mb'] = used
                    memory_stats['nvidia_smi_total_mb'] = total
                    memory_stats['nvidia_smi_free_mb'] = total - used
            except:
                pass
        
        return memory_stats
    
    def get_cpu_memory_usage(self) -> Dict[str, float]:
        """Get current CPU memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'cpu_rss_mb': memory_info.rss / 1024**2,
            'cpu_vms_mb': memory_info.vms / 1024**2,
            'cpu_percent': process.memory_percent()
        }
    
    def profile_memory(self, tag: str = '') -> Dict[str, float]:
        """Profile current memory usage."""
        if not self.enabled:
            return {}
        
        memory_stats = {
            'timestamp': time.time(),
            'tag': tag
        }
        
        # GPU memory
        gpu_stats = self.get_gpu_memory_usage()
        memory_stats.update(gpu_stats)
        
        # CPU memory
        cpu_stats = self.get_cpu_memory_usage()
        memory_stats.update(cpu_stats)
        
        self.measurements.append(memory_stats)
        
        return memory_stats
    
    def get_memory_breakdown(self, model: torch.nn.Module) -> Dict[str, float]:
        """Get detailed memory breakdown by component."""
        if not self.config['detailed_breakdown']:
            return {}
        
        breakdown = {}
        
        # Model parameters
        param_memory = 0
        for param in model.parameters():
            param_memory += param.numel() * param.element_size()
        breakdown['parameters_mb'] = param_memory / 1024**2
        
        # Gradients
        grad_memory = 0
        for param in model.parameters():
            if param.grad is not None:
                grad_memory += param.grad.numel() * param.grad.element_size()
        breakdown['gradients_mb'] = grad_memory / 1024**2
        
        # LoRA parameters specifically
        lora_param_memory = 0
        lora_grad_memory = 0
        for name, param in model.named_parameters():
            if 'lora' in name.lower():
                lora_param_memory += param.numel() * param.element_size()
                if param.grad is not None:
                    lora_grad_memory += param.grad.numel() * param.grad.element_size()
        breakdown['lora_parameters_mb'] = lora_param_memory / 1024**2
        breakdown['lora_gradients_mb'] = lora_grad_memory / 1024**2
        
        return breakdown
    
    def save_profile(self, filepath: str):
        """Save memory profile to file."""
        save_results({'measurements': self.measurements}, filepath)
    
    def reset(self):
        """Reset memory measurements."""
        self.measurements = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


# Model checkpoint utilities
def save_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   metrics: Dict[str, float],
                   config: Dict,
                   filepath: str) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, filepath)
    get_logger().info(f"Checkpoint saved to: {filepath}")
    
    # Log to MLflow
    if _mlflow_run is not None:
        mlflow.log_artifact(filepath)


def load_checkpoint(filepath: str,
                   model: torch.nn.Module,
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    get_logger().info(f"Checkpoint loaded from: {filepath}")
    
    return checkpoint


# Gradient utilities
def compute_gradient_norm(model: torch.nn.Module,
                         norm_type: float = 2.0) -> float:
    """Compute gradient norm for model parameters."""
    total_norm = 0.0
    
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    total_norm = total_norm ** (1.0 / norm_type)
    
    return total_norm


def clip_gradients(model: torch.nn.Module,
                  max_norm: float,
                  norm_type: float = 2.0) -> float:
    """Clip gradients by norm."""
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm, norm_type
    )


# Device utilities
def get_device(config: Dict) -> torch.device:
    """Get compute device based on configuration."""
    if config['experiment']['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        get_logger().info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        get_logger().info("Using CPU")
    
    return device


def move_to_device(data: Union[torch.Tensor, Dict, List],
                  device: torch.device) -> Union[torch.Tensor, Dict, List]:
    """Recursively move data to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


# Time utilities
class Timer:
    """Simple timer utility."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
        self.lap_time = self.start_time
    
    def lap(self) -> float:
        current_time = time.time()
        lap_duration = current_time - self.lap_time
        self.lap_time = current_time
        return lap_duration
    
    def total(self) -> float:
        return time.time() - self.start_time


# Cleanup utilities
def cleanup():
    """Clean up resources."""
    global _tensorboard_writer, _mlflow_run
    
    if _tensorboard_writer is not None:
        _tensorboard_writer.close()
        _tensorboard_writer = None
    
    if _mlflow_run is not None:
        mlflow.end_run()
        _mlflow_run = None 