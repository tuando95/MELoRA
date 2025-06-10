#!/usr/bin/env python3
"""
Simple test script for MELoRA training with reduced complexity.
"""

import os
import sys
import time
import torch

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_loader import DatasetLoader
from model import MELoRAModel
from trainer import MELoRATrainer
import utils

def test_melora_training():
    """Test MELoRA training with minimal configuration."""
    print("Starting MELoRA training test...")
    
    # Load configuration
    config = utils.load_config('config.yml')
    
    # Override config for faster testing
    config['meta_learning']['num_meta_iterations'] = 5
    config['meta_learning']['default_meta_batch_size'] = 2
    config['meta_learning']['inner']['default_num_steps'] = 1
    config['datasets']['few_shot']['num_tasks_train'] = 10
    config['datasets']['few_shot']['num_tasks_val'] = 5
    config['datasets']['few_shot']['default_k_shot'] = 3
    config['datasets']['few_shot']['query_set_size'] = 10
    config['memory_optimization']['hessian_approximation']['enabled'] = False
    config['memory_optimization']['checkpointing']['enabled'] = False
    
    print("Configuration overrides applied for testing")
    
    # Set up logging
    logger = utils.setup_logging(config)
    utils.set_seed(config['experiment']['seed'], config)
    
    # Initialize dataset loader
    print("Loading datasets...")
    start_time = time.time()
    dataset_loader = DatasetLoader(config_path='config.yml')
    print(f"Dataset loading took {time.time() - start_time:.2f} seconds")
    
    # Create meta-datasets
    print("Creating meta-datasets...")
    start_time = time.time()
    train_tasks = dataset_loader.create_meta_dataset(split='train')
    val_tasks = dataset_loader.create_meta_dataset(split='val')
    print(f"Meta-dataset creation took {time.time() - start_time:.2f} seconds")
    print(f"Train tasks: {len(train_tasks)}, Val tasks: {len(val_tasks)}")
    
    # Initialize model
    print("Creating model...")
    start_time = time.time()
    model = MELoRAModel(config)
    print(f"Model creation took {time.time() - start_time:.2f} seconds")
    
    # Log model info
    memory_usage = model.get_memory_usage()
    print(f"Model memory usage: {memory_usage}")
    
    # Initialize trainer
    print("Creating trainer...")
    trainer = MELoRATrainer(model, config, dataset_loader)
    
    # Start training
    print("Starting training...")
    start_time = time.time()
    
    try:
        trainer.train(
            meta_train_tasks=train_tasks,
            meta_val_tasks=val_tasks,
            num_iterations=5
        )
        print(f"Training completed successfully in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = test_melora_training()
    if success:
        print("✅ MELoRA training test passed!")
    else:
        print("❌ MELoRA training test failed!")
        sys.exit(1) 