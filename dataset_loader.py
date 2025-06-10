"""
Dataset loader for MELoRA experiments.
Handles loading and preprocessing of GLUE, SuperGLUE, and KILT datasets.
"""

import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
import yaml
from tqdm import tqdm

# Try to import utils, fallback to direct yaml loading if not available
try:
    import utils
except ImportError:
    utils = None


class FewShotDataset(Dataset):
    """Dataset wrapper for few-shot learning tasks."""
    
    def __init__(self, examples: List[Dict], tokenizer, max_length: int = 512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Ensure tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        inputs = self.tokenizer(
            example['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': torch.tensor(example['label'], dtype=torch.long)
        }


class DatasetLoader:
    """DatasetLoader for handling GLUE, SuperGLUE, and KILT datasets."""
    
    def __init__(self, config_path: str = 'config.yml'):
        # Load config
        if utils:
            self.config = utils.load_config(config_path)
        else:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
        self.datasets_config = self.config['datasets']
        self.few_shot_config = self.datasets_config['few_shot']
        
        # Initialize tokenizer
        model_name = self.config['models']['selected_model']
        model_info = next(m for m in self.config['models']['available_models'] 
                         if m['name'] == model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_info['pretrained'])
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Cache for loaded datasets to avoid repeated downloads
        self.dataset_cache = {}
        self._preload_datasets()
    
    def _preload_datasets(self):
        """Pre-load all enabled datasets to avoid repeated downloads."""
        print("Pre-loading datasets...")
        
        # Pre-load GLUE datasets
        if self.datasets_config['glue']['enabled']:
            for task_config in self.datasets_config['glue']['tasks']:
                task_name = task_config['name']
                print(f"Loading GLUE task: {task_name}")
                try:
                    self.dataset_cache[('glue', task_name)] = self._load_glue_task(task_name)
                except Exception as e:
                    print(f"Warning: Failed to load GLUE task {task_name}: {e}")
        
        # Pre-load SuperGLUE datasets
        if self.datasets_config['superglue']['enabled']:
            for task_config in self.datasets_config['superglue']['tasks']:
                task_name = task_config['name']
                print(f"Loading SuperGLUE task: {task_name}")
                try:
                    self.dataset_cache[('superglue', task_name)] = self._load_superglue_task(task_name)
                except Exception as e:
                    print(f"Warning: Failed to load SuperGLUE task {task_name}: {e}")
                    
        print(f"Finished pre-loading {len(self.dataset_cache)} datasets")
    
    def load_dataset_splits(self, dataset_name: str, task_name: str) -> Dict[str, List]:
        """Load train/val/test splits for a specific dataset and task."""
        cache_key = (dataset_name, task_name)
        
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
        
        # If not in cache, try to load it (fallback)
        print(f"Dataset {cache_key} not in cache, loading on demand...")
        
        if dataset_name == 'glue':
            data = self._load_glue_task(task_name)
        elif dataset_name == 'superglue':
            data = self._load_superglue_task(task_name)
        elif dataset_name == 'kilt':
            data = self._load_kilt_task(task_name)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        self.dataset_cache[cache_key] = data
        return data
    
    def _load_glue_task(self, task_name: str) -> Dict[str, List]:
        """Load GLUE task data."""
        # Map task names to HuggingFace dataset names
        task_mapping = {
            'sst2': 'sst2',
            'mnli': 'mnli',
            'qqp': 'qqp',
            'rte': 'rte'
        }
        
        dataset = load_dataset('glue', task_mapping[task_name])
        
        # Process examples based on task type
        if task_name == 'sst2':
            processor = self._process_sst2
        elif task_name == 'mnli':
            processor = self._process_mnli
        elif task_name == 'qqp':
            processor = self._process_qqp
        elif task_name == 'rte':
            processor = self._process_rte
        else:
            raise ValueError(f"Unknown GLUE task: {task_name}")
        
        # Handle different validation split names
        validation_data = []
        if 'validation' in dataset:
            validation_data = [processor(ex) for ex in dataset['validation']]
        elif 'validation_matched' in dataset:
            # For MNLI, use validation_matched as the main validation set
            validation_data = [processor(ex) for ex in dataset['validation_matched']]
        elif 'dev' in dataset:
            # Some datasets use 'dev' instead of 'validation'
            validation_data = [processor(ex) for ex in dataset['dev']]
        else:
            # If no validation split, split train data
            print(f"Warning: No validation split found for {task_name}, splitting train data")
            train_examples = [processor(ex) for ex in dataset['train']]
            # Use last 20% of train for validation
            split_idx = int(0.8 * len(train_examples))
            validation_data = train_examples[split_idx:]
            train_examples = train_examples[:split_idx]
            
            return {
                'train': train_examples,
                'validation': validation_data,
                'test': [processor(ex) for ex in dataset.get('test', [])]
            }
        
        return {
            'train': [processor(ex) for ex in dataset['train']],
            'validation': validation_data,
            'test': [processor(ex) for ex in dataset.get('test', [])]
        }
    
    def _load_superglue_task(self, task_name: str) -> Dict[str, List]:
        """Load SuperGLUE task data."""
        dataset = load_dataset('super_glue', task_name)
        
        if task_name == 'boolq':
            processor = self._process_boolq
        elif task_name == 'cb':
            processor = self._process_cb
        elif task_name == 'wic':
            processor = self._process_wic
        else:
            raise ValueError(f"Unknown SuperGLUE task: {task_name}")
        
        # Handle different validation split names
        validation_data = []
        if 'validation' in dataset:
            validation_data = [processor(ex) for ex in dataset['validation']]
        elif 'dev' in dataset:
            validation_data = [processor(ex) for ex in dataset['dev']]
        else:
            # If no validation split, split train data
            print(f"Warning: No validation split found for {task_name}, splitting train data")
            train_examples = [processor(ex) for ex in dataset['train']]
            # Use last 20% of train for validation
            split_idx = int(0.8 * len(train_examples))
            validation_data = train_examples[split_idx:]
            train_examples = train_examples[:split_idx]
            
            return {
                'train': train_examples,
                'validation': validation_data,
                'test': [processor(ex) for ex in dataset.get('test', [])]
            }
        
        return {
            'train': [processor(ex) for ex in dataset['train']],
            'validation': validation_data,
            'test': [processor(ex) for ex in dataset.get('test', [])]
        }
    
    def _load_kilt_task(self, task_name: str) -> Dict[str, List]:
        """Load KILT task data."""
        # KILT requires special handling
        # For now, return empty data
        print(f"Warning: KILT dataset {task_name} not implemented yet")
        return {'train': [], 'validation': [], 'test': []}
    
    # Task-specific processors
    def _process_sst2(self, example):
        return {
            'text': example['sentence'],
            'label': example['label'],
            'task': 'sst2'
        }
    
    def _process_mnli(self, example):
        return {
            'text': f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}",
            'label': example['label'],
            'task': 'mnli'
        }
    
    def _process_qqp(self, example):
        return {
            'text': f"Question 1: {example['question1']} Question 2: {example['question2']}",
            'label': example['label'],
            'task': 'qqp'
        }
    
    def _process_rte(self, example):
        return {
            'text': f"Premise: {example['sentence1']} Hypothesis: {example['sentence2']}",
            'label': example['label'],
            'task': 'rte'
        }
    
    def _process_boolq(self, example):
        return {
            'text': f"Passage: {example['passage']} Question: {example['question']}",
            'label': int(example['label']),
            'task': 'boolq'
        }
    
    def _process_cb(self, example):
        return {
            'text': f"Premise: {example['premise']} Hypothesis: {example['hypothesis']}",
            'label': example['label'],
            'task': 'cb'
        }
    
    def _process_wic(self, example):
        return {
            'text': f"Sentence 1: {example['sentence1']} Sentence 2: {example['sentence2']} Word: {example['word']}",
            'label': int(example['label']),
            'task': 'wic'
        }
    
    def create_few_shot_task(self, 
                           dataset_name: str, 
                           task_name: str,
                           k_shot: Optional[int] = None,
                           query_size: Optional[int] = None,
                           seed: Optional[int] = None) -> Tuple[List, List]:
        """Create a few-shot task with support and query sets."""
        if k_shot is None:
            k_shot = self.few_shot_config['default_k_shot']
        if query_size is None:
            query_size = self.few_shot_config['query_set_size']
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Load full dataset
        data_splits = self.load_dataset_splits(dataset_name, task_name)
        train_data = data_splits['train']
        
        # Get number of classes
        labels = [ex['label'] for ex in train_data]
        num_classes = len(set(labels))
        
        # Group examples by class
        class_examples = defaultdict(list)
        for example in train_data:
            class_examples[example['label']].append(example)
        
        # Sample support set (k examples per class)
        support_set = []
        for label in range(num_classes):
            if label in class_examples:
                examples = class_examples[label]
                if len(examples) >= k_shot:
                    sampled = random.sample(examples, k_shot)
                    support_set.extend(sampled)
                else:
                    # If not enough examples, use all available
                    support_set.extend(examples)
        
        # Sample query set
        remaining_examples = []
        for label in range(num_classes):
            if label in class_examples:
                examples = [ex for ex in class_examples[label] 
                           if ex not in support_set]
                remaining_examples.extend(examples)
        
        if len(remaining_examples) >= query_size:
            query_set = random.sample(remaining_examples, query_size)
        else:
            query_set = remaining_examples
        
        # Shuffle sets
        random.shuffle(support_set)
        random.shuffle(query_set)
        
        return support_set, query_set
    
    def create_meta_dataset(self, 
                          dataset_names: Optional[List[str]] = None,
                          num_tasks: Optional[int] = None,
                          split: str = 'train') -> List[Tuple[List, List]]:
        """Create a meta-dataset of few-shot tasks."""
        if dataset_names is None:
            dataset_names = []
            if self.datasets_config['glue']['enabled']:
                dataset_names.extend([('glue', task['name']) 
                                    for task in self.datasets_config['glue']['tasks']])
            if self.datasets_config['superglue']['enabled']:
                dataset_names.extend([('superglue', task['name']) 
                                    for task in self.datasets_config['superglue']['tasks']])
        
        if num_tasks is None:
            if split == 'train':
                num_tasks = self.few_shot_config['num_tasks_train']
            elif split == 'val':
                num_tasks = self.few_shot_config['num_tasks_val']
            else:
                num_tasks = self.few_shot_config['num_tasks_test']
        
        meta_dataset = []
        
        # Create tasks
        for i in tqdm(range(num_tasks), desc=f"Creating {split} meta-dataset"):
            # Randomly select a dataset and task
            dataset_name, task_name = random.choice(dataset_names)
            
            # Create few-shot task with different random seed
            support_set, query_set = self.create_few_shot_task(
                dataset_name, task_name, seed=i
            )
            
            meta_dataset.append((support_set, query_set))
        
        return meta_dataset
    
    def get_data_loader(self, 
                       examples: List[Dict],
                       batch_size: int = 32,
                       shuffle: bool = True) -> DataLoader:
        """Create a DataLoader from examples."""
        dataset = FewShotDataset(examples, self.tokenizer)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
    
    def save_meta_dataset(self, meta_dataset: List, filepath: str):
        """Save meta-dataset to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert to serializable format
        serializable_dataset = []
        for support_set, query_set in meta_dataset:
            serializable_dataset.append({
                'support': support_set,
                'query': query_set
            })
        
        with open(filepath, 'w') as f:
            json.dump(serializable_dataset, f, indent=2)
    
    def load_meta_dataset(self, filepath: str) -> List[Tuple[List, List]]:
        """Load meta-dataset from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        meta_dataset = []
        for task_data in data:
            meta_dataset.append((task_data['support'], task_data['query']))
        
        return meta_dataset 