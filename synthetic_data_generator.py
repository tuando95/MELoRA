"""
Synthetic data generator for MELoRA experiments.
Generates controlled synthetic tasks for systematic evaluation.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import random
import yaml
from scipy.special import softmax
from tqdm import tqdm


class SyntheticDataGenerator:
    """Generates synthetic datasets for meta-learning experiments."""
    
    def __init__(self, config_path: str = 'config.yml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.synthetic_config = self.config['synthetic_data']
        self.task_gen_config = self.synthetic_config['task_generation']
        self.task_dist_config = self.synthetic_config['task_distribution']
        
        # Initialize vocabulary embeddings
        self.vocab_size = self.task_gen_config['vocab_size']
        self.embedding_dim = self.task_gen_config['embedding_dim']
        self._initialize_vocabulary()
        
    def _initialize_vocabulary(self):
        """Initialize vocabulary and embeddings."""
        # Generate random embeddings for vocabulary
        np.random.seed(self.config['reproducibility']['seed'])
        self.vocab_embeddings = np.random.randn(
            self.vocab_size, self.embedding_dim
        )
        # Normalize embeddings
        norms = np.linalg.norm(self.vocab_embeddings, axis=1, keepdims=True)
        self.vocab_embeddings = self.vocab_embeddings / (norms + 1e-8)
        
        # Create vocabulary tokens
        self.vocabulary = [f"token_{i}" for i in range(self.vocab_size)]
        
    def generate_task_parameters(self,
                               num_classes: int = 2,
                               task_similarity: float = 0.5,
                               seed: Optional[int] = None) -> Dict:
        """Generate parameters for a synthetic task."""
        if seed is not None:
            np.random.seed(seed)
            
        # Sample classification weights from Gaussian
        # Variance controls task similarity
        w = np.random.normal(0, task_similarity, 
                           size=(num_classes, self.embedding_dim))
        
        # Sample bias terms
        b = np.random.normal(0, task_similarity * 0.1, size=num_classes)
        
        # Normalize weights to control task difficulty
        w = w / np.linalg.norm(w, axis=1, keepdims=True)
        
        return {
            'weights': w,
            'bias': b,
            'num_classes': num_classes,
            'task_similarity': task_similarity
        }
    
    def generate_sequence(self,
                         task_params: Dict,
                         sequence_length: int,
                         difficulty_beta: float = 0.5) -> Tuple[List[int], np.ndarray]:
        """Generate a sequence for a given task."""
        weights = task_params['weights']
        
        # Sample token indices based on task-specific distribution
        # Higher beta = stronger correlation with labels
        token_probs = np.zeros(self.vocab_size)
        
        # Create task-specific token distribution
        for token_idx in range(self.vocab_size):
            embedding = self.vocab_embeddings[token_idx]
            # Compute affinity with each class
            affinities = weights @ embedding
            max_affinity = np.max(affinities)
            token_probs[token_idx] = np.exp(difficulty_beta * max_affinity)
        
        # Normalize probabilities
        token_probs = token_probs / np.sum(token_probs)
        
        # Sample sequence
        sequence_indices = np.random.choice(
            self.vocab_size, 
            size=sequence_length,
            p=token_probs
        )
        
        # Compute mean embedding
        sequence_embeddings = self.vocab_embeddings[sequence_indices]
        mean_embedding = np.mean(sequence_embeddings, axis=0)
        
        return sequence_indices.tolist(), mean_embedding
    
    def assign_label(self,
                    mean_embedding: np.ndarray,
                    task_params: Dict,
                    linear_separability: float = 0.7) -> int:
        """Assign label to a sequence based on task parameters."""
        weights = task_params['weights']
        bias = task_params['bias']
        
        # Compute logits
        logits = weights @ mean_embedding + bias
        
        # Add noise to control linear separability
        noise_scale = (1 - linear_separability) * np.std(logits)
        noise = np.random.normal(0, noise_scale, size=logits.shape)
        logits = logits + noise
        
        # Sample label from softmax distribution
        probs = softmax(logits)
        label = np.random.choice(len(probs), p=probs)
        
        return label
    
    def generate_dataset(self,
                        task_params: Dict,
                        num_samples: int,
                        sequence_length: int = 64,
                        difficulty_beta: float = 0.5,
                        linear_separability: float = 0.7) -> List[Dict]:
        """Generate a dataset for a specific task."""
        dataset = []
        
        for _ in range(num_samples):
            # Generate sequence
            sequence_indices, mean_embedding = self.generate_sequence(
                task_params, sequence_length, difficulty_beta
            )
            
            # Assign label
            label = self.assign_label(
                mean_embedding, task_params, linear_separability
            )
            
            # Create text from indices
            text = ' '.join([self.vocabulary[idx] for idx in sequence_indices])
            
            dataset.append({
                'text': text,
                'label': label,
                'sequence_indices': sequence_indices,
                'mean_embedding': mean_embedding.tolist(),
                'task': 'synthetic'
            })
        
        return dataset
    
    def generate_few_shot_task(self,
                             k_shot: int = 10,
                             query_size: int = 100,
                             num_classes: Optional[int] = None,
                             sequence_length: Optional[int] = None,
                             task_similarity: Optional[float] = None,
                             difficulty_beta: Optional[float] = None,
                             linear_separability: Optional[float] = None,
                             seed: Optional[int] = None) -> Tuple[List[Dict], List[Dict]]:
        """Generate a few-shot task with support and query sets."""
        # Use default values if not specified
        if num_classes is None:
            num_classes = self.task_gen_config['default_num_classes']
        if sequence_length is None:
            sequence_length = self.task_gen_config['default_seq_length']
        if task_similarity is None:
            task_similarity = self.task_dist_config['default_similarity']
        if difficulty_beta is None:
            difficulty_beta = self.task_dist_config['default_difficulty']
        if linear_separability is None:
            linear_separability = self.task_dist_config['linear_separability']
            
        # Generate task parameters
        task_params = self.generate_task_parameters(
            num_classes, task_similarity, seed
        )
        
        # Generate support set (k examples per class)
        support_set = []
        for class_idx in range(num_classes):
            class_examples = []
            attempts = 0
            while len(class_examples) < k_shot and attempts < k_shot * 10:
                example = self.generate_dataset(
                    task_params, 1, sequence_length, 
                    difficulty_beta, linear_separability
                )[0]
                if example['label'] == class_idx:
                    class_examples.append(example)
                attempts += 1
            
            # If we couldn't generate enough examples for this class,
            # fill with any examples and relabel
            while len(class_examples) < k_shot:
                example = self.generate_dataset(
                    task_params, 1, sequence_length,
                    difficulty_beta, linear_separability
                )[0]
                example['label'] = class_idx
                class_examples.append(example)
            
            support_set.extend(class_examples)
        
        # Generate query set
        query_set = self.generate_dataset(
            task_params, query_size, sequence_length,
            difficulty_beta, linear_separability
        )
        
        # Shuffle sets
        random.shuffle(support_set)
        random.shuffle(query_set)
        
        return support_set, query_set
    
    def generate_meta_dataset(self,
                            num_tasks: Optional[int] = None,
                            vary_parameters: bool = True) -> List[Tuple[List[Dict], List[Dict]]]:
        """Generate a meta-dataset of synthetic tasks."""
        if num_tasks is None:
            num_tasks = self.synthetic_config['num_synthetic_tasks']
            
        meta_dataset = []
        
        for i in tqdm(range(num_tasks), desc="Generating synthetic tasks"):
            if vary_parameters:
                # Sample parameters from configured ranges
                num_classes = random.choice(
                    self.task_gen_config['num_classes_options']
                )
                sequence_length = random.choice(
                    self.task_gen_config['sequence_lengths']
                )
                task_similarity = random.choice(
                    self.task_dist_config['task_similarity_sigmas']
                )
                difficulty_beta = random.choice(
                    self.task_dist_config['task_difficulty_betas']
                )
            else:
                # Use default parameters
                num_classes = None
                sequence_length = None
                task_similarity = None
                difficulty_beta = None
            
            # Generate task
            support_set, query_set = self.generate_few_shot_task(
                num_classes=num_classes,
                sequence_length=sequence_length,
                task_similarity=task_similarity,
                difficulty_beta=difficulty_beta,
                seed=i
            )
            
            meta_dataset.append((support_set, query_set))
        
        return meta_dataset
    
    def analyze_task_properties(self, 
                              support_set: List[Dict],
                              query_set: List[Dict]) -> Dict:
        """Analyze properties of a generated task."""
        all_examples = support_set + query_set
        
        # Extract labels
        labels = [ex['label'] for ex in all_examples]
        support_labels = [ex['label'] for ex in support_set]
        query_labels = [ex['label'] for ex in query_set]
        
        # Compute statistics
        num_classes = len(set(labels))
        label_distribution = {i: labels.count(i) for i in range(num_classes)}
        support_distribution = {i: support_labels.count(i) 
                              for i in range(num_classes)}
        query_distribution = {i: query_labels.count(i) 
                            for i in range(num_classes)}
        
        # Compute sequence statistics
        sequence_lengths = [len(ex['text'].split()) for ex in all_examples]
        
        # Compute embedding statistics if available
        if 'mean_embedding' in all_examples[0]:
            embeddings = np.array([ex['mean_embedding'] for ex in all_examples])
            embedding_variance = np.var(embeddings, axis=0).mean()
        else:
            embedding_variance = None
        
        return {
            'num_classes': num_classes,
            'total_examples': len(all_examples),
            'support_size': len(support_set),
            'query_size': len(query_set),
            'label_distribution': label_distribution,
            'support_distribution': support_distribution,
            'query_distribution': query_distribution,
            'avg_sequence_length': np.mean(sequence_lengths),
            'std_sequence_length': np.std(sequence_lengths),
            'embedding_variance': embedding_variance
        }
    
    def validate_synthetic_data(self, meta_dataset: List[Tuple]) -> Dict:
        """Validate properties of synthetic meta-dataset."""
        all_stats = []
        
        for support_set, query_set in tqdm(meta_dataset[:100], 
                                         desc="Validating synthetic data"):
            stats = self.analyze_task_properties(support_set, query_set)
            all_stats.append(stats)
        
        # Aggregate statistics
        aggregated = {
            'avg_support_size': np.mean([s['support_size'] for s in all_stats]),
            'avg_query_size': np.mean([s['query_size'] for s in all_stats]),
            'avg_num_classes': np.mean([s['num_classes'] for s in all_stats]),
            'avg_sequence_length': np.mean([s['avg_sequence_length'] 
                                          for s in all_stats]),
            'class_balance_score': self._compute_class_balance_score(all_stats)
        }
        
        return aggregated
    
    def _compute_class_balance_score(self, stats_list: List[Dict]) -> float:
        """Compute how balanced the class distributions are."""
        balance_scores = []
        
        for stats in stats_list:
            dist = stats['label_distribution']
            if len(dist) > 0:
                counts = list(dist.values())
                # Compute coefficient of variation
                cv = np.std(counts) / (np.mean(counts) + 1e-8)
                balance_scores.append(1 - cv)  # Higher score = more balanced
        
        return np.mean(balance_scores) if balance_scores else 0.0 