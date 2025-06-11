"""
Main entry point for MELoRA experiments.
Orchestrates the complete experimental pipeline.
"""

import os
import argparse
import json
from typing import Dict, List, Tuple, Optional, Any

import torch
import numpy as np
import pandas as pd

# Import all modules
from dataset_loader import DatasetLoader
from synthetic_data_generator import SyntheticDataGenerator
from model import MELoRAModel
from trainer import MELoRATrainer
from baseline import create_baseline
from evaluation import Evaluator
from visualization import Visualizer
from analysis import Analyzer
import utils


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='MELoRA: Memory-Efficient LoRA Meta-Learning',
        epilog="""
        Examples:
        # Train all baselines only
        python main.py --baselines_only
        
        # Train specific baselines only
        python main.py --baselines_only --baseline_methods fomaml reptile
        
        # Train only fine-tuning baselines
        python main.py --mode baselines_only --baseline_methods fine_tuning lora_fine_tuning
        
        # Full MELoRA experiment (default)
        python main.py --mode full
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Experiment configuration
    parser.add_argument('--config', type=str, default='config.yml',
                       help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment run')
    
    # Mode selection
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'analyze', 'full', 'baselines_only'],
                       help='Experiment mode')
    parser.add_argument('--skip_baselines', action='store_true',
                       help='Skip baseline comparisons')
    parser.add_argument('--synthetic_only', action='store_true',
                       help='Use only synthetic data')
    
    # Baseline-specific options
    parser.add_argument('--baselines_only', action='store_true',
                       help='Train only baselines (skip MELoRA)')
    parser.add_argument('--baseline_methods', nargs='+', 
                       choices=['full_maml', 'fomaml', 'reptile', 'fine_tuning', 'lora_fine_tuning'],
                       help='Specific baseline methods to train (if not specified, trains all enabled baselines)')
    parser.add_argument('--skip_analysis', action='store_true',
                       help='Skip comparative analysis')
    
    # Model selection
    parser.add_argument('--model', type=str, default=None,
                       help='Model to use (overrides config)')
    parser.add_argument('--lora_rank', type=int, default=None,
                       help='LoRA rank (overrides config)')
    
    # Training parameters
    parser.add_argument('--num_iterations', type=int, default=None,
                       help='Number of meta-training iterations')
    parser.add_argument('--meta_batch_size', type=int, default=None,
                       help='Meta batch size')
    
    # Evaluation parameters
    parser.add_argument('--num_test_tasks', type=int, default=200,
                       help='Number of test tasks for evaluation')
    
    # Checkpointing
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to load')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from latest checkpoint')
    
    # Hardware
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'cpu'],
                       help='Device to use (overrides config)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (overrides config)')
    
    return parser.parse_args()


def setup_experiment(args) -> Dict:
    """Set up experiment configuration and directories."""
    # Load configuration
    config = utils.load_config(args.config)
    
    # Override config with command line arguments
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
    if args.model:
        config['models']['selected_model'] = args.model
    if args.lora_rank:
        config['lora']['default_rank'] = args.lora_rank
    if args.num_iterations:
        config['meta_learning']['num_meta_iterations'] = args.num_iterations
    if args.meta_batch_size:
        config['meta_learning']['default_meta_batch_size'] = args.meta_batch_size
    if args.device:
        config['experiment']['device'] = args.device
    if args.seed:
        config['experiment']['seed'] = args.seed
        config['reproducibility']['seed'] = args.seed
        
    # Set up logging
    logger = utils.setup_logging(config)
    
    # Set random seeds
    utils.set_seed(config['experiment']['seed'], config)
    
    # Set up experiment tracking
    utils.setup_tensorboard(config)
    utils.setup_mlflow(config)
    
    # Create output directories
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
    
    logger.info("Experiment setup complete")
    logger.info(f"Model: {config['models']['selected_model']}")
    logger.info(f"LoRA rank: {config['lora']['default_rank']}")
    logger.info(f"Device: {config['experiment']['device']}")
    
    return config


def load_or_create_datasets(config: Dict, args) -> Tuple[DatasetLoader, Dict[str, List]]:
    """Load or create datasets for experiments."""
    logger = utils.get_logger()
    
    # Initialize dataset loader
    dataset_loader = DatasetLoader(config_path=args.config)
    
    all_datasets = {
        'train': [],
        'val': [],
        'test': []
    }
    
    # Load synthetic data if enabled
    if config['synthetic_data']['enabled'] or args.synthetic_only:
        logger.info("Generating synthetic datasets...")
        synthetic_generator = SyntheticDataGenerator(config_path=args.config)
        
        # Generate meta-datasets
        synthetic_train = synthetic_generator.generate_meta_dataset(
            num_tasks=config['synthetic_data']['num_synthetic_tasks']
        )
        
        # Split into train/val/test
        n_train = int(len(synthetic_train) * 0.6)
        n_val = int(len(synthetic_train) * 0.2)
        
        all_datasets['train'].extend(synthetic_train[:n_train])
        all_datasets['val'].extend(synthetic_train[n_train:n_train+n_val])
        all_datasets['test'].extend(synthetic_train[n_train+n_val:])
        
        # Validate synthetic data
        validation_results = synthetic_generator.validate_synthetic_data(
            synthetic_train[:100]
        )
        logger.info(f"Synthetic data validation: {validation_results}")
        
    # Load real datasets unless synthetic only
    if not args.synthetic_only:
        logger.info("Loading real datasets...")
        
        # Create meta-datasets for each split
        for split in ['train', 'val', 'test']:
            split_name = 'validation' if split == 'val' else split
            meta_dataset = dataset_loader.create_meta_dataset(split=split_name)
            all_datasets[split].extend(meta_dataset)
            
    logger.info(f"Dataset sizes - Train: {len(all_datasets['train'])}, "
               f"Val: {len(all_datasets['val'])}, Test: {len(all_datasets['test'])}")
    
    return dataset_loader, all_datasets


def train_melora(config: Dict, 
                model: MELoRAModel,
                dataset_loader: DatasetLoader,
                datasets: Dict[str, List],
                args) -> MELoRATrainer:
    """Train MELoRA model."""
    logger = utils.get_logger()
    logger.info("Starting MELoRA training...")
    
    # Initialize trainer
    trainer = MELoRATrainer(model, config, dataset_loader)
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    elif args.resume:
        # Find latest checkpoint
        checkpoint_dir = config['experiment']['checkpoint_dir']
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
        if checkpoints:
            latest_checkpoint = max(checkpoints, 
                                  key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
            trainer.load_checkpoint(os.path.join(checkpoint_dir, latest_checkpoint))
            
    # Train model
    trainer.train(
        meta_train_tasks=datasets['train'],
        meta_val_tasks=datasets['val'],
        num_iterations=args.num_iterations
    )
    
    return trainer


def train_baselines_only(config: Dict,
                        model: MELoRAModel,
                        dataset_loader: DatasetLoader,
                        datasets: Dict[str, List],
                        args) -> Dict[str, Dict[str, Any]]:
    """Train only baseline methods."""
    logger = utils.get_logger()
    results = {}
    
    # Ensure checkpoint directory exists
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
    
    # Determine which baselines to train
    if args.baseline_methods:
        # Use specified baseline methods
        baseline_methods = args.baseline_methods
        logger.info(f"Training specified baselines: {baseline_methods}")
    else:
        # Use all enabled baselines from config
        baseline_methods = [m['name'] for m in config['baselines']['methods'] if m['enabled']]
        logger.info(f"Training all enabled baselines: {baseline_methods}")
    
    # Prepare test tasks for evaluation
    test_tasks = datasets['test'][:args.num_test_tasks]
    
    # Train and evaluate each baseline
    for baseline_name in baseline_methods:
        logger.info(f"Training baseline: {baseline_name}")
        
        try:
            # Create baseline model
            baseline_model = create_baseline(
                baseline_name, model, config, dataset_loader
            )
            
            # Train baseline
            logger.info(f"Starting training for {baseline_name}...")
            baseline_model.train(datasets['train'], datasets['val'])
            logger.info(f"Completed training for {baseline_name}")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config['experiment']['checkpoint_dir'], 
                f'{baseline_name}_checkpoint.pt'
            )
            baseline_model.save_checkpoint(checkpoint_path)
            logger.info(f"Saved {baseline_name} checkpoint to {checkpoint_path}")
            
            # Evaluate baseline
            logger.info(f"Evaluating {baseline_name}...")
            baseline_results = baseline_model.evaluate(test_tasks)
            results[baseline_name] = baseline_results
            logger.info(f"{baseline_name} evaluation results: {baseline_results}")
            
        except Exception as e:
            logger.error(f"Failed to train/evaluate {baseline_name}: {e}", exc_info=True)
            results[baseline_name] = {
                'training_completed': False,
                'evaluation_completed': False,
                'error': str(e)
            }
    
    return results


def evaluate_methods(config: Dict,
                    model: MELoRAModel,
                    dataset_loader: DatasetLoader,
                    datasets: Dict[str, List],
                    test_tasks: List[Tuple[List, List]],
                    args) -> Dict[str, Dict[str, Any]]:
    """Evaluate MELoRA and baseline methods."""
    logger = utils.get_logger()
    results = {}
    
    # Limit test tasks if specified
    test_tasks = test_tasks[:args.num_test_tasks]
    
    # Evaluate MELoRA
    logger.info("Evaluating MELoRA...")
    evaluator = Evaluator(model, config, dataset_loader)
    melora_results = evaluator.evaluate_meta_learning(test_tasks)
    results['MELoRA'] = melora_results
    
    # Save MELoRA evaluation results
    evaluator.save_results(
        os.path.join(config['experiment']['output_dir'], 'melora_results.json')
    )
    
    # Generate evaluation report
    report = evaluator.generate_report(
        os.path.join(config['experiment']['output_dir'], 'melora_report.txt')
    )
    logger.info(f"MELoRA Results:\n{report}")
    
    # Evaluate baselines unless skipped
    if not args.skip_baselines:
        baseline_methods = [m['name'] for m in config['baselines']['methods'] if m['enabled']]
        
        for baseline_name in baseline_methods:
            logger.info(f"Evaluating baseline: {baseline_name}")
            
            try:
                # Create baseline model
                baseline_model = create_baseline(
                    baseline_name, model, config, dataset_loader
                )
                
                # Train baseline (using full training data for fair comparison)
                # All baselines should be trained, including fine-tuning methods
                baseline_model.train(datasets['train'], datasets['val'])
                    
                # Evaluate baseline
                baseline_results = baseline_model.evaluate(test_tasks)
                results[baseline_name] = baseline_results
                
                logger.info(f"{baseline_name} results: {baseline_results}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {baseline_name}: {e}")
                
    return results


def analyze_results(config: Dict,
                   results: Dict[str, Dict[str, Any]],
                   model: MELoRAModel,
                   dataset_loader: DatasetLoader,
                   test_tasks: List[Tuple[List, List]],
                   args):
    """Perform comprehensive analysis of results."""
    logger = utils.get_logger()
    logger.info("Performing comprehensive analysis...")
    
    # Initialize analyzer
    analyzer = Analyzer(config)
    
    # Analyze experiment results
    analysis_results = analyzer.analyze_experiment_results(
        results,
        save_path=os.path.join(config['experiment']['output_dir'], 'analysis_results')
    )
    
    # Perform ablation study if enabled
    if config['analysis']['ablation']['enabled']:
        logger.info("Performing ablation study...")
        ablation_results = analyzer.ablation_study(
            model, dataset_loader, test_tasks[:50]
        )
        
    # Perform sensitivity analysis if enabled
    if config['analysis']['ablation']['sensitivity_analysis']:
        logger.info("Performing sensitivity analysis...")
        sensitivity_results = analyzer.sensitivity_analysis(
            model, dataset_loader, test_tasks[:50]
        )
        
    # Generate final report
    report_path = os.path.join(config['experiment']['output_dir'], 'final_report.txt')
    with open(report_path, 'w') as f:
        f.write(analyzer.generate_analysis_report())
        
    logger.info(f"Analysis complete. Report saved to {report_path}")


def visualize_results(config: Dict,
                     results: Dict[str, Dict[str, Any]],
                     trainer: Optional[MELoRATrainer] = None):
    """Generate visualizations for experimental results."""
    logger = utils.get_logger()
    logger.info("Generating visualizations...")
    
    # Initialize visualizer
    visualizer = Visualizer(config)
    
    # Plot method comparison
    visualizer.plot_parameter_comparison(
        results,
        metric='accuracy',
        save_name='method_comparison_accuracy'
    )
    
    visualizer.plot_parameter_comparison(
        results,
        metric='f1_macro',
        save_name='method_comparison_f1'
    )
    
    # Plot memory-performance trade-off
    results_list = [v for v in results.values()]
    labels = list(results.keys())
    visualizer.plot_pareto_frontier(
        results_list,
        x_metric='memory',
        y_metric='accuracy',
        labels=labels,
        save_name='memory_performance_tradeoff'
    )
    
    # Plot learning curves if trainer available
    if trainer and hasattr(trainer, 'metrics_history'):
        visualizer.plot_learning_curves(
            trainer.metrics_history,
            save_name='learning_curves'
        )
        
    # Plot memory usage profile
    if trainer and hasattr(trainer.memory_profiler, 'measurements'):
        visualizer.plot_memory_usage(
            trainer.memory_profiler.measurements,
            save_name='memory_profile'
        )
        
    logger.info(f"Visualizations saved to {config['visualization']['save_path']}")


def main():
    """Main entry point for MELoRA experiments."""
    # Parse arguments
    args = parse_arguments()
    
    # Set up experiment
    config = setup_experiment(args)
    logger = utils.get_logger()
    
    try:
        # Load or create datasets
        dataset_loader, datasets = load_or_create_datasets(config, args)
        
        # Create model
        logger.info("Creating MELoRA model...")
        model = MELoRAModel(config)
        
        # Log model information
        memory_usage = model.get_memory_usage()
        logger.info(f"Model memory usage: {memory_usage}")
        
        trainer = None
        results = {}
        
        # Handle baseline-only mode
        if args.mode == 'baselines_only' or args.baselines_only:
            logger.info("Running in baselines-only mode")
            results = train_baselines_only(config, model, dataset_loader, datasets, args)
            
            # Save baseline training results
            baseline_results_path = os.path.join(
                config['experiment']['output_dir'], 'baseline_training_results.json'
            )
            with open(baseline_results_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Baseline training results saved to {baseline_results_path}")
            
            # Save comparison results as CSV if we have evaluation results
            if results and any('accuracy_mean' in v for v in results.values() if isinstance(v, dict)):
                comparison_df = pd.DataFrame(results).T
                comparison_csv_path = os.path.join(config['experiment']['output_dir'], 'baseline_comparison_results.csv')
                comparison_df.to_csv(comparison_csv_path)
                logger.info(f"Baseline comparison results saved to {comparison_csv_path}")
            
        # MELoRA training mode
        elif args.mode in ['train', 'full']:
            trainer = train_melora(config, model, dataset_loader, datasets, args)
            
        # Evaluation mode
        if args.mode in ['evaluate', 'full'] and not (args.baselines_only or args.mode == 'baselines_only'):
            results = evaluate_methods(
                config, model, dataset_loader, datasets, datasets['test'], args
            )
            
            # Save comparison results
            comparison_df = pd.DataFrame(results).T
            comparison_df.to_csv(
                os.path.join(config['experiment']['output_dir'], 'comparison_results.csv')
            )
            
        # Analysis mode (skip if requested or in baselines-only mode)
        if (args.mode in ['analyze', 'full'] and results and 
            not args.skip_analysis and not (args.baselines_only or args.mode == 'baselines_only')):
            analyze_results(
                config, results, model, dataset_loader, datasets['test'], args
            )
            
        # Generate visualizations
        if results:
            visualize_results(config, results, trainer)
            
        logger.info("Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}", exc_info=True)
        raise
        
    finally:
        # Cleanup
        utils.cleanup()
        

if __name__ == '__main__':
    import pandas as pd  # Import here to avoid circular imports
    main() 