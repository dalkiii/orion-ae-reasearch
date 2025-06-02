import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import create_datasets, create_data_loaders
from src.trainer import ClassifierTrainer
from src.evaluator import ResultsEvaluator
from src.utils.config import ConfigManager, ExperimentConfig
from src.utils.logger import Logger
from src.utils.utils import set_seed, create_directories, save_results_json
from typing import List, Dict, Any


def run_single_experiment(config: ExperimentConfig, 
                         train_loader, 
                         val_loader, 
                         test_loader,
                         run_id: int,
                         logger: Logger) -> Dict[str, Any]:
    """
    Run a single experiment (one run)
    
    Args:
        config: Experiment configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        run_id: Run identifier
        logger: Logger instance
    
    Returns:
        Results dictionary
    """
    logger.log_run_start(run_id, config.num_runs)
    
    # Initialize trainer
    trainer = ClassifierTrainer(config.training, logger)
    
    # Train model
    history = trainer.train(train_loader, val_loader, run_id)
    
    # Evaluate on test set
    results = trainer.evaluate(test_loader, run_id)
    
    # Add training history to results
    results['history'] = history
    
    logger.log_run_end(run_id, results)
    
    return results


def run_multiple_experiments(config: ExperimentConfig) -> List[Dict[str, Any]]:
    """
    Run multiple experiments
    
    Args:
        config: Experiment configuration
    
    Returns:
        List of results from all runs
    """
    # Initialize logger
    logger = Logger(name=config.experiment_name, log_dir="./logs")
    logger.log_experiment_start(config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Create directories
    create_directories(config.results_dir, config.checkpoint_dir, "./logs")
    
    # Create datasets
    logger.info("Creating datasets...")
    start_time = time.time()
    
    _, train_dataset, val_dataset, test_dataset = create_datasets(
        root_dir=config.data.root_dir,
        imbalance_factors=config.data.imbalance_factors,
        test_samples_per_class=config.data.test_samples_per_class,
        train_val_split=config.data.train_val_split,
        seed=config.seed
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers
    )
    
    data_time = time.time() - start_time
    logger.info(f"Dataset creation completed in {data_time:.2f} seconds")
    
    # Run multiple experiments
    all_results = []
    experiment_start_time = time.time()
    
    for run_id in range(config.num_runs):
        run_start_time = time.time()
        
        # Set different seed for each run
        set_seed(config.seed + run_id)
        
        # Run single experiment
        results = run_single_experiment(
            config, train_loader, val_loader, test_loader, run_id, logger
        )
        
        # Save individual run results
        run_results_path = os.path.join(config.results_dir, f"run_{run_id + 1}_results.json")
        save_results_json(results, run_results_path)
        
        all_results.append(results)
        
        run_time = time.time() - run_start_time
        logger.info(f"Run {run_id + 1} completed in {run_time:.2f} seconds")
    
    total_experiment_time = time.time() - experiment_start_time
    logger.info(f"All runs completed in {total_experiment_time:.2f} seconds")
    
    return all_results


def analyze_results(all_results: List[Dict[str, Any]], 
                   config: ExperimentConfig,
                   logger: Logger) -> Dict[str, Any]:
    """
    Analyze results from multiple runs
    
    Args:
        all_results: List of results from all runs
        config: Experiment configuration
        logger: Logger instance
    
    Returns:
        Average performance metrics
    """
    logger.info("Analyzing results...")
    
    # Get class names (you might want to make this configurable)
    class_names = ["05cNm", "10cNm", "20cNm", "30cNm", "40cNm", "50cNm", "60cNm"]
    
    # Initialize evaluator
    evaluator = ResultsEvaluator(class_names=class_names, results_dir=config.results_dir)
    
    # Plot individual run training curves
    for i, results in enumerate(all_results):
        if 'history' in results:
            evaluator.plot_training_curves(results['history'], run_id=i)
    
    # Plot individual confusion matrices
    for i, results in enumerate(all_results):
        evaluator.plot_confusion_matrix(
            results['confusion_matrix'],
            title=f"Confusion Matrix (Run {i + 1})",
            save_name=f"confusion_matrix_run{i + 1}.png"
        )
    
    # Calculate average performance
    avg_performance = evaluator.calculate_average_performance(
        all_results, num_classes=config.model.num_classes
    )
    
    # Plot class performance heatmap
    evaluator.plot_class_performance_heatmap(all_results, num_classes=config.model.num_classes)
    
    # Save average results
    avg_results_path = os.path.join(config.results_dir, "average_results.json")
    save_results_json(avg_performance, avg_results_path)
    
    logger.log_experiment_end(avg_performance)
    
    return avg_performance


def main():
    """Main function"""
    # Load configuration
    config_path = "configs/classification_config.yaml"
    
    if os.path.exists(config_path):
        config = ConfigManager.load_from_yaml(config_path)
        print(f"Loaded configuration from {config_path}")
    else:
        # Create default configuration
        config = ConfigManager.create_default_config()
        
        # Update data root directory (you need to change this)
        config.data.root_dir = "./data/resized/measurementSeries_B"
        
        # Save default configuration
        os.makedirs("configs", exist_ok=True)
        ConfigManager.save_to_yaml(config, config_path)
        print(f"Created default configuration at {config_path}")
        print("Please update the data.root_dir in the config file and run again.")
        return
    
    # Validate configuration
    if not os.path.exists(config.data.root_dir):
        print(f"Error: Data directory does not exist: {config.data.root_dir}")
        print("Please update the data.root_dir in the config file.")
        return
    
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Number of runs: {config.num_runs}")
    print(f"Data directory: {config.data.root_dir}")
    print(f"Results will be saved to: {config.results_dir}")
    
    try:
        # Run experiments
        all_results = run_multiple_experiments(config)
        
        # Analyze results
        logger = Logger(name=config.experiment_name, log_dir="./logs")
        avg_performance = analyze_results(all_results, config, logger)
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Average Accuracy: {avg_performance['avg_accuracy']:.4f} ± {avg_performance['std_accuracy']:.4f}")
        print(f"Average Macro F1: {avg_performance['avg_macro_f1']:.4f} ± {avg_performance['std_macro_f1']:.4f}")
        print(f"Results saved to: {config.results_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()