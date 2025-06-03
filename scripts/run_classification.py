import sys
import os
import time
import copy
import glob
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Subset
from torchvision import transforms

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.dataset import create_datasets, create_data_loaders
from src.dataset import ScalogramDataset
from src.trainer import ClassifierTrainer
from src.evaluator import ResultsEvaluator
from src.gan.models import IMBSpecGANGenerator
from src.utils.config import ConfigManager, ExperimentConfig
from src.utils.logger import Logger
from src.utils.utils import set_seed, create_directories, save_results_json
from typing import List, Dict, Any


class GANAugmentedDataset(ScalogramDataset):
    """Extended dataset with GAN augmentation capabilities"""
    
    def __init__(self, 
                 root_dir: str,
                 transform=None,
                 imbalance_factors=None,
                 test_samples_per_class: int = 100,
                 seed: int = 42,
                 gan_checkpoint: str = None,
                 gan_latent_dim: int = 200,
                 gan_num_classes: int = 7,
                 gan_d_model_dim: int = 128,
                 augmentation_strategy: str = "balance"):
        """Initialize GAN-augmented dataset"""
        # Initialize base dataset
        super().__init__(
            root_dir=root_dir,
            transform=None,  # Apply transform after GAN augmentation
            imbalance_factors=imbalance_factors,
            test_samples_per_class=test_samples_per_class,
            seed=seed,
            data_format="classification",
            load_mode="directory"
        )
        
        self.final_transform = transform
        self.gan_checkpoint = gan_checkpoint
        self.gan_latent_dim = gan_latent_dim
        self.gan_num_classes = gan_num_classes
        self.gan_d_model_dim = gan_d_model_dim
        self.augmentation_strategy = augmentation_strategy
        
        # Store original counts before augmentation
        self.original_counts = self._count_samples_by_class()
        
        # Apply GAN augmentation if checkpoint provided
        if self.gan_checkpoint:
            self._add_synthetic_samples()
    
    def _count_samples_by_class(self):
        """Count samples by class"""
        counts = defaultdict(int)
        for _, label in self.data_samples:
            counts[label] += 1
        return dict(counts)
    
    def _add_synthetic_samples(self):
        """Add synthetic samples using trained GAN"""
        print(f"Adding synthetic samples using checkpoint: {self.gan_checkpoint}")
        
        if not os.path.exists(self.gan_checkpoint):
            print(f"Warning: GAN checkpoint not found: {self.gan_checkpoint}")
            return
        
        # Determine augmentation targets
        current_counts = self.original_counts
        
        if self.augmentation_strategy == "balance":
            # Balance to majority class count
            majority_counts = []
            for class_name, factor in self.imbalance_factors.items():
                if factor == 1.0:  # Majority class
                    class_label = self.class_map[class_name]
                    if class_label in current_counts:
                        majority_counts.append(current_counts[class_label])
            
            if not majority_counts:
                print("No majority classes found for balancing")
                return
            
            target_count = int(np.mean(majority_counts))
            
        elif self.augmentation_strategy == "oversample":
            # Oversample all classes to maximum count
            target_count = max(current_counts.values()) if current_counts else 0
        
        else:
            raise ValueError(f"Unknown augmentation strategy: {self.augmentation_strategy}")
        
        # Initialize GAN generator
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = self.gan_latent_dim + self.gan_num_classes
        
        generator = IMBSpecGANGenerator(
            input_dim=input_dim,
            d=self.gan_d_model_dim
        ).to(device)
        
        # Load checkpoint
        generator.load_state_dict(torch.load(self.gan_checkpoint, map_location=device))
        generator.eval()
        
        # Transform for converting tensor to PIL
        to_pil = transforms.ToPILImage()
        
        augmentation_counts = {}
        
        # Generate synthetic samples for each class
        for class_name, class_label in self.class_map.items():
            current_count = current_counts.get(class_label, 0)
            needed = max(0, target_count - current_count)
            
            if needed == 0:
                augmentation_counts[class_name] = 0
                continue
            
            print(f"  Generating {needed} samples for class {class_name}")
            
            # Generate synthetic samples
            with torch.no_grad():
                for _ in range(needed):
                    # Sample latent vector
                    z = torch.randn(1, self.gan_latent_dim, device=device)
                    
                    # Create one-hot label
                    one_hot = torch.zeros(1, self.gan_num_classes, device=device)
                    one_hot[0, class_label] = 1
                    
                    # Generate sample
                    latent_input = torch.cat([z, one_hot], dim=1)
                    fake_image = generator(latent_input)
                    
                    # Convert to PIL Image
                    # Normalize from [-1, 1] to [0, 1]
                    fake_image = (fake_image + 1) / 2
                    fake_image = torch.clamp(fake_image, 0, 1)
                    
                    # Convert to PIL
                    pil_image = to_pil(fake_image.squeeze(0).cpu()).convert('RGB')
                    
                    # Add to dataset
                    self.data_samples.append((pil_image, class_label))
            
            augmentation_counts[class_name] = needed
        
        # Print augmentation summary
        print("\nAugmentation Summary:")
        for class_name, added in augmentation_counts.items():
            class_label = self.class_map[class_name]
            original = current_counts.get(class_label, 0)
            final = original + added
            print(f"  {class_name}: {original} → {final} (+{added})")
    
    def __getitem__(self, idx):
        """Get item with final transform applied"""
        sample, label = self.data_samples[idx]
        
        # Apply final transform if provided
        if self.final_transform:
            sample = self.final_transform(sample)
        
        return sample, torch.tensor(label, dtype=torch.long)


def create_gan_augmented_datasets(root_dir: str,
                                 imbalance_factors: dict,
                                 test_samples_per_class: int = 100,
                                 train_val_split: float = 0.8,
                                 seed: int = 42,
                                 gan_checkpoint: str = None,
                                 gan_config: dict = None):
    """Create GAN-augmented datasets"""
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Default GAN config
    default_gan_config = {
        'latent_dim': 200,
        'num_classes': 7,
        'd_model_dim': 128,
        'augmentation_strategy': 'balance'
    }
    
    if gan_config:
        default_gan_config.update(gan_config)
    
    # Create full dataset with GAN augmentation
    full_dataset = GANAugmentedDataset(
        root_dir=root_dir,
        transform=transform,
        imbalance_factors=imbalance_factors,
        test_samples_per_class=test_samples_per_class,
        seed=seed,
        gan_checkpoint=gan_checkpoint,
        gan_latent_dim=default_gan_config['latent_dim'],
        gan_num_classes=default_gan_config['num_classes'],
        gan_d_model_dim=default_gan_config['d_model_dim'],
        augmentation_strategy=default_gan_config['augmentation_strategy']
    )
    
    # Split into train and validation
    total_samples = len(full_dataset.data_samples)
    indices = list(range(total_samples))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_size = int(train_val_split * total_samples)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # Create test dataset (without augmentation)
    test_dataset = full_dataset.get_test_dataset(transform)
    
    return full_dataset, train_dataset, val_dataset, test_dataset


def run_single_experiment(config: ExperimentConfig, 
                         train_loader, 
                         val_loader, 
                         test_loader,
                         run_id: int,
                         logger: Logger) -> Dict[str, Any]:
    """Run a single experiment (one run)"""
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


def run_multiple_experiments(config: ExperimentConfig,
                           gan_checkpoint: str = None,
                           gan_config: dict = None) -> List[Dict[str, Any]]:
    """Run multiple experiments"""
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
    
    if gan_checkpoint:
        # Use GAN-augmented datasets
        logger.info(f"Using GAN augmentation with checkpoint: {gan_checkpoint}")
        _, train_dataset, val_dataset, test_dataset = create_gan_augmented_datasets(
            root_dir=config.data.root_dir,
            imbalance_factors=config.data.imbalance_factors,
            test_samples_per_class=config.data.test_samples_per_class,
            train_val_split=config.data.train_val_split,
            seed=config.seed,
            gan_checkpoint=gan_checkpoint,
            gan_config=gan_config
        )
    else:
        # Use regular datasets (default behavior)
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
    """Analyze results from multiple runs"""
    logger.info("Analyzing results...")
    
    # Get class names
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


def evaluate_gan_checkpoints(config: ExperimentConfig,
                           checkpoint_dir: str,
                           checkpoint_interval: int = 30,
                           gan_config: dict = None) -> pd.DataFrame:
    """Evaluate classification performance across GAN checkpoints"""
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "netG_epoch_*.pth"))
    
    if not checkpoint_files:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Filter checkpoints by interval
    epoch_nums = []
    for cp in checkpoint_files:
        match = re.search(r"netG_epoch_(\d+)\.pth", os.path.basename(cp))
        if match:
            epoch_nums.append(int(match.group(1)))
    
    if not epoch_nums:
        raise ValueError("No valid checkpoint files found")
    
    min_epoch = min(epoch_nums)
    valid_checkpoints = []
    
    for cp in checkpoint_files:
        match = re.search(r"netG_epoch_(\d+)\.pth", os.path.basename(cp))
        if match:
            epoch = int(match.group(1))
            if (epoch - min_epoch) % checkpoint_interval == 0:
                valid_checkpoints.append((epoch, cp))
    
    valid_checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(valid_checkpoints)} valid checkpoints to evaluate")
    
    # Store results
    results = []
    
    # Evaluate baseline (no augmentation)
    print("\n" + "="*60)
    print("EVALUATING BASELINE (NO AUGMENTATION)")
    print("="*60)
    
    baseline_results = run_multiple_experiments(config)
    baseline_performance = analyze_results(baseline_results, config,
                                         Logger(name=config.experiment_name, log_dir="./logs"))
    
    results.append({
        'Checkpoint': 'Baseline',
        'Epoch': 0,
        'Accuracy_Mean': baseline_performance['avg_accuracy'],
        'Accuracy_Std': baseline_performance['std_accuracy'],
        'Macro_F1_Mean': baseline_performance['avg_macro_f1'],
        'Macro_F1_Std': baseline_performance['std_macro_f1']
    })
    
    print(f"Baseline - Accuracy: {baseline_performance['avg_accuracy']:.4f} ± {baseline_performance['std_accuracy']:.4f}, "
          f"Macro F1: {baseline_performance['avg_macro_f1']:.4f} ± {baseline_performance['std_macro_f1']:.4f}")
    
    best_f1 = baseline_performance['avg_macro_f1']
    best_checkpoint = 'Baseline'
    
    # Evaluate each checkpoint
    for epoch, checkpoint_path in valid_checkpoints:
        print(f"\n" + "="*60)
        print(f"EVALUATING CHECKPOINT: {os.path.basename(checkpoint_path)}")
        print("="*60)
        
        # Update experiment name and results directory for this checkpoint
        checkpoint_config = copy.deepcopy(config)
        checkpoint_config.experiment_name = f"{config.experiment_name}_epoch_{epoch}"
        checkpoint_config.results_dir = os.path.join(config.results_dir, f"epoch_{epoch}")
        
        # Run experiments with GAN augmentation
        checkpoint_results = run_multiple_experiments(
            checkpoint_config, 
            gan_checkpoint=checkpoint_path,
            gan_config=gan_config
        )
        
        checkpoint_performance = analyze_results(
            checkpoint_results, checkpoint_config,
            Logger(name=checkpoint_config.experiment_name, log_dir="./logs")
        )
        
        print(f"Epoch {epoch} - Accuracy: {checkpoint_performance['avg_accuracy']:.4f} ± {checkpoint_performance['std_accuracy']:.4f}, "
              f"Macro F1: {checkpoint_performance['avg_macro_f1']:.4f} ± {checkpoint_performance['std_macro_f1']:.4f}")
        
        # Store results
        results.append({
            'Checkpoint': os.path.basename(checkpoint_path),
            'Epoch': epoch,
            'Accuracy_Mean': checkpoint_performance['avg_accuracy'],
            'Accuracy_Std': checkpoint_performance['std_accuracy'],
            'Macro_F1_Mean': checkpoint_performance['avg_macro_f1'],
            'Macro_F1_Std': checkpoint_performance['std_macro_f1']
        })
        
        # Track best performance
        if checkpoint_performance['avg_macro_f1'] > best_f1:
            best_f1 = checkpoint_performance['avg_macro_f1']
            best_checkpoint = os.path.basename(checkpoint_path)
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    print(f"\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Best checkpoint: {best_checkpoint}")
    print(f"Best Macro F1: {best_f1:.4f}")
    
    # Save results
    output_path = os.path.join(config.results_dir, "checkpoint_evaluation_results.csv")
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    
    return df


def plot_checkpoint_performance(df: pd.DataFrame, save_path: str = None):
    """Plot checkpoint performance results"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Filter out baseline for plotting
    plot_df = df[df['Epoch'] > 0].copy()
    
    if len(plot_df) == 0:
        print("No checkpoint data to plot")
        return
    
    epochs = plot_df['Epoch'].values
    acc_mean = plot_df['Accuracy_Mean'].values * 100
    acc_std = plot_df['Accuracy_Std'].values * 100
    f1_mean = plot_df['Macro_F1_Mean'].values * 100
    f1_std = plot_df['Macro_F1_Std'].values * 100
    
    # Accuracy plot
    ax1.errorbar(epochs, acc_mean, yerr=acc_std, 
                marker='o', linewidth=2, capsize=5, label='Accuracy')
    ax1.set_xlabel('Training Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Classification Accuracy vs GAN Training Progress')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # F1 score plot
    ax2.errorbar(epochs, f1_mean, yerr=f1_std, 
                marker='s', linewidth=2, capsize=5, label='Macro F1', color='orange')
    ax2.set_xlabel('Training Epochs')
    ax2.set_ylabel('Macro F1 Score (%)')
    ax2.set_title('Macro F1 Score vs GAN Training Progress')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add baseline lines if available
    baseline_row = df[df['Epoch'] == 0]
    if len(baseline_row) > 0:
        baseline_acc = baseline_row['Accuracy_Mean'].iloc[0] * 100
        baseline_f1 = baseline_row['Macro_F1_Mean'].iloc[0] * 100
        
        ax1.axhline(y=baseline_acc, color='red', linestyle='--', alpha=0.7, 
                   label=f'Baseline ({baseline_acc:.1f}%)')
        ax2.axhline(y=baseline_f1, color='red', linestyle='--', alpha=0.7, 
                   label=f'Baseline ({baseline_f1:.1f}%)')
        
        ax1.legend()
        ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run classification experiments")
    
    # Basic arguments (keeping original simple interface)
    parser.add_argument("--config", type=str, default="configs/classification_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data-root", type=str,
                       help="Root directory containing data (overrides config)")
    parser.add_argument("--results-dir", type=str,
                       help="Results directory (overrides config)")
    parser.add_argument("--num-runs", type=int,
                       help="Number of runs (overrides config)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # GAN augmentation arguments (optional, for advanced usage)
    parser.add_argument("--gan-checkpoint", type=str,
                       help="Path to GAN checkpoint for augmentation (enables GAN mode)")
    parser.add_argument("--gan-checkpoint-dir", type=str,
                       help="Directory containing GAN checkpoints for evaluation")
    parser.add_argument("--gan-latent-dim", type=int, default=200,
                       help="GAN latent dimension")
    parser.add_argument("--gan-num-classes", type=int, default=7,
                       help="Number of classes for GAN")
    parser.add_argument("--gan-d-model-dim", type=int, default=128,
                       help="GAN model dimension")
    parser.add_argument("--augmentation-strategy", type=str, default="balance",
                       choices=["balance", "oversample"],
                       help="Augmentation strategy")
    parser.add_argument("--checkpoint-interval", type=int, default=30,
                       help="Checkpoint interval for evaluation")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configuration
    if os.path.exists(args.config):
        config = ConfigManager.load_from_yaml(args.config)
        print(f"Loaded configuration from {args.config}")
    else:
        # Create default configuration
        config = ConfigManager.create_default_config()
        
        # Update with default data root directory
        if not args.data_root:
            config.data.root_dir = "./data/resized/measurementSeries_B"
        
        # Save default configuration
        os.makedirs("configs", exist_ok=True)
        ConfigManager.save_to_yaml(config, args.config)
        print(f"Created default configuration at {args.config}")
        
        if not args.data_root:
            print("Please update the data.root_dir in the config file or use --data-root argument.")
            return
    
    # Override config with command line arguments
    if args.data_root:
        config.data.root_dir = args.data_root
    if args.results_dir:
        config.results_dir = args.results_dir
    if args.num_runs:
        config.num_runs = args.num_runs
    
    # Validate data directory
    if not os.path.exists(config.data.root_dir):
        print(f"Error: Data directory does not exist: {config.data.root_dir}")
        return
    
    # Determine execution mode based on arguments
    use_gan = args.gan_checkpoint is not None
    evaluate_checkpoints = args.gan_checkpoint_dir is not None
    
    # Prepare GAN config if needed
    gan_config = None
    if use_gan or evaluate_checkpoints:
        # Start with config file settings if available
        if hasattr(config, 'gan'):
            gan_config = {
                'latent_dim': getattr(config.gan, 'latent_dim', 200),
                'num_classes': getattr(config.gan, 'num_classes', 7),
                'd_model_dim': getattr(config.gan, 'd_model_dim', 128),
                'augmentation_strategy': getattr(config.gan, 'augmentation_strategy', 'balance')
            }
        else:
            gan_config = {
                'latent_dim': 200,
                'num_classes': 7,
                'd_model_dim': 128,
                'augmentation_strategy': 'balance'
            }
        
        # Override with command line arguments if they differ from defaults
        if args.gan_latent_dim != 200:
            gan_config['latent_dim'] = args.gan_latent_dim
        if args.gan_num_classes != 7:
            gan_config['num_classes'] = args.gan_num_classes
        if args.gan_d_model_dim != 128:
            gan_config['d_model_dim'] = args.gan_d_model_dim
        if args.augmentation_strategy != 'balance':
            gan_config['augmentation_strategy'] = args.augmentation_strategy
    
    try:
        if evaluate_checkpoints:
            # Multiple checkpoint evaluation mode
            print(f"Evaluating GAN checkpoints in: {args.gan_checkpoint_dir}")
            print(f"Checkpoint interval: {args.checkpoint_interval}")
            
            # Use checkpoint evaluation settings from config if available
            checkpoint_interval = args.checkpoint_interval
            if hasattr(config, 'checkpoint_evaluation') and args.checkpoint_interval == 30:  # Using default
                checkpoint_interval = getattr(config.checkpoint_evaluation, 'checkpoint_interval', 30)
            
            # Run checkpoint evaluation
            df = evaluate_gan_checkpoints(
                config, 
                args.gan_checkpoint_dir,
                checkpoint_interval,
                gan_config
            )
            
            # Plot results
            plot_save_path = os.path.join(config.results_dir, "checkpoint_performance_vs_epochs.png")
            plot_checkpoint_performance(df, save_path=plot_save_path)
            
            print(f"\nCheckpoint evaluation completed successfully!")
            print(f"Results saved to: {config.results_dir}")
        
        else:
            # Single experiment mode (default)
            print(f"Starting experiment: {config.experiment_name}")
            print(f"Number of runs: {config.num_runs}")
            print(f"Data directory: {config.data.root_dir}")
            print(f"Results will be saved to: {config.results_dir}")
            
            if use_gan:
                print(f"Using GAN augmentation: {args.gan_checkpoint}")
            else:
                print("Running standard classification on imbalanced data (default)")
            
            # Run experiments
            all_results = run_multiple_experiments(config, args.gan_checkpoint, gan_config)
            
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