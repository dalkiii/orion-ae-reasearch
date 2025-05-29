import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.gan.gan_trainer import GANTrainer
from src.utils.logger import Logger
from src.utils.utils import set_seed, create_directories


def load_gan_config(config_path: str) -> dict:
    """Load GAN configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_gan_config(config_dict: dict) -> dict:
    """Create GAN configuration from dictionary"""
    gan_config = config_dict.get("gan", {})
    paths_config = config_dict.get("paths", {})
    experiment_config = config_dict.get("experiment", {})
    
    # Merge all configurations
    merged_config = {
        **gan_config,
        **paths_config,
        **experiment_config
    }
    
    return merged_config


def validate_config(config: dict) -> None:
    """Validate configuration"""
    required_keys = ["data_root", "checkpoint_dir", "results_dir"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Check if data directory exists
    if not os.path.exists(config["data_root"]):
        raise ValueError(f"Data directory does not exist: {config["data_root"]}")


def print_config_summary(config: dict) -> None:
    """Print configuration summary"""
    print("="*60)
    print("GAN TRAINING CONFIGURATION")
    print("="*60)
    print(f"Data root: {config.get("data_root", "N/A")}")
    print(f"Checkpoint directory: {config.get("checkpoint_dir", "N/A")}")
    print(f"Results directory: {config.get("results_dir", "N/A")}")
    print(f"Batch size: {config.get("batch_size", 64)}")
    print(f"Number of epochs: {config.get("num_epochs", 2000)}")
    print(f"Latent dimension: {config.get("latent_dim", 200)}")
    print(f"Number of classes: {config.get("num_classes", 7)}")
    print(f"Generator LR: {config.get("g_lr", 1e-5)}")
    print(f"Discriminator LR: {config.get("d_lr", 1e-6)}")
    print(f"Lambda GP: {config.get("lambda_gp", 15.0)}")
    
    # Print imbalance ratios
    imbalance_ratios = config.get("imbalance_ratios", {})
    if imbalance_ratios:
        print(f"Class imbalance ratios:")
        for class_name, ratio in imbalance_ratios.items():
            print(f"  {class_name}cNm: {ratio}")
    
    print("="*60)


def main():
    """Main GAN training function"""
    parser = argparse.ArgumentParser(description="Train GAN for acoustic emission data augmentation")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/gan_config.yaml",
        help="Path to GAN configuration file"
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        help="Root directory containing training data (overrides config)"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        help="Checkpoint directory (overrides config)"
    )
    parser.add_argument(
        "--results-dir", 
        type=str, 
        help="Results directory (overrides config)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--no-resume", 
        action="store_true",
        help="Don't resume from checkpoint, start from scratch"
    )
    parser.add_argument(
        "--generate-only", 
        action="store_true",
        help="Only generate data using existing model, skip training"
    )
    parser.add_argument(
        "--samples-per-class", 
        type=int, 
        default=1000,
        help="Number of samples to generate per class (for --generate-only)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config_dict = load_gan_config(args.config)
        print(f"Loaded configuration from: {args.config}")
    else:
        print(f"Configuration file not found: {args.config}")
        print("Using default configuration...")
        config_dict = {
            "gan": {},
            "paths": {
                "data_root": "./data",
                "checkpoint_dir": "./checkpoints/gan",
                "results_dir": "./results/gan"
            },
            "experiment": {"seed": 42, "resume_training": True}
        }
    
    # Override with command line arguments
    if args.data_root:
        config_dict.setdefault("paths", {})["data_root"] = args.data_root
    if args.checkpoint_dir:
        config_dict.setdefault("paths", {})["checkpoint_dir"] = args.checkpoint_dir
    if args.results_dir:
        config_dict.setdefault("paths", {})["results_dir"] = args.results_dir
    if args.epochs:
        config_dict.setdefault("gan", {})["num_epochs"] = args.epochs
    if args.batch_size:
        config_dict.setdefault("gan", {})["batch_size"] = args.batch_size
    if args.no_resume:
        config_dict.setdefault("experiment", {})["resume_training"] = False
    
    # Create merged configuration
    config = create_gan_config(config_dict)
    
    # Set seed
    seed = args.seed or config.get("seed", 42)
    set_seed(seed)
    config["seed"] = seed
    
    # Validate configuration
    validate_config(config)
    
    # Print configuration summary
    print_config_summary(config)
    
    # Create directories
    create_directories(
        config["checkpoint_dir"], 
        config["results_dir"],
        "./logs"
    )
    
    # Initialize logger
    logger = Logger(name="GANTraining", log_dir="./logs")
    
    # Initialize trainer
    trainer = GANTrainer(config, logger)
    
    if args.generate_only:
        # Only generate data
        print("\nGenerating data using existing model...")
        
        # Load latest checkpoint
        start_epoch = trainer.load_checkpoint()
        if start_epoch == 0:
            print("Error: No trained model found. Please train a model first.")
            return
        
        # Generate and save data
        output_dir = config.get("generated_data_dir", "./data/generated")
        trainer.save_generated_data(output_dir, args.samples_per_class)
        
        print(f"Data generation completed! Check {output_dir}")
        
    else:
        # Full training
        if not args.no_resume and config.get("resume_training", True):
            response = input("\nResume from existing checkpoint if available? (Y/n): ")
            resume = response.lower() not in ["n", "no"]
        else:
            resume = False
        
        if not resume:
            response = input("This will start training from scratch. Continue? (y/N): ")
            if response.lower() not in ["y", "yes"]:
                print("Training cancelled.")
                return
        
        # Load data
        imbalance_ratios = config.get("imbalance_ratios")
        dataloader = trainer.load_data(imbalance_ratios)
        
        # Train the model
        trainer.train(dataloader, resume=resume)
        
        print("\n" + "="*60)
        print("GAN TRAINING COMPLETED!")
        print("="*60)
        print(f"Checkpoints saved to: {config["checkpoint_dir"]}")
        print(f"Results saved to: {config["results_dir"]}")
        
        # Optionally generate data after training
        generate_response = input("\nGenerate augmented data now? (y/N): ")
        if generate_response.lower() in ["y", "yes"]:
            output_dir = config.get("generated_data_dir", "./data/generated")
            samples_per_class = config.get("samples_per_class", 1000)
            trainer.save_generated_data(output_dir, samples_per_class)
            print(f"Generated data saved to: {output_dir}")
        
        print("="*60)

if __name__ == "__main__":
    main()