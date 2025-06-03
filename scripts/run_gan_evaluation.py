import sys
import os
import yaml
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.gan.models import IMBSpecGANGenerator, GANPerformanceAnalyzer
from src.utils.logger import Logger
from src.utils.utils import set_seed, create_directories


def load_evaluation_config(config_path: str) -> dict:
    """Load evaluation configuration from YAML file"""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_evaluation_config(config_dict: dict) -> dict:
    """Create evaluation configuration from dictionary"""
    gan_config = config_dict.get("gan", {})
    paths_config = config_dict.get("paths", {})
    evaluation_config = config_dict.get("evaluation", {})
    experiment_config = config_dict.get("experiment", {})
    
    # Merge all configurations
    merged_config = {
        **gan_config,
        **paths_config,
        **evaluation_config,
        **experiment_config
    }
    
    return merged_config


def validate_config(config: dict) -> None:
    """Validate configuration"""
    required_keys = ["checkpoint_dir", "results_dir"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Check if checkpoint directory exists
    if not os.path.exists(config["checkpoint_dir"]):
        raise ValueError(f"Checkpoint directory does not exist: {config['checkpoint_dir']}")


def print_config_summary(config: dict) -> None:
    """Print configuration summary"""
    print("="*60)
    print("GAN EVALUATION CONFIGURATION")
    print("="*60)
    print(f"Checkpoint directory: {config.get('checkpoint_dir', 'N/A')}")
    print(f"Results directory: {config.get('results_dir', 'N/A')}")
    print(f"Latent dimension: {config.get('latent_dim', 200)}")
    print(f"Number of classes: {config.get('num_classes', 7)}")
    print(f"Model dimension: {config.get('d_model_dim', 128)}")
    print(f"Samples per class: {config.get('num_samples_per_class', 100)}")
    print(f"Checkpoint interval: {config.get('checkpoint_interval', 30)}")
    print(f"t-SNE perplexity: {config.get('tsne_perplexity', 30)}")
    print(f"Random seed: {config.get('seed', 42)}")
    
    # Print analysis options
    analysis_options = config.get("analysis_options", {})
    if analysis_options:
        print(f"Analysis options:")
        for option, value in analysis_options.items():
            print(f"  {option}: {value}")
    
    print("="*60)


def create_default_config() -> dict:
    """Create default evaluation configuration"""
    return {
        "gan": {
            "latent_dim": 200,
            "num_classes": 7,
            "d_model_dim": 128
        },
        "paths": {
            "checkpoint_dir": "./checkpoints/gan",
            "results_dir": "./results/gan/evaluation"
        },
        "evaluation": {
            "num_samples_per_class": 100,
            "checkpoint_interval": 30,
            "tsne_perplexity": 30,
            "analysis_options": {
                "save_tsne_plots": True,
                "save_individual_epochs": True,
                "create_animations": False
            }
        },
        "experiment": {
            "seed": 42
        }
    }


def run_comprehensive_analysis(config: dict, logger: Logger) -> None:
    """Run comprehensive GAN performance analysis"""
    logger.info("Starting comprehensive GAN performance analysis...")
    
    # Initialize generator
    input_dim = config.get("latent_dim", 200) + config.get("num_classes", 7)
    generator = IMBSpecGANGenerator(
        input_dim=input_dim,
        d=config.get("d_model_dim", 128)
    )
    
    # Initialize analyzer
    analyzer = GANPerformanceAnalyzer(
        generator=generator,
        checkpoint_dir=config["checkpoint_dir"],
        latent_dim=config.get("latent_dim", 200),
        num_classes=config.get("num_classes", 7)
    )
    
    # Set analysis parameters
    analyzer.num_samples_per_class = config.get("num_samples_per_class", 100)
    
    # Run complete analysis
    silhouette_scores, dispersion_by_epoch = analyzer.run_complete_analysis(
        save_dir=config["results_dir"]
    )
    
    # Additional visualizations based on config
    analysis_options = config.get("analysis_options", {})
    
    if analysis_options.get("save_individual_epochs", True):
        logger.info("Creating individual epoch visualizations...")
        checkpoint_epochs = analyzer.checkpoint_epochs
        
        # Create visualizations for key epochs
        key_epochs = []
        if checkpoint_epochs:
            key_epochs.append(min(checkpoint_epochs))  # First epoch
            key_epochs.append(max(checkpoint_epochs))  # Last epoch
            
            # Best performing epoch
            if silhouette_scores:
                best_epoch = max(silhouette_scores, key=silhouette_scores.get)
                if best_epoch not in key_epochs:
                    key_epochs.append(best_epoch)
            
            # Middle epochs for progression
            if len(checkpoint_epochs) > 3:
                mid_idx = len(checkpoint_epochs) // 2
                mid_epoch = checkpoint_epochs[mid_idx]
                if mid_epoch not in key_epochs:
                    key_epochs.append(mid_epoch)
        
        if key_epochs:
            analyzer.visualize_tsne_by_epoch(
                epochs_to_plot=sorted(key_epochs),
                save_dir=os.path.join(config["results_dir"], "epoch_analysis/")
            )
    
    # Print summary results
    logger.info("="*50)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*50)
    
    if silhouette_scores:
        best_epoch = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_epoch]
        worst_epoch = min(silhouette_scores, key=silhouette_scores.get)
        worst_score = silhouette_scores[worst_epoch]
        avg_score = sum(silhouette_scores.values()) / len(silhouette_scores)
        
        logger.info(f"Silhouette Score Analysis:")
        logger.info(f"  Best epoch: {best_epoch} (Score: {best_score:.4f})")
        logger.info(f"  Worst epoch: {worst_epoch} (Score: {worst_score:.4f})")
        logger.info(f"  Average score: {avg_score:.4f}")
        logger.info(f"  Score improvement: {best_score - worst_score:.4f}")
    
    if dispersion_by_epoch:
        logger.info(f"Class Dispersion Analysis:")
        class_names = analyzer.class_names
        
        for class_idx in range(config.get("num_classes", 7)):
            class_dispersions = [disp[class_idx] for disp in dispersion_by_epoch.values()]
            avg_dispersion = sum(class_dispersions) / len(class_dispersions)
            logger.info(f"  {class_names[class_idx]}: Avg dispersion = {avg_dispersion:.4f}")
    
    logger.info("="*50)
    
    return silhouette_scores, dispersion_by_epoch


def run_specific_epoch_analysis(config: dict, epochs: list, logger: Logger) -> None:
    """Run analysis for specific epochs only"""
    logger.info(f"Running analysis for specific epochs: {epochs}")
    
    # Initialize generator
    input_dim = config.get("latent_dim", 200) + config.get("num_classes", 7)
    generator = IMBSpecGANGenerator(
        input_dim=input_dim,
        d=config.get("d_model_dim", 128)
    )
    
    # Initialize analyzer
    analyzer = GANPerformanceAnalyzer(
        generator=generator,
        checkpoint_dir=config["checkpoint_dir"],
        latent_dim=config.get("latent_dim", 200),
        num_classes=config.get("num_classes", 7)
    )
    
    analyzer.num_samples_per_class = config.get("num_samples_per_class", 100)
    analyzer.checkpoint_epochs = epochs
    
    # Run analysis for specific epochs
    analyzer.apply_tsne_analysis(
        perplexity=config.get("tsne_perplexity", 30),
        random_state=config.get("seed", 42)
    )
    
    # Calculate metrics
    silhouette_scores = analyzer.calculate_silhouette_scores()
    dispersion_by_epoch = analyzer.calculate_class_dispersion()
    
    # Create visualizations
    results_dir = config["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    
    analyzer.visualize_silhouette_scores(
        silhouette_scores, 
        os.path.join(results_dir, "specific_epochs_silhouette.png")
    )
    
    analyzer.visualize_class_dispersion(
        dispersion_by_epoch,
        os.path.join(results_dir, "specific_epochs_dispersion.png")
    )
    
    analyzer.visualize_tsne_by_epoch(
        epochs_to_plot=epochs,
        save_dir=os.path.join(results_dir, "specific_epochs_tsne/")
    )
    
    logger.info(f"Specific epoch analysis completed. Results saved to: {results_dir}")


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate GAN performance and quality")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/gan_evaluation_config.yaml",
        help="Path to evaluation configuration file"
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
        nargs="+",
        type=int,
        help="Specific epochs to analyze (space-separated list)"
    )
    parser.add_argument(
        "--samples-per-class", 
        type=int, 
        help="Number of samples per class for analysis (overrides config)"
    )
    parser.add_argument(
        "--tsne-perplexity", 
        type=int, 
        help="t-SNE perplexity parameter (overrides config)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick analysis with fewer samples"
    )
    parser.add_argument(
        "--no-individual-plots", 
        action="store_true",
        help="Skip individual epoch t-SNE plots"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config_dict = load_evaluation_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        else:
            print(f"Configuration file not found: {args.config}")
            print("Using default configuration...")
            config_dict = create_default_config()
        
        # Override with command line arguments
        if args.checkpoint_dir:
            config_dict.setdefault("paths", {})["checkpoint_dir"] = args.checkpoint_dir
        if args.results_dir:
            config_dict.setdefault("paths", {})["results_dir"] = args.results_dir
        if args.samples_per_class:
            config_dict.setdefault("evaluation", {})["num_samples_per_class"] = args.samples_per_class
        if args.tsne_perplexity:
            config_dict.setdefault("evaluation", {})["tsne_perplexity"] = args.tsne_perplexity
        if args.quick:
            config_dict.setdefault("evaluation", {})["num_samples_per_class"] = 50
        if args.no_individual_plots:
            config_dict.setdefault("evaluation", {}).setdefault("analysis_options", {})["save_individual_epochs"] = False
        
        # Create merged configuration
        config = create_evaluation_config(config_dict)
        
        # Set seed
        seed = args.seed or config.get("seed", 42)
        set_seed(seed)
        config["seed"] = seed
        
        # Validate configuration
        validate_config(config)
        
        # Print configuration summary
        print_config_summary(config)
        
        # Create directories
        create_directories(config["results_dir"], "./logs")
        
        # Initialize logger
        logger = Logger(name="GANEvaluation", log_dir="./logs")
        
        if args.epochs:
            # Run analysis for specific epochs only
            run_specific_epoch_analysis(config, args.epochs, logger)
            
            print("\n" + "="*60)
            print("SPECIFIC EPOCH ANALYSIS COMPLETED!")
            print("="*60)
            print(f"Analyzed epochs: {args.epochs}")
            print(f"Results saved to: {config['results_dir']}")
            
        else:
            # Run comprehensive analysis
            silhouette_scores, dispersion_by_epoch = run_comprehensive_analysis(config, logger)
            
            print("\n" + "="*60)
            print("COMPREHENSIVE EVALUATION COMPLETED!")
            print("="*60)
            print(f"Analyzed {len(silhouette_scores)} checkpoints")
            print(f"Results saved to: {config['results_dir']}")
            
            if silhouette_scores:
                best_epoch = max(silhouette_scores, key=silhouette_scores.get)
                best_score = silhouette_scores[best_epoch]
                print(f"Best performing model: Epoch {best_epoch} (Silhouette Score: {best_score:.4f})")
        
        print("="*60)
        print("Generated files:")
        print("- silhouette_scores.png: Class separation quality over training")
        print("- class_dispersion.png: Intra-class variance analysis")
        print("- tsne/: t-SNE visualizations for key epochs")
        if not args.no_individual_plots:
            print("- epoch_analysis/: Individual epoch analysis")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()