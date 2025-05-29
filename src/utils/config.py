from dataclasses import dataclass
from typing import Dict, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Data configuration"""
    root_dir: str
    imbalance_factors: Optional[Dict[str, float]] = None
    test_samples_per_class: int = 100
    train_val_split: float = 0.8
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "googlenet"
    num_classes: int = 7
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # adam, adamw, sgd
    checkpoint_path: str = "./ckpt/best_model.pth"
    
    # Model config
    model_name: str = "googlenet"
    num_classes: int = 7
    pretrained: bool = True


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    experiment_name: str = "scalogram_classification"
    num_runs: int = 5
    seed: int = 42
    results_dir: str = "./results"
    checkpoint_dir: str = "./ckpt"
    
    # Sub-configs
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    
    def __post_init__(self):
        """Initialize sub-configs if not provided"""
        if self.data is None:
            self.data = DataConfig(root_dir="")
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()


class ConfigManager:
    """Configuration manager for loading and saving configs"""
    
    @staticmethod
    def load_from_yaml(config_path: str) -> ExperimentConfig:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Create data config
        data_config = DataConfig(**config_dict.get("data", {}))
        
        # Create model config
        model_config = ModelConfig(**config_dict.get("model", {}))
        
        # Create training config
        training_config = TrainingConfig(**config_dict.get("training", {}))
        
        # Create experiment config
        experiment_dict = config_dict.get("experiment", {})
        experiment_config = ExperimentConfig(
            **experiment_dict,
            data=data_config,
            model=model_config,
            training=training_config
        )
        
        return experiment_config
    
    @staticmethod
    def save_to_yaml(config: ExperimentConfig, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            "experiment": {
                "experiment_name": config.experiment_name,
                "num_runs": config.num_runs,
                "seed": config.seed,
                "results_dir": config.results_dir,
                "checkpoint_dir": config.checkpoint_dir
            },
            "data": {
                "root_dir": config.data.root_dir,
                "imbalance_factors": config.data.imbalance_factors,
                "test_samples_per_class": config.data.test_samples_per_class,
                "train_val_split": config.data.train_val_split,
                "batch_size": config.data.batch_size,
                "num_workers": config.data.num_workers
            },
            "model": {
                "model_name": config.model.model_name,
                "num_classes": config.model.num_classes,
                "pretrained": config.model.pretrained,
            },
            "training": {
                "num_epochs": config.training.num_epochs,
                "learning_rate": config.training.learning_rate,
                "weight_decay": config.training.weight_decay,
                "optimizer": config.training.optimizer,
                "checkpoint_path": config.training.checkpoint_path,
                "model_name": config.training.model_name,
                "num_classes": config.training.num_classes,
                "pretrained": config.training.pretrained,
            }
        }
        
        # Create directory if it doesn"t exist
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def create_default_config() -> ExperimentConfig:
        """Create default configuration"""
        return ExperimentConfig(
            experiment_name="acoustic_emission_classification",
            num_runs=5,
            seed=42,
            results_dir="./results",
            checkpoint_dir="./ckpt",
            data=DataConfig(
                root_dir="/path/to/your/data",
                imbalance_factors={
                    "05cNm": 0.005,
                    "10cNm": 0.005,
                    "20cNm": 0.005,
                    "30cNm": 1.0,
                    "40cNm": 1.0,
                    "50cNm": 1.0,
                    "60cNm": 1.0
                },
                test_samples_per_class=100,
                train_val_split=0.8,
                batch_size=32,
                num_workers=4
            ),
            model=ModelConfig(
                model_name="googlenet",
                num_classes=7,
                pretrained=True,
            ),
            training=TrainingConfig(
                num_epochs=10,
                learning_rate=1e-3,
                weight_decay=1e-4,
                optimizer="adamw",
                checkpoint_path="./ckpt/best_model.pth",
                model_name="googlenet",
                num_classes=7,
                pretrained=True,
            )
        )