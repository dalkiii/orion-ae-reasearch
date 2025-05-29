import logging
import os
from datetime import datetime
from typing import Optional


class Logger:
    """Custom logger class for experiment tracking"""
    
    def __init__(self, 
                 name: str = "ScalogramClassification",
                 log_dir: str = "./logs",
                 console_level: int = logging.INFO,
                 file_level: int = logging.DEBUG):
        """
        Initialize logger
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            console_level: Console logging level
            file_level: File logging level
        """
        self.name = name
        self.log_dir = log_dir
        self.console_level = console_level
        self.file_level = file_level
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger with console and file handlers"""
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Create formatters
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%H:%M:%S"
        )
        
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.console_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"experiment_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(self.file_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message: str) -> None:
        """Log critical message"""
        self.logger.critical(message)
    
    def log_experiment_start(self, config) -> None:
        """Log experiment start information"""
        self.info("="*60)
        self.info("EXPERIMENT STARTED")
        self.info("="*60)
        self.info(f"Experiment Name: {config.experiment_name}")
        self.info(f"Number of Runs: {config.num_runs}")
        self.info(f"Random Seed: {config.seed}")
        self.info(f"Data Directory: {config.data.root_dir}")
        self.info(f"Model: {config.model.model_name}")
        self.info(f"Number of Classes: {config.model.num_classes}")
        self.info(f"Number of Epochs: {config.training.num_epochs}")
        self.info(f"Learning Rate: {config.training.learning_rate}")
        self.info(f"Batch Size: {config.data.batch_size}")
        if config.data.imbalance_factors:
            self.info("Class Imbalance Factors:")
            for class_name, factor in config.data.imbalance_factors.items():
                self.info(f"  {class_name}: {factor}")
        self.info("-"*60)
    
    def log_experiment_end(self, avg_results: dict) -> None:
        """Log experiment end information"""
        self.info("="*60)
        self.info("EXPERIMENT COMPLETED")
        self.info("="*60)
        self.info(f"Average Accuracy: {avg_results['avg_accuracy']:.4f} ± {avg_results['std_accuracy']:.4f}")
        self.info(f"Average Macro F1: {avg_results['avg_macro_f1']:.4f} ± {avg_results['std_macro_f1']:.4f}")
        self.info("="*60)
    
    def log_run_start(self, run_id: int, total_runs: int) -> None:
        """Log run start information"""
        self.info(f"\n{'='*20} RUN {run_id + 1}/{total_runs} {'='*20}")
    
    def log_run_end(self, run_id: int, results: dict) -> None:
        """Log run end information"""
        self.info(f"Run {run_id + 1} completed:")
        self.info(f"  Test Accuracy: {results['accuracy']:.4f}")
        self.info(f"  Macro F1 Score: {results['macro_f1']:.4f}")
        self.info("-"*50)


def setup_logging(log_dir: str = "./logs", 
                 log_level: str = "INFO") -> logging.Logger:
    """
    Setup basic logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert string level to logging constant
    level_dict = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    level = level_dict.get(log_level.upper(), logging.INFO)
    
    # Setup basic configuration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"experiment_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)