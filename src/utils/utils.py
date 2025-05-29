import random
import numpy as np
import torch
import os
from typing import Dict, Any, List
import json


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(*dirs: str) -> None:
    """
    Create directories if they don't exist
    
    Args:
        *dirs: Directory paths to create
    """
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human readable format
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)} minutes {remaining_seconds:.2f} seconds"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        remaining_seconds = seconds % 60
        return f"{int(hours)} hours {int(remaining_minutes)} minutes {remaining_seconds:.2f} seconds"


def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, else CPU)
    
    Returns:
        PyTorch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def calculate_class_weights(class_counts: Dict[int, int]) -> torch.Tensor:
    """
    Calculate class weights for handling class imbalance
    
    Args:
        class_counts: Dictionary mapping class indices to counts
    
    Returns:
        Tensor of class weights
    """
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    
    weights = []
    for class_idx in sorted(class_counts.keys()):
        weight = total_samples / (num_classes * class_counts[class_idx])
        weights.append(weight)
    
    return torch.FloatTensor(weights)


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """
    Get current learning rate from optimizer
    
    Args:
        optimizer: PyTorch optimizer
    
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_learning_rate(optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    """
    Update learning rate in optimizer
    
    Args:
        optimizer: PyTorch optimizer
        new_lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop
        
        Args:
            val_loss: Current validation loss
            model: PyTorch model
        
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        return False