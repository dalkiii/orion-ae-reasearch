"""
Training utilities for scalogram classification
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from .classifier import create_model
from .utils.logger import Logger
from .utils.config import TrainingConfig


class ClassifierTrainer:
    """Trainer class for scalogram classification"""
    
    def __init__(self, 
                 config: TrainingConfig,
                 logger: Optional[Logger] = None):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or Logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model = create_model(
            model_name=config.model_name,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # Best validation loss for model saving
        self.best_val_loss = float('inf')
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration"""
        if self.config.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100.0 * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': val_loss / (batch_idx + 1),
                    'acc': 100.0 * correct / total
                })
        
        epoch_loss = val_loss / len(val_loader)
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              run_id: int = 0) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            run_id: Run identifier for saving checkpoints
        
        Returns:
            Training history
        """
        self.logger.info(f"Starting training (Run {run_id + 1})")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model: {self.config.model_name}")
        
        # Create checkpoint directory
        checkpoint_dir = os.path.dirname(self.config.checkpoint_path)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Log results
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_path = self.config.checkpoint_path.replace(".pth", f"_run{run_id}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                self.logger.info(f"Saved best model to {checkpoint_path}")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader, run_id: int = 0) -> Dict[str, Any]:
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            run_id: Run identifier
        
        Returns:
            Evaluation results
        """
        # Load best model
        checkpoint_path = self.config.checkpoint_path.replace(".pth", f"_run{run_id}.pth")
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
        
        all_preds = []
        all_labels = []
        
        self.logger.info(f"Evaluating on test set (Run {run_id + 1})")
        
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        class_report = classification_report(all_labels, all_preds, output_dict=True)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'class_reports': class_report,
            'confusion_matrix': conf_matrix,
            'predictions': all_preds,
            'true_labels': all_labels
        }
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Macro F1 Score: {macro_f1:.4f}")
        
        return results
    
    def get_model(self) -> nn.Module:
        """Get the trained model"""
        return self.model
    
    def get_history(self) -> Dict[str, List[float]]:
        """Get training history"""
        return self.history