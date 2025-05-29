import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class GoogleNetClassifier(nn.Module):
    """GoogleNet-based classifier for scalogram data"""
    
    def __init__(self, num_classes: int = 7, pretrained: bool = True):
        """
        Initialize GoogleNet classifier
        
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super(GoogleNetClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.backbone = models.googlenet(pretrained=pretrained)
        
        # Replace the final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.backbone(x)


def create_model(model_name: str = "googlenet", 
                num_classes: int = 7, 
                pretrained: bool = True) -> nn.Module:
    """
    Factory function to create models
    
    Args:
        model_name: Name of the model ('googlenet')
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    if model_name == "googlenet":
        return GoogleNetClassifier(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def load_model(checkpoint_path: str, 
              model_name: str = "googlenet", 
              num_classes: int = 7,
              device: Optional[torch.device] = None) -> nn.Module:
    """
    Load model from checkpoint
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Name of the model
        num_classes: Number of output classes
        device: Device to load model on
    
    Returns:
        Loaded PyTorch model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model