import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class IMBSpecGANGenerator(nn.Module):
    """Generator for IMBSpecGAN with conditional generation"""
    
    def __init__(self, input_dim: int, d: int = 64):
        """
        Initialize generator
        
        Args:
            input_dim: Input dimension (latent_dim + num_classes)
            d: Base number of filters
        """
        super(IMBSpecGANGenerator, self).__init__()
        self.d = d
        
        # Initial linear layer
        self.fc = nn.Linear(input_dim, 7 * 7 * 16 * d, bias=False)
        self.bn0 = nn.BatchNorm1d(7 * 7 * 16 * d)
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(16 * d, 8 * d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8 * d)
        
        self.deconv2 = nn.ConvTranspose2d(8 * d, 4 * d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * d)
        
        self.deconv3 = nn.ConvTranspose2d(4 * d, 2 * d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(2 * d)
        
        self.deconv4 = nn.ConvTranspose2d(2 * d, d, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(d)
        
        self.deconv5 = nn.ConvTranspose2d(d, 1, kernel_size=4, stride=2, padding=1, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Linear layer with batch norm and ReLU
        x = self.fc(x)
        x = self.bn0(x)
        x = F.relu(x)
        
        # Reshape to feature maps
        x = x.view(-1, 16 * self.d, 7, 7)
        
        # Deconvolutional layers
        x = self.deconv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.deconv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.deconv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.deconv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.deconv5(x)
        x = torch.tanh(x)
        
        return x


class IMBSpecGANDiscriminator(nn.Module):
    """Discriminator for IMBSpecGAN with auxiliary classifier"""
    
    def __init__(self, input_channels: int = 1, num_classes: int = 7, d: int = 64, noise_std: float = 0.02):
        """
        Initialize discriminator
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of classes for auxiliary classifier
            d: Base number of filters
            noise_std: Standard deviation for input noise
        """
        super(IMBSpecGANDiscriminator, self).__init__()
        self.d = d
        self.noise_std = noise_std
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, d, kernel_size=5, stride=2, padding=2, bias=False)
        
        self.conv2 = nn.Conv2d(d, 2 * d, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * d)
        
        self.conv3 = nn.Conv2d(2 * d, 4 * d, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn3 = nn.BatchNorm2d(4 * d)
        
        self.conv4 = nn.Conv2d(4 * d, 8 * d, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn4 = nn.BatchNorm2d(8 * d)
        
        self.conv5 = nn.Conv2d(8 * d, 16 * d, kernel_size=5, stride=2, padding=2, bias=False)
        self.bn5 = nn.BatchNorm2d(16 * d)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        
        # Output layers
        self.fc = nn.Linear(16 * d * 7 * 7, 1, bias=False)
        self.aux_fc = nn.Linear(16 * d * 7 * 7, num_classes, bias=False)
        self.embed = nn.Embedding(num_classes, 16 * d * 7 * 7)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input images
            labels: Class labels
        
        Returns:
            Tuple of (discriminator_logits, class_logits)
        """
        # Add noise to input
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        
        # Convolutional layers
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.bn2(self.conv2(x)))
        x = self.lrelu(self.bn3(self.conv3(x)))
        x = self.lrelu(self.bn4(self.conv4(x)))
        x = self.lrelu(self.bn5(self.conv5(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Discriminator output
        img_logit = self.fc(x)
        
        # Conditional discriminator with label embedding
        label_embed = self.embed(labels)
        proj = torch.sum(x * label_embed, dim=1, keepdim=True)
        final_logit = img_logit + proj
        
        # Auxiliary classifier output
        class_logits = self.aux_fc(x)
        
        return final_logit, class_logits


def sample_latent(batch_size: int, 
                 latent_dim: int, 
                 num_classes: int, 
                 device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample latent vectors and random labels
    
    Args:
        batch_size: Batch size
        latent_dim: Latent dimension
        num_classes: Number of classes
        device: Device to create tensors on
    
    Returns:
        Tuple of (latent_input, labels)
    """
    # Sample random noise
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # Sample random labels
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    
    # Create one-hot encoding
    one_hot = torch.zeros(batch_size, num_classes, device=device)
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    # Concatenate noise and one-hot labels
    latent_input = torch.cat([z, one_hot], dim=1)
    
    return latent_input, labels


def compute_gradient_penalty(discriminator: nn.Module,
                           real_samples: torch.Tensor,
                           fake_samples: torch.Tensor,
                           labels: torch.Tensor,
                           device: torch.device,
                           lambda_gp: float = 10.0) -> torch.Tensor:
    """
    Compute gradient penalty for improved WGAN training
    
    Args:
        discriminator: Discriminator model
        real_samples: Real samples
        fake_samples: Generated samples
        labels: Class labels
        device: Device
        lambda_gp: Gradient penalty coefficient
    
    Returns:
        Gradient penalty loss
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation factor
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Interpolated samples
    interpolated = epsilon * real_samples + (1 - epsilon) * fake_samples
    interpolated.requires_grad_(True)
    
    # Get discriminator output for interpolated samples
    d_interpolated, _ = discriminator(interpolated, labels)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((grad_norm - 1) ** 2).mean()
    
    return penalty


def initialize_weights(model: nn.Module) -> None:
    """Initialize model weights"""
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)