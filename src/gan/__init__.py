"""
GAN modules for acoustic emission data augmentation
"""
from .models import (
    IMBSpecGANGenerator, 
    IMBSpecGANDiscriminator, 
    sample_latent, 
    compute_gradient_penalty,
    initialize_weights
)
from .gan_dataset import create_gan_dataloader
from .gan_trainer import GANTrainer

__all__ = [
    'IMBSpecGANGenerator', 
    'IMBSpecGANDiscriminator', 
    'sample_latent', 
    'compute_gradient_penalty',
    'initialize_weights',
    'create_gan_dataloader',
    'GANTrainer'
]