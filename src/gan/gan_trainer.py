"""
GAN trainer for acoustic emission data augmentation
"""
import os
import glob
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, Optional, Any
import numpy as np

from .models import (
    IMBSpecGANGenerator, 
    IMBSpecGANDiscriminator, 
    sample_latent, 
    compute_gradient_penalty,
    initialize_weights
)
from .gan_dataset import create_gan_dataloader
from ..utils.logger import Logger
from ..utils.utils import set_seed, create_directories


class GANTrainer:
    """Trainer for IMBSpecGAN"""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[Logger] = None):
        """
        Initialize GAN trainer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or Logger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training parameters
        self.latent_dim = config.get('latent_dim', 200)
        self.num_classes = config.get('num_classes', 7)
        self.batch_size = config.get('batch_size', 64)
        self.num_epochs = config.get('num_epochs', 2000)
        self.d_updates = config.get('d_updates', 1)
        self.d_model_dim = config.get('d_model_dim', 128)
        
        # Optimizer parameters
        self.g_lr = config.get('g_lr', 1e-5)
        self.d_lr = config.get('d_lr', 1e-6)
        self.beta1 = config.get('beta1', 0.5)
        self.beta2 = config.get('beta2', 0.9)
        
        # Loss parameters
        self.lambda_gp = config.get('lambda_gp', 15.0)
        
        # Paths
        self.data_root = config.get('data_root', '')
        self.checkpoint_dir = config.get('checkpoint_dir', './checkpoints')
        self.results_dir = config.get('results_dir', './results')
        
        # Create directories
        create_directories(self.checkpoint_dir, self.results_dir)
        
        # Initialize models
        self._initialize_models()
        
        # Training history
        self.history = {
            'g_loss': [],
            'd_loss': [],
            'epoch': []
        }
    
    def _initialize_models(self) -> None:
        """Initialize generator and discriminator"""
        input_dim = self.latent_dim + self.num_classes
        
        self.generator = IMBSpecGANGenerator(
            input_dim=input_dim,
            d=self.d_model_dim
        ).to(self.device)
        
        self.discriminator = IMBSpecGANDiscriminator(
            input_channels=1,
            num_classes=self.num_classes,
            d=self.d_model_dim
        ).to(self.device)
        
        # Initialize weights
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)
        
        # Initialize optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=self.g_lr,
            betas=(self.beta1, self.beta2)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=self.d_lr,
            betas=(self.beta1, self.beta2)
        )
        
        self.logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
        self.logger.info(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters()):,}")
    
    def load_data(self, imbalance_ratios: Optional[Dict[str, float]] = None) -> DataLoader:
        """
        Load and prepare data using simplified approach
        
        Args:
            imbalance_ratios: Class imbalance ratios
        
        Returns:
            DataLoader for training
        """

        self.logger.info("Loading GAN training data...")
        
        # Create dataloader directly
        dataloader = create_gan_dataloader(
            root_dir=self.data_root,
            imbalance_ratios=imbalance_ratios,
            batch_size=self.batch_size,
            num_workers=4,
            seed=self.config.get('seed', 42)
        )
        
        self.logger.info(f"Created dataloader with batch_size={self.batch_size}")
        
        return dataloader
    
    def train_step_discriminator(self, real_imgs: torch.Tensor, real_labels: torch.Tensor) -> float:
        """
        Single discriminator training step
        
        Args:
            real_imgs: Real images
            real_labels: Real labels
        
        Returns:
            Discriminator loss
        """
        # Zero gradients
        self.optimizer_d.zero_grad()
        
        # Generate fake images
        latent_input, sampled_labels = sample_latent(
            real_imgs.size(0), self.latent_dim, self.num_classes, self.device
        )
        fake_imgs = self.generator(latent_input)
        
        # Discriminator outputs
        d_real, aux_real = self.discriminator(real_imgs, real_labels)
        d_fake, _ = self.discriminator(fake_imgs.detach(), sampled_labels)
        
        # Gradient penalty
        gp = compute_gradient_penalty(
            self.discriminator, real_imgs, fake_imgs.detach(), 
            real_labels, self.device, self.lambda_gp
        )
        
        # Adversarial loss (WGAN-GP)
        d_loss_adv = d_fake.mean() - d_real.mean() + gp
        
        # Auxiliary classifier loss
        aux_loss = F.cross_entropy(aux_real, real_labels)
        
        # Total discriminator loss
        d_loss = d_loss_adv + aux_loss
        
        # Backward pass
        d_loss.backward()
        self.optimizer_d.step()
        
        return d_loss.item()
    
    def train_step_generator(self, batch_size: int) -> float:
        """
        Single generator training step
        
        Args:
            batch_size: Batch size
        
        Returns:
            Generator loss
        """
        # Zero gradients
        self.optimizer_g.zero_grad()
        
        # Generate fake images
        latent_input, sampled_labels = sample_latent(
            batch_size, self.latent_dim, self.num_classes, self.device
        )
        fake_imgs = self.generator(latent_input)
        
        # Discriminator outputs
        g_adv, aux_fake = self.discriminator(fake_imgs, sampled_labels)
        
        # Adversarial loss
        g_loss_adv = -g_adv.mean()
        
        # Auxiliary classifier loss
        g_aux_loss = F.cross_entropy(aux_fake, sampled_labels)
        
        # Total generator loss
        g_loss = g_loss_adv + g_aux_loss
        
        # Backward pass
        g_loss.backward()
        self.optimizer_g.step()
        
        return g_loss.item()
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Tuple of (average_d_loss, average_g_loss)
        """
        self.generator.train()
        self.discriminator.train()
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        count = 0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for real_imgs, real_labels in progress_bar:
            real_imgs = real_imgs.to(self.device)
            real_labels = real_labels.to(self.device)
            
            # Train discriminator
            d_loss = 0.0
            for _ in range(self.d_updates):
                d_loss += self.train_step_discriminator(real_imgs, real_labels)
            d_loss /= self.d_updates
            
            # Train generator
            g_loss = self.train_step_generator(real_imgs.size(0))
            
            # Update statistics
            running_d_loss += d_loss
            running_g_loss += g_loss
            count += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'd_loss': f'{d_loss:.4f}',
                'g_loss': f'{g_loss:.4f}'
            })
        
        avg_d_loss = running_d_loss / count
        avg_g_loss = running_g_loss / count
        
        return avg_d_loss, avg_g_loss
    
    def generate_samples(self, num_samples: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sample images
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (generated_images, labels)
        """
        self.generator.eval()
        
        with torch.no_grad():
            latent_input, sampled_labels = sample_latent(
                num_samples, self.latent_dim, self.num_classes, self.device
            )
            fake_samples = self.generator(latent_input)
        
        return fake_samples.cpu(), sampled_labels.cpu()
    
    def visualize_progress(self, epoch: int, num_samples: int = 8) -> None:
        """
        Visualize training progress
        
        Args:
            epoch: Current epoch
            num_samples: Number of samples to visualize
        """
        fake_samples, sampled_labels = self.generate_samples(num_samples)
        sampled_labels = sampled_labels.numpy()
        
        # Create visualization
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        
        for i, ax in enumerate(axs.flatten()):
            if i < len(fake_samples):
                img = fake_samples[i].squeeze(0)
                ax.imshow(img, cmap='gray', vmin=-1, vmax=1)
                ax.set_title(f"Class: {sampled_labels[i]}")
            ax.axis("off")
        
        plt.suptitle(f"Generated Samples - Epoch {epoch}", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        save_path = os.path.join(self.results_dir, f"samples_epoch_{epoch:04d}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to {save_path}")
    
    def save_checkpoint(self, epoch: int) -> None:
        """
        Save model checkpoints
        
        Args:
            epoch: Current epoch
        """
        # Save generator
        gen_path = os.path.join(self.checkpoint_dir, f"netG_epoch_{epoch}.pth")
        torch.save(self.generator.state_dict(), gen_path)
        
        # Save discriminator
        disc_path = os.path.join(self.checkpoint_dir, f"netD_epoch_{epoch}.pth")
        torch.save(self.discriminator.state_dict(), disc_path)
        
        # Save training state
        state_path = os.path.join(self.checkpoint_dir, f"training_state_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'history': self.history,
            'config': self.config
        }, state_path)
        
        self.logger.info(f"Checkpoints saved: {gen_path}, {disc_path}, {state_path}")
    
    def load_checkpoint(self) -> int:
        """
        Load latest checkpoint
        
        Returns:
            Starting epoch number
        """
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, "netG_epoch_*.pth"))
        
        if not checkpoint_files:
            self.logger.info("No checkpoint found. Training from scratch.")
            return 0
        
        # Find latest checkpoint
        latest_epoch = 0
        latest_checkpoint = None
        pattern = re.compile(r"netG_epoch_(\d+)\.pth")
        
        for file in checkpoint_files:
            match = pattern.search(file)
            if match:
                epoch_num = int(match.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_checkpoint = file
        
        if latest_checkpoint is not None:
            self.logger.info(f"Loading generator checkpoint from {latest_checkpoint}")
            self.generator.load_state_dict(torch.load(latest_checkpoint, map_location=self.device))
            
            # Load discriminator
            netD_checkpoint = os.path.join(self.checkpoint_dir, f"netD_epoch_{latest_epoch}.pth")
            if os.path.exists(netD_checkpoint):
                self.logger.info(f"Loading discriminator checkpoint from {netD_checkpoint}")
                self.discriminator.load_state_dict(torch.load(netD_checkpoint, map_location=self.device))
            
            # Load training state
            state_checkpoint = os.path.join(self.checkpoint_dir, f"training_state_epoch_{latest_epoch}.pth")
            if os.path.exists(state_checkpoint):
                state = torch.load(state_checkpoint, map_location=self.device)
                self.optimizer_g.load_state_dict(state['optimizer_g_state_dict'])
                self.optimizer_d.load_state_dict(state['optimizer_d_state_dict'])
                self.history = state.get('history', self.history)
                self.logger.info("Loaded training state")
        
        return latest_epoch
    
    def plot_training_curves(self) -> None:
        """Plot and save training curves"""
        if not self.history['epoch']:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = self.history['epoch']
        
        # Loss curves
        ax1.plot(epochs, self.history['d_loss'], label='Discriminator Loss', color='blue')
        ax1.plot(epochs, self.history['g_loss'], label='Generator Loss', color='red')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Losses')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss ratio
        if len(self.history['d_loss']) > 0 and len(self.history['g_loss']) > 0:
            ratio = [g/d if d != 0 else 0 for g, d in zip(self.history['g_loss'], self.history['d_loss'])]
            ax2.plot(epochs, ratio, label='G_loss / D_loss', color='green')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Ratio')
            ax2.set_title('Generator/Discriminator Loss Ratio')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.results_dir, "training_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved to {save_path}")
    
    def train(self, dataloader: DataLoader, resume: bool = True) -> None:
        """
        Main training loop
        
        Args:
            dataloader: Training data loader
            resume: Whether to resume from checkpoint
        """
        self.logger.info("Starting GAN training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Number of epochs: {self.num_epochs}")
        
        # Load checkpoint if resuming
        start_epoch = 0
        if resume:
            start_epoch = self.load_checkpoint()
        
        self.logger.info(f"Starting from epoch {start_epoch + 1}")
        
        # Training loop
        for epoch in range(start_epoch, self.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train for one epoch
            avg_d_loss, avg_g_loss = self.train_epoch(dataloader)
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['d_loss'].append(avg_d_loss)
            self.history['g_loss'].append(avg_g_loss)
            
            # Log progress
            self.logger.info(f"Epoch {epoch + 1}: D_loss = {avg_d_loss:.4f}, G_loss = {avg_g_loss:.4f}")
            
            # Visualize progress
            if (epoch + 1) % 100 == 0:
                self.visualize_progress(epoch + 1)
            
            # Save checkpoints
            if (epoch + 1) % 30 == 0:
                self.save_checkpoint(epoch + 1)
                self.plot_training_curves()
        
        # Final checkpoint and visualization
        self.save_checkpoint(self.num_epochs)
        self.plot_training_curves()
        self.visualize_progress(self.num_epochs)
        
        self.logger.info("Training completed!")
    
    def generate_augmented_data(self, 
                              target_class: int, 
                              num_samples: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate augmented data for a specific class
        
        Args:
            target_class: Target class label
            num_samples: Number of samples to generate
        
        Returns:
            Tuple of (generated_images, labels)
        """
        self.generator.eval()
        
        generated_images = []
        labels = []
        
        with torch.no_grad():
            # Generate in batches to avoid memory issues
            batch_size = min(self.batch_size, num_samples)
            num_batches = (num_samples + batch_size - 1) // batch_size
            
            for _ in range(num_batches):
                current_batch_size = min(batch_size, num_samples - len(generated_images))
                
                # Create latent input for specific class
                z = torch.randn(current_batch_size, self.latent_dim, device=self.device)
                class_labels = torch.full((current_batch_size,), target_class, device=self.device)
                one_hot = torch.zeros(current_batch_size, self.num_classes, device=self.device)
                one_hot.scatter_(1, class_labels.unsqueeze(1), 1)
                latent_input = torch.cat([z, one_hot], dim=1)
                
                # Generate samples
                fake_samples = self.generator(latent_input)
                generated_images.append(fake_samples.cpu())
                labels.append(class_labels.cpu())
        
        all_images = torch.cat(generated_images, dim=0)[:num_samples]
        all_labels = torch.cat(labels, dim=0)[:num_samples]
        
        return all_images, all_labels
    
    def save_generated_data(self, output_dir: str, samples_per_class: int = 1000) -> None:
        """
        Save generated data for each class
        
        Args:
            output_dir: Output directory
            samples_per_class: Number of samples to generate per class
        """
        create_directories(output_dir)
        
        self.logger.info(f"Generating {samples_per_class} samples per class...")
        
        for class_id in range(self.num_classes):
            self.logger.info(f"Generating samples for class {class_id}...")
            
            # Generate samples
            images, labels = self.generate_augmented_data(class_id, samples_per_class)
            
            # Convert to numpy
            images_np = images.numpy()
            labels_np = labels.numpy()
            
            # Save as numpy file
            class_name = f"{class_id:02d}cNm"  # e.g., "05cNm"
            class_dir = os.path.join(output_dir, class_name)
            create_directories(class_dir)
            
            output_file = os.path.join(class_dir, f"generated_{class_name}_{samples_per_class}.npy")
            np.save(output_file, images_np)
            
            self.logger.info(f"Saved {len(images_np)} samples to {output_file}")
        
        self.logger.info(f"All generated data saved to {output_dir}")
    
    def get_generator(self) -> nn.Module:
        """Get the trained generator"""
        return self.generator
    
    def get_discriminator(self) -> nn.Module:
        """Get the trained discriminator"""
        return self.discriminator
    
    def get_history(self) -> Dict[str, list]:
        """Get training history"""
        return self.history