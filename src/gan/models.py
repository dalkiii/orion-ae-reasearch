import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from typing import Tuple, Dict, List, Optional


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


class GANPerformanceAnalyzer:
    """
    Performance analyzer for GAN models with t-SNE and clustering analysis
    """
    
    def __init__(self, 
                 generator: IMBSpecGANGenerator,
                 checkpoint_dir: str,
                 latent_dim: int = 200,
                 num_classes: int = 7,
                 device: Optional[torch.device] = None):
        """
        Initialize the analyzer
        
        Args:
            generator: Generator model instance
            checkpoint_dir: Directory containing model checkpoints
            latent_dim: Latent dimension
            num_classes: Number of classes
            device: PyTorch device
        """
        self.generator = generator
        self.checkpoint_dir = checkpoint_dir
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Analysis configuration
        self.num_samples_per_class = 100
        self.class_names = ['05cNm', '10cNm', '20cNm', '30cNm', '40cNm', '50cNm', '60cNm']
        
        # Results storage
        self.tsne_results = {}
        self.all_labels = []
        self.checkpoint_epochs = []
        
        # Move generator to device
        self.generator = self.generator.to(self.device)
        self.generator.eval()
    
    def find_checkpoint_epochs(self, interval: int = 30) -> List[int]:
        """
        Find available checkpoint epochs from the checkpoint folder
        
        Args:
            interval: Epoch interval to select checkpoints
            
        Returns:
            Sorted list of checkpoint epochs
        """
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError(f"Checkpoint directory not found: {self.checkpoint_dir}")
        
        checkpoint_files = os.listdir(self.checkpoint_dir)
        checkpoint_epochs = []
        
        for filename in checkpoint_files:
            match = re.match(r"netG_epoch_(\d+)\.pth", filename)
            if match:
                epoch = int(match.group(1))
                if epoch % interval == 0:
                    checkpoint_epochs.append(epoch)
        
        self.checkpoint_epochs = sorted(checkpoint_epochs)
        print(f"Found checkpoint epochs: {self.checkpoint_epochs}")
        return self.checkpoint_epochs
    
    def generate_images_for_epoch(self, epoch: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate images for all classes using a specific epoch checkpoint
        
        Args:
            epoch: Epoch number of the checkpoint
            
        Returns:
            Tuple of (generated_images_array, labels_array)
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, f"netG_epoch_{epoch}.pth")
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        self.generator.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.generator.eval()
        
        epoch_images = []
        epoch_labels = []
        
        # Generate images for each class
        for class_idx in range(self.num_classes):
            # Generate random noise
            z = torch.randn(self.num_samples_per_class, self.latent_dim, device=self.device)
            
            # Create class labels and one-hot encoding
            labels = torch.full((self.num_samples_per_class,), class_idx, device=self.device)
            one_hot = torch.zeros(self.num_samples_per_class, self.num_classes, device=self.device)
            one_hot.scatter_(1, labels.unsqueeze(1), 1)
            
            # Combine noise and class information
            latent_input = torch.cat([z, one_hot], dim=1)
            
            # Generate images
            with torch.no_grad():
                fake_images = self.generator(latent_input)
                # Flatten images for analysis
                flattened_images = fake_images.view(fake_images.size(0), -1).cpu().numpy()
                epoch_images.append(flattened_images)
                epoch_labels.extend([class_idx] * self.num_samples_per_class)
        
        # Combine all images and labels
        epoch_images = np.vstack(epoch_images)
        epoch_labels = np.array(epoch_labels)
        
        return epoch_images, epoch_labels
    
    def apply_tsne_analysis(self, perplexity: int = 30, random_state: int = 42):
        """
        Apply t-SNE dimensionality reduction to generated images for all checkpoints
        
        Args:
            perplexity: t-SNE perplexity parameter
            random_state: Random state for reproducibility
        """
        print("Generating images and applying t-SNE for each checkpoint...")
        
        for epoch in tqdm(self.checkpoint_epochs, desc="Processing checkpoints"):
            # Generate images for this epoch
            epoch_images, epoch_labels = self.generate_images_for_epoch(epoch)
            
            # Apply t-SNE dimensionality reduction
            tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
            tsne_result = tsne.fit_transform(epoch_images)
            
            # Store results
            self.tsne_results[epoch] = tsne_result
            self.all_labels = epoch_labels  # Same label structure for all epochs
    
    def calculate_silhouette_scores(self) -> Dict[int, float]:
        """
        Calculate silhouette scores for each checkpoint to measure class separation quality
        
        Returns:
            Dictionary mapping epoch to silhouette score
        """
        print("Calculating silhouette scores...")
        silhouette_scores = {}
        
        for epoch in self.checkpoint_epochs:
            if epoch in self.tsne_results:
                # Calculate silhouette score on t-SNE reduced data
                score = silhouette_score(self.tsne_results[epoch], self.all_labels)
                silhouette_scores[epoch] = score
                print(f"Epoch {epoch}: Silhouette Score = {score:.4f}")
        
        return silhouette_scores
    
    def calculate_class_dispersion(self) -> Dict[int, List[float]]:
        """
        Calculate class dispersion (intra-class variance) for each checkpoint
        
        Returns:
            Dictionary mapping epoch to list of dispersions per class
        """
        print("Analyzing class dispersion...")
        dispersion_by_epoch = {}
        
        for epoch, tsne_result in self.tsne_results.items():
            dispersions = []
            
            for class_idx in range(self.num_classes):
                # Get points for this class
                mask = self.all_labels == class_idx
                class_points = tsne_result[mask]
                
                if len(class_points) > 1:
                    # Calculate class center
                    center = np.mean(class_points, axis=0)
                    
                    # Calculate average distance from center (dispersion)
                    distances = np.sqrt(np.sum((class_points - center)**2, axis=1))
                    dispersion = np.mean(distances)
                else:
                    dispersion = 0.0
                
                dispersions.append(dispersion)
            
            dispersion_by_epoch[epoch] = dispersions
        
        return dispersion_by_epoch
    
    def visualize_tsne_by_epoch(self, epochs_to_plot: Optional[List[int]] = None, 
                               save_dir: str = "./results/gan/tsne/"):
        """
        Visualize t-SNE plots for specific epochs
        
        Args:
            epochs_to_plot: List of epochs to plot (if None, plots all)
            save_dir: Directory to save plots
        """
        os.makedirs(save_dir, exist_ok=True)
        
        if epochs_to_plot is None:
            epochs_to_plot = self.checkpoint_epochs
        
        for epoch in epochs_to_plot:
            if epoch not in self.tsne_results:
                continue
                
            plt.figure(figsize=(10, 8))
            
            tsne_result = self.tsne_results[epoch]
            colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
            
            for class_idx in range(self.num_classes):
                mask = self.all_labels == class_idx
                class_points = tsne_result[mask]
                
                plt.scatter(class_points[:, 0], class_points[:, 1], 
                           c=[colors[class_idx]], label=self.class_names[class_idx], 
                           alpha=0.7, s=30)
            
            plt.title(f't-SNE Visualization - Epoch {epoch}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            save_path = os.path.join(save_dir, f"tsne_epoch_{epoch}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    def visualize_silhouette_scores(self, silhouette_scores: Dict[int, float], 
                                   save_path: str = "./results/gan/silhouette_scores.png"):
        """
        Create visualization for silhouette scores across epochs
        
        Args:
            silhouette_scores: Dictionary of silhouette scores by epoch
            save_path: Output file path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        epochs = list(silhouette_scores.keys())
        scores = list(silhouette_scores.values())
        
        plt.plot(epochs, scores, 'o-', linewidth=2, markersize=8, color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Silhouette Score')
        plt.title('Class Separation Quality by Training Epoch')
        plt.grid(True, alpha=0.3)
        
        # Highlight best epoch
        best_epoch = max(silhouette_scores, key=silhouette_scores.get)
        best_score = silhouette_scores[best_epoch]
        plt.scatter([best_epoch], [best_score], color='red', s=100, zorder=5)
        plt.annotate(f'Best: Epoch {best_epoch}\nScore: {best_score:.4f}', 
                    xy=(best_epoch, best_score), xytext=(10, 10),
                    textcoords='offset points', ha='left',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Silhouette scores plot saved to: {save_path}")
        print(f"Best checkpoint: Epoch {best_epoch} with Silhouette Score = {best_score:.4f}")
    
    def visualize_class_dispersion(self, dispersion_by_epoch: Dict[int, List[float]], 
                                  save_path: str = "./results/gan/class_dispersion.png"):
        """
        Create visualization for class dispersion across epochs
        
        Args:
            dispersion_by_epoch: Dictionary of dispersions by epoch and class
            save_path: Output file path
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(14, 8))
        epochs_sorted = sorted(dispersion_by_epoch.keys())
        
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_classes))
        
        for class_idx in range(self.num_classes):
            dispersions = [dispersion_by_epoch[epoch][class_idx] for epoch in epochs_sorted]
            plt.plot(epochs_sorted, dispersions, 'o-', 
                    label=self.class_names[class_idx], 
                    linewidth=2, color=colors[class_idx])
        
        plt.xlabel('Epoch')
        plt.ylabel('Class Dispersion (average distance from center)')
        plt.title('Class Dispersion by Training Epoch')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Class dispersion plot saved to: {save_path}")
    
    def run_complete_analysis(self, save_dir: str = "./results/gan/") -> Tuple[Dict[int, float], Dict[int, List[float]]]:
        """
        Run the complete analysis pipeline
        
        Args:
            save_dir: Directory to save results
            
        Returns:
            Tuple of (silhouette_scores, dispersion_by_epoch)
        """
        print("Starting comprehensive GAN performance analysis...")
        
        # Step 1: Find available checkpoints
        self.find_checkpoint_epochs(interval=30)
        
        if not self.checkpoint_epochs:
            raise ValueError("No checkpoints found in the specified directory")
        
        # Step 2: Apply t-SNE analysis to all checkpoints
        self.apply_tsne_analysis()
        
        # Step 3: Calculate quality metrics
        silhouette_scores = self.calculate_silhouette_scores()
        dispersion_by_epoch = self.calculate_class_dispersion()
        
        # Step 4: Create visualizations
        os.makedirs(save_dir, exist_ok=True)
        
        self.visualize_silhouette_scores(silhouette_scores, 
                                       os.path.join(save_dir, "silhouette_scores.png"))
        self.visualize_class_dispersion(dispersion_by_epoch, 
                                      os.path.join(save_dir, "class_dispersion.png"))
        
        # Step 5: Create t-SNE visualizations for key epochs
        key_epochs = [min(self.checkpoint_epochs), max(self.checkpoint_epochs)]
        if len(silhouette_scores) > 0:
            best_epoch = max(silhouette_scores, key=silhouette_scores.get)
            if best_epoch not in key_epochs:
                key_epochs.append(best_epoch)
        
        self.visualize_tsne_by_epoch(key_epochs, os.path.join(save_dir, "tsne/"))
        
        print(f"\nAnalysis complete! Results saved to: {save_dir}")
        print("Generated files:")
        print("1. silhouette_scores.png - Class separation quality over epochs")
        print("2. class_dispersion.png - Intra-class variance over epochs")
        print("3. tsne/ - t-SNE visualizations for key epochs")
        
        return silhouette_scores, dispersion_by_epoch