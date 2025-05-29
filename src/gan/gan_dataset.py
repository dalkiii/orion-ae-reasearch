from ..dataset import ScalogramDataset
from torch.utils.data import DataLoader
from typing import Dict, Optional


def create_gan_dataloader(root_dir: str,
                         imbalance_ratios: Optional[Dict[str, float]] = None,
                         batch_size: int = 64,
                         num_workers: int = 4,
                         seed: int = 42) -> DataLoader:
    """
    Create GAN dataloader using unified ScalogramDataset
    
    Args:
        root_dir: Root directory containing data
        imbalance_ratios: Dictionary mapping class names to sampling ratios
        batch_size: Batch size
        num_workers: Number of workers
        seed: Random seed
    
    Returns:
        DataLoader for GAN training
    """
    # Create dataset for GAN training
    dataset = ScalogramDataset(
        root_dir=root_dir,
        transform=None,
        imbalance_factors=imbalance_ratios,
        test_samples_per_class=0,  # No test split for GAN
        seed=seed,
        data_format="gan",
        load_mode="npy"
    )
    
    # Create balanced sampler
    sampler = dataset.get_balanced_sampler()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True
    )
    
    return dataloader