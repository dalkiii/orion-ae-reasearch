import numpy as np
import os
import random
import torch
import glob
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, Subset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Any, Union


class ScalogramDataset(Dataset):
    """Unified dataset class for loading and processing scalogram data"""
    
    def __init__(self, 
                 root_dir: str, 
                 transform=None, 
                 imbalance_factors: Optional[Dict[str, float]] = None,
                 test_samples_per_class: int = 100,
                 seed: int = 42,
                 data_format: str = "classification",  # "classification" or "gan"
                 load_mode: str = "npy"):  # "npy" or "directory"
        """
        Initialize the scalogram dataset
        
        Args:
            root_dir: Root directory containing class folders or data files
            transform: Transform to be applied to samples
            imbalance_factors: Dictionary of class imbalance factors
            test_samples_per_class: Number of test samples per class
            seed: Random seed for reproducibility
            data_format: "classification" (PIL Images) or "gan" (numpy arrays)
            load_mode: "npy" (from .npy files) or "directory" (from folder structure)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.imbalance_factors = imbalance_factors or {}
        self.test_samples_per_class = test_samples_per_class
        self.seed = seed
        self.data_format = data_format
        self.load_mode = load_mode
        
        # Class mapping - supports both formats
        self.class_map = {
            "05cNm": 0, "10cNm": 1, "20cNm": 2, "30cNm": 3,
            "40cNm": 4, "50cNm": 5, "60cNm": 6
        }
        self.class_map_simple = {"05": 0, "10": 1, "20": 2, "30": 3, "40": 4, "50": 5, "60": 6}
        self.reverse_class_map = {v: k for k, v in self.class_map.items()}
        
        # Data storage
        self.data_samples = []
        self.test_samples = []
        
        # Set random seed
        random.seed(self.seed)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and process all data from the root directory"""
        print("Loading scalogram data...")
        
        if self.load_mode == "directory":
            all_samples = self._load_from_directory()
        else:  # npy mode
            all_samples = self._load_from_npy_files()
        
        if not all_samples:
            raise ValueError("No data loaded. Check data path and format.")
        
        # Print class distribution
        self._print_class_distribution(all_samples, "Original")
        
        # Split test data (only for classification format)
        if self.data_format == "classification":
            self._split_test_data(all_samples)
        
        # Apply imbalance to training data
        self._apply_imbalance(all_samples)
        
        # Print final distributions
        self._print_final_distributions()
    
    def _load_from_directory(self) -> Dict[int, List]:
        """Load data from directory structure (original classification format)"""
        all_samples = defaultdict(list)
        
        for class_folder in sorted(os.listdir(self.root_dir)):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path) or class_folder not in self.class_map:
                continue
                
            class_label = self.class_map[class_folder]
            print(f"Loading {class_folder} (label {class_label})...")
            
            samples = self._load_class_samples(class_path, class_label)
            all_samples[class_label].extend(samples)
            
            print(f"  Loaded {len(samples)} samples")
        
        return all_samples
    
    def _load_from_npy_files(self) -> Dict[int, List]:
        """Load data from .npy files (GAN format)"""
        all_samples = defaultdict(list)
        
        # Find all numpy files
        pattern = os.path.join(self.root_dir, "*cNm", "*.npy")
        file_list = glob.glob(pattern)
        
        print(f"Found {len(file_list)} numpy files in {self.root_dir}")
        
        for file_path in file_list:
            try:
                parts = file_path.split(os.sep)
                tightening_folder = parts[-2]  # e.g., "30cNm"
                torque_str = tightening_folder.replace("cNm", "")
                
                # Use appropriate class mapping
                class_mapping = self.class_map_simple if torque_str in self.class_map_simple else self.class_map
                class_key = torque_str if torque_str in self.class_map_simple else tightening_folder
                
                if class_key not in class_mapping:
                    print(f"Skipped file (unknown class): {file_path}")
                    continue
                
                class_label = class_mapping[class_key]
                data = np.load(file_path).astype(np.float32)
                
                print(f"Loading: {file_path}, Shape: {data.shape}")
                
                # Process each sample in the file
                for i in range(data.shape[0]):
                    sample = data[i]
                    processed_sample = self._process_numpy_sample(sample)
                    all_samples[class_label].append((processed_sample, class_label))
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return all_samples
    
    def _load_class_samples(self, class_path: str, class_label: int) -> List[Tuple[Union[Image.Image, np.ndarray], int]]:
        """Load samples from a single class directory"""
        samples = []
        
        for file_name in os.listdir(class_path):
            if not file_name.endswith(".npy"):
                continue
                
            file_path = os.path.join(class_path, file_name)
            try:
                scalogram_data = np.load(file_path)
                print(f"    Loaded {file_path}: {scalogram_data.shape}")
                
                for i in range(scalogram_data.shape[0]):
                    if self.data_format == "classification":
                        # Convert to PIL Image for classification
                        sample = self._normalize_sample(scalogram_data[i, :, :])
                        sample_image = Image.fromarray(sample.astype(np.uint8)).convert('RGB')
                        samples.append((sample_image, class_label))
                    else:
                        # Keep as numpy array for GAN
                        sample = self._process_numpy_sample(scalogram_data[i, :, :])
                        samples.append((sample, class_label))
                    
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
        
        return samples
    
    def _normalize_sample(self, sample: np.ndarray) -> np.ndarray:
        """Normalize sample to 0-255 range for PIL Image"""
        sample_min = sample.min()
        sample_max = sample.max()
        
        if sample_max > sample_min:
            sample = (sample - sample_min) / (sample_max - sample_min) * 255
        
        return sample
    
    def _process_numpy_sample(self, sample: np.ndarray) -> np.ndarray:
        """Process numpy sample for GAN training"""
        # Ensure proper dimensions
        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=-1)  # Add channel dimension
        elif sample.ndim == 3 and sample.shape[0] in [1, 3]:
            # Already in channel-first format
            pass
        elif sample.ndim == 3:
            # Assume channel-last, no change needed for now
            pass
        
        return sample
    
    def _split_test_data(self, all_samples: Dict[int, List]):
        """Split test data from all samples (only for classification)"""
        test_indices = {}
        self.test_samples = []
        
        for class_label, samples in all_samples.items():
            if len(samples) == 0:
                continue
                
            # Select test samples
            count_to_select = min(self.test_samples_per_class, len(samples))
            test_indices[class_label] = random.sample(range(len(samples)), count_to_select)
            
            # Add to test samples
            for idx in test_indices[class_label]:
                self.test_samples.append(samples[idx])
            
            class_name = self.reverse_class_map[class_label]
            print(f"Selected {count_to_select} test samples for class {class_name}")
        
        # Remove test samples from training data
        trainval_samples = []
        for class_label, samples in all_samples.items():
            if class_label in test_indices:
                selected_indices = set(test_indices[class_label])
                trainval_samples.extend([
                    samples[i] for i in range(len(samples)) 
                    if i not in selected_indices
                ])
        
        # Update all_samples to contain only train/val data
        all_samples.clear()
        for sample, label in trainval_samples:
            all_samples[label].append((sample, label))
    
    def _apply_imbalance(self, all_samples: Dict[int, List]):
        """Apply class imbalance to data"""
        if not self.imbalance_factors:
            self.data_samples = [item for sublist in all_samples.values() for item in sublist]
            print("No imbalance applied.")
            return
        
        print("\nApplying imbalance factors:")
        filtered_samples = []
        
        for label, samples in all_samples.items():
            class_name = self.reverse_class_map[label]
            # Support both formats for imbalance factors
            factor = self.imbalance_factors.get(class_name, 
                     self.imbalance_factors.get(class_name.replace("cNm", ""), 1.0))
            allowed_count = max(1, int(len(samples) * factor))
            
            # Shuffle and select
            random.shuffle(samples)
            filtered_samples.extend(samples[:allowed_count])
            
            print(f"  {class_name}: {len(samples)} → {allowed_count} (factor: {factor})")
        
        self.data_samples = filtered_samples
    
    def _print_class_distribution(self, all_samples: Dict[int, List], prefix: str):
        """Print class distribution"""
        print(f"\n{prefix} class distribution:")
        for class_label, samples in sorted(all_samples.items()):
            class_name = self.reverse_class_map[class_label]
            print(f"  {class_name}: {len(samples)} samples")
    
    def _print_final_distributions(self):
        """Print final data distributions"""
        print(f"\nFinal distribution:")
        print(f"  Train/Val: {len(self.data_samples)}")
        if self.data_format == "classification":
            print(f"  Test: {len(self.test_samples)}")
        
        # Train/Val class distribution
        trainval_counts = defaultdict(int)
        for _, label in self.data_samples:
            trainval_counts[label] += 1
        
        print(f"\nTrain/Val class distribution:")
        for class_label, count in sorted(trainval_counts.items()):
            class_name = self.reverse_class_map[class_label]
            print(f"  {class_name}: {count}")
        
        # Test class distribution (only for classification)
        if self.data_format == "classification" and self.test_samples:
            test_counts = defaultdict(int)
            for _, label in self.test_samples:
                test_counts[label] += 1
            
            print(f"\nTest class distribution:")
            for class_label, count in sorted(test_counts.items()):
                class_name = self.reverse_class_map[class_label]
                print(f"  {class_name}: {count}")
    
    def __len__(self) -> int:
        return len(self.data_samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, label = self.data_samples[idx]
        
        if self.data_format == "gan":
            # For GAN: handle numpy arrays
            if isinstance(sample, np.ndarray):
                # Convert to tensor and adjust dimensions
                if sample.ndim == 2:
                    sample = np.expand_dims(sample, axis=0)  # Add channel dimension
                elif sample.ndim == 3 and sample.shape[-1] in [1, 3]:
                    sample = np.transpose(sample, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                
                sample = torch.tensor(sample, dtype=torch.float32)
            
            if self.transform:
                sample = self.transform(sample)
                
        else:
            # For classification: handle PIL Images
            if isinstance(sample, torch.Tensor):
                pass  # Already a tensor
            elif self.transform:
                sample = self.transform(sample)
        
        return sample, torch.tensor(label, dtype=torch.long)
    
    def get_test_dataset(self, transform=None):
        """Get test dataset (only for classification)"""
        if self.data_format != "classification":
            raise ValueError("Test dataset only available for classification format")
        return TestDataset(self.test_samples, transform)
    
    def get_balanced_sampler(self) -> WeightedRandomSampler:
        """
        Create balanced sampler for training
        
        Returns:
            WeightedRandomSampler instance
        """
        # Extract labels
        labels = [label for _, label in self.data_samples]
        labels_array = np.array(labels)
        
        # Calculate class frequencies
        class_counts = np.bincount(labels_array)
        class_weights = 1.0 / class_counts  # Higher weight for minority classes
        
        # Assign weight to each sample
        sample_weights = np.array([class_weights[label] for label in labels])
        sample_weights = torch.from_numpy(sample_weights).double()
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(labels),
            replacement=True
        )
        
        return sampler
    
    def get_num_classes(self) -> int:
        """Get number of classes"""
        return len(self.class_map)
    
    def get_class_names(self) -> List[str]:
        """Get class names"""
        return list(self.class_map.keys())
    
    def visualize_samples(self, num_samples: int = 10, save_path: Optional[str] = None) -> None:
        """
        Visualize samples from each class
        
        Args:
            num_samples: Number of samples per class to visualize
            save_path: Optional path to save the visualization
        """
        # Extract labels and group samples by class
        samples_by_class = defaultdict(list)
        for i, (_, label) in enumerate(self.data_samples):
            samples_by_class[label].append(i)
        
        classes = sorted(samples_by_class.keys())
        n_classes = len(classes)
        
        fig, axes = plt.subplots(n_classes, num_samples, figsize=(num_samples * 2, n_classes * 2))
        
        if n_classes == 1:
            axes = np.expand_dims(axes, 0)
        
        for i, cls in enumerate(classes):
            cls_indices = samples_by_class[cls]
            chosen = random.sample(cls_indices, min(num_samples, len(cls_indices)))
            
            for j in range(num_samples):
                ax = axes[i, j] if n_classes > 1 else axes[j]
                ax.axis("off")
                
                if j < len(chosen):
                    sample, _ = self.data_samples[chosen[j]]
                    
                    # Handle different sample types
                    if isinstance(sample, Image.Image):
                        ax.imshow(sample)
                    elif isinstance(sample, np.ndarray):
                        if sample.ndim == 3 and sample.shape[-1] == 1:
                            ax.imshow(sample.squeeze(), cmap='gray', vmin=-1, vmax=1)
                        elif sample.ndim == 3:
                            ax.imshow(sample, vmin=-1, vmax=1)
                        elif sample.ndim == 2:
                            ax.imshow(sample, cmap='gray', vmin=-1, vmax=1)
                    elif isinstance(sample, torch.Tensor):
                        sample_np = sample.numpy()
                        if sample_np.ndim == 3 and sample_np.shape[0] == 1:
                            ax.imshow(sample_np.squeeze(), cmap='gray')
                        elif sample_np.ndim == 3:
                            ax.imshow(np.transpose(sample_np, (1, 2, 0)))
                        else:
                            ax.imshow(sample_np, cmap='gray')
                else:
                    ax.set_visible(False)
            
            # Set class label
            class_name = self.reverse_class_map[cls]
            if n_classes > 1:
                axes[i, 0].set_ylabel(f"Class {class_name}", fontsize=14)
            else:
                axes[0].set_ylabel(f"Class {class_name}", fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


class TestDataset(Dataset):
    """Separate test dataset class"""
    
    def __init__(self, samples: List[Tuple[Any, int]], transform=None):
        self.samples = samples
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample, label = self.samples[idx]
        
        if isinstance(sample, torch.Tensor):
            return sample, torch.tensor(label, dtype=torch.long)
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, torch.tensor(label, dtype=torch.long)


def print_class_distribution(labels: np.ndarray, title: str = "Class Distribution") -> None:
    """Print class distribution statistics"""
    unique_classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    
    print(f"\n{title}:")
    print("-" * 40)
    for cls, count in zip(unique_classes, counts):
        percentage = (count / total) * 100
        print(f"  Class {cls}: {count:4d} samples ({percentage:5.1f}%)")
    print(f"  Total: {total} samples")
    print("-" * 40)
    

def get_transforms() -> transforms.Compose:
    """Get standard image transforms for scalogram data"""
    return transforms.Compose([
        transforms.Lambda(lambda x: x.squeeze(-1) if isinstance(x, np.ndarray) and x.ndim == 3 else x),  # (H,W,1) → (H,W)
        transforms.Lambda(lambda x: Image.fromarray(x.astype(np.uint8), mode='L') if isinstance(x, np.ndarray) else x),  # Grayscale PIL
        transforms.Grayscale(num_output_channels=3),  # Grayscale → RGB
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def create_datasets(root_dir: str, 
                   imbalance_factors: Optional[Dict[str, float]] = None,
                   test_samples_per_class: int = 100,
                   train_val_split: float = 0.8,
                   seed: int = 42) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets
    
    Args:
        root_dir: Root directory containing class folders
        imbalance_factors: Dictionary of class imbalance factors
        test_samples_per_class: Number of test samples per class
        train_val_split: Ratio for train/validation split
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (full_dataset, train_dataset, val_dataset, test_dataset)
    """
    transform = get_transforms()
    
    # Create main dataset
    dataset = ScalogramDataset(
        root_dir=root_dir,
        transform=transform,
        imbalance_factors=imbalance_factors,
        test_samples_per_class=test_samples_per_class,
        seed=seed
    )
    
    # Split train/validation
    total_samples = len(dataset)
    train_size = int(train_val_split * total_samples)
    
    # Create indices and shuffle
    indices = list(range(total_samples))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    # Create test dataset
    test_dataset = dataset.get_test_dataset(transform)
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Validation: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    return dataset, train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset: Dataset,
                       val_dataset: Dataset,
                       test_dataset: Dataset,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader