"""
Base dataset class for medical image segmentation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional, Callable
import os


class BaseMedicalDataset(Dataset):
    """
    Base dataset class for medical image segmentation.
    
    Subclasses should implement:
    - _load_image(case_id): Load image for a case
    - _load_label(case_id): Load label for a case
    - get_case_ids(split): Get case IDs for a split
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 1,
        num_classes: int = 9,
        ignore_label: int = 255,
        transform: Optional[Callable] = None,
        horizontal_flip_probability: float = 0.0
    ):
        """
        Args:
            data_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            in_channels: Number of input channels (1 for grayscale, 3 for RGB)
            num_classes: Number of segmentation classes
            ignore_label: Label value to ignore
            transform: Optional data augmentation transform
        """
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.transform = transform
        self.horizontal_flip_probability = float(horizontal_flip_probability)
        
        self.case_ids = self.get_case_ids(split)
    
    def get_case_ids(self, split: str) -> list:
        """
        Get case IDs for a specific split.
        
        Args:
            split: Dataset split
        
        Returns:
            List of case IDs
        """
        raise NotImplementedError("Subclasses must implement get_case_ids")
    
    def _load_image(self, case_id: str) -> np.ndarray:
        """
        Load image for a case.
        
        Args:
            case_id: Case identifier
        
        Returns:
            Image array
        """
        raise NotImplementedError("Subclasses must implement _load_image")
    
    def _load_label(self, case_id: str) -> np.ndarray:
        """
        Load label for a case.
        
        Args:
            case_id: Case identifier
        
        Returns:
            Label array
        """
        raise NotImplementedError("Subclasses must implement _load_label")
    
    def __len__(self) -> int:
        return len(self.case_ids)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            (image, label) tuple
        """
        case_id = self.case_ids[idx]
        
        # Load image and label
        image = self._load_image(case_id)
        label = self._load_label(case_id)
        
        if self.split == 'train' and self.horizontal_flip_probability > 0:
            if np.random.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=-1).copy()
                label = np.flip(label, axis=-1).copy()

        # Apply optional paired transforms
        if self.transform is not None:
            image, label = self.transform(image, label)

        # Grayscale loaders return H x W; models require C x H x W.
        if isinstance(image, np.ndarray) and image.ndim == 2:
            image = image[None, ...]
        
        # Convert to tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label).long()
        
        return image, label
