"""
ISIC 2018 dermoscopy dataset loader.

Data: 2594 dermoscopy images with lesion segmentation masks.
Split: 1815 train / 259 val / 100 test.
"""

import os
import numpy as np
from PIL import Image
from typing import Tuple, List
from .base_dataset import BaseMedicalDataset


class ISICDataset(BaseMedicalDataset):
    """
    ISIC 2018 dermoscopy dataset.
    
    Classes: background, lesion
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        num_classes: int = 2,
        ignore_label: int = 255,
        transform=None,
        split_seed: int = 2027,
        horizontal_flip_probability: float = 0.0
    ):
        self.split_seed = split_seed
        super().__init__(
            data_dir, split, image_size, in_channels, num_classes, ignore_label,
            transform, horizontal_flip_probability
        )
    
    def get_case_ids(self, split: str) -> List[str]:
        """
        Get image IDs for split using fixed seed.
        
        Returns 1815/259/520 train/val/test from 2594 training images.
        """
        # All training images
        input_dir = os.path.join(self.data_dir, 'ISIC2018_Task1-2_Training_Input')
        all_images = sorted(
            f.replace('.jpg', '') for f in os.listdir(input_dir) if f.endswith('.jpg')
        )
        if len(all_images) != 2594:
            raise ValueError(f"Expected 2,594 released ISIC training images, found {len(all_images)}")
        
        # Fixed split using seed
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(all_images)
        
        train_images = all_images[:1815]
        val_images = all_images[1815:2074]
        test_images = all_images[2074:2594]  # 520 images for test
        
        if split == 'train':
            return train_images
        elif split == 'val':
            return val_images
        elif split == 'test':
            return test_images
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_image(self, image_id: str) -> np.ndarray:
        """
        Load RGB dermoscopy image.
        
        Preprocessing per paper: RGB/255 normalization.
        Horizontal flip probability 0.5 (should be handled in transform).
        
        Returns:
            (H, W, 3) RGB image normalized to [0, 1]
        """
        input_dir = os.path.join(self.data_dir, 'ISIC2018_Task1-2_Training_Input')
        image_path = os.path.join(input_dir, f"{image_id}.jpg")
        
        image = Image.open(image_path).convert('RGB')
        image = np.array(image, dtype=np.float32) / 255.0  # RGB/255 as per paper
        
        # Resize to target size (bilinear as per paper)
        from skimage.transform import resize
        image = resize(image, self.image_size + (3,), order=1, mode='constant', anti_aliasing=True)
        
        # Transpose to (C, H, W)
        image = np.transpose(image, (2, 0, 1))
        
        return image
    
    def _load_label(self, image_id: str) -> np.ndarray:
        """
        Load segmentation mask.
        
        Returns:
            (H, W) label mask
        """
        label_dir = os.path.join(self.data_dir, 'ISIC2018_Task1_Training_GroundTruth')
        label_path = os.path.join(label_dir, f"{image_id}_segmentation.png")
        
        label = Image.open(label_path)
        label = np.array(label, dtype=np.int64)
        
        # Binary mask: 0 = background, 1 = lesion
        label = (label > 0).astype(np.int64)
        
        # Resize to target size
        from skimage.transform import resize
        label = resize(label, self.image_size, order=0, mode='constant', anti_aliasing=False)
        
        return label
