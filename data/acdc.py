"""
ACDC cardiac MRI dataset loader.

Data: 150 patients with 4 cardiac structure labels.
Split: 70 train / 10 val / 20 test (from 100 training patients).
"""

import os
import numpy as np
import nibabel as nib
from typing import Tuple, List
from .base_dataset import BaseMedicalDataset


class ACDCDataset(BaseMedicalDataset):
    """
    ACDC cardiac MRI dataset.
    
    Structures: background, right ventricle, myocardium, left ventricle
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 1,
        num_classes: int = 4,
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
        Get patient IDs for split using fixed seed.
        
        Returns 70/10/20 train/val/test from 100 training patients.
        """
        training_dir = os.path.join(self.data_dir, 'training')
        if not os.path.isdir(training_dir):
            raise FileNotFoundError(f"ACDC training directory not found: {training_dir}")
        all_patients = sorted(
            name for name in os.listdir(training_dir)
            if name.startswith('patient') and os.path.isdir(os.path.join(training_dir, name))
        )
        if len(all_patients) != 100:
            raise ValueError(f"Expected 100 labeled ACDC training patients, found {len(all_patients)}")
        
        # Fixed split using seed
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(all_patients)
        
        train_patients = all_patients[:70]
        val_patients = all_patients[70:80]
        test_patients = all_patients[80:100]
        
        if split == 'train':
            return train_patients
        elif split == 'val':
            return val_patients
        elif split == 'test':
            return test_patients
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_image(self, patient_id: str) -> np.ndarray:
        """
        Load MRI frame and extract random frame.
        
        Preprocessing per paper: clip nonzero p0.5/p99.5 then z-score normalization.
        Horizontal flip probability 0.5 (should be handled in transform).
        
        Returns:
            (H, W) normalized MRI slice
        """
        # Load all frames for this patient
        patient_dir = os.path.join(self.data_dir, 'training', patient_id)
        frames = sorted(
            f for f in os.listdir(patient_dir)
            if f.endswith('.nii.gz') and '_gt' not in f and '_4d' not in f
        )
        
        # Select random frame
        frame_path = os.path.join(patient_dir, frames[0])
        vol = nib.load(frame_path).get_fdata()
        
        # Clip nonzero p0.5/p99.5 as per paper
        nonzero_vals = vol[vol != 0]
        if len(nonzero_vals) > 0:
            p05 = np.percentile(nonzero_vals, 0.5)
            p995 = np.percentile(nonzero_vals, 99.5)
            vol = np.clip(vol, p05, p995)
        
        # Z-score normalization as per paper
        if vol.std() > 0:
            vol = (vol - vol.mean()) / vol.std()
        
        # Select middle slice
        slice_idx = vol.shape[2] // 2
        image = vol[:, :, slice_idx]
        
        # Resize to target size (bilinear as per paper)
        from skimage.transform import resize
        image = resize(image, self.image_size, order=1, mode='constant', anti_aliasing=True)
        
        return image
    
    def _load_label(self, patient_id: str) -> np.ndarray:
        """
        Load label frame and extract corresponding slice.
        
        Returns:
            (H, W) label slice
        """
        patient_dir = os.path.join(self.data_dir, 'training', patient_id)
        frames = sorted(f for f in os.listdir(patient_dir) if f.endswith('_gt.nii.gz'))
        
        frame_path = os.path.join(patient_dir, frames[0])
        label_vol = nib.load(frame_path).get_fdata()
        
        # Select same slice as image
        slice_idx = label_vol.shape[2] // 2
        label = label_vol[:, :, slice_idx].astype(np.int64)
        
        # Resize to target size
        from skimage.transform import resize
        label = resize(label, self.image_size, order=0, mode='constant', anti_aliasing=False)
        
        return label
