"""
Synapse multi-organ CT dataset loader.

Data: 50 abdominal CT scans with 8 organ labels.
Split: 14 train / 4 val / 12 test (from 30 released labeled cases).
"""

import os
import numpy as np
import nibabel as nib
from typing import Tuple, List
from .base_dataset import BaseMedicalDataset


class SynapseDataset(BaseMedicalDataset):
    """
    Synapse multi-organ CT dataset.
    
    Organs: aorta, gallbladder, left/right kidney, liver, pancreas, spleen, stomach
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 1,
        num_classes: int = 9,
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
        self.patient_ids = list(self.case_ids)
        self.case_ids = self._build_slice_index(self.patient_ids)

    def _build_slice_index(self, patient_ids: List[str]) -> List[tuple]:
        """Enumerate every axial slice while preserving the patient split."""
        samples = []
        for case_id in patient_ids:
            path = os.path.join(self.data_dir, 'train_vol', f"{case_id}.nii.gz")
            depth = int(nib.load(path).shape[2])
            samples.extend((case_id, slice_idx) for slice_idx in range(depth))
        return samples

    @staticmethod
    def _unpack_sample(sample_id):
        if not isinstance(sample_id, tuple) or len(sample_id) != 2:
            raise ValueError(f"Expected (case_id, slice_index), got {sample_id!r}")
        return sample_id[0], int(sample_id[1])
    
    def get_case_ids(self, split: str) -> List[str]:
        """
        Get case IDs for split using fixed seed.
        
        Returns 14/4/12 train/val/test from 30 released cases.
        """
        volume_dir = os.path.join(self.data_dir, 'train_vol')
        if not os.path.isdir(volume_dir):
            raise FileNotFoundError(f"Synapse volume directory not found: {volume_dir}")
        all_cases = sorted(
            filename[:-7]
            for filename in os.listdir(volume_dir)
            if filename.endswith('.nii.gz')
        )
        if len(all_cases) != 30:
            raise ValueError(f"Expected 30 released labeled Synapse cases, found {len(all_cases)}")
        
        # Fixed split using seed
        rng = np.random.RandomState(self.split_seed)
        rng.shuffle(all_cases)
        
        train_cases = all_cases[:14]
        val_cases = all_cases[14:18]
        test_cases = all_cases[18:30]
        
        if split == 'train':
            return train_cases
        elif split == 'val':
            return val_cases
        elif split == 'test':
            return test_cases
        else:
            raise ValueError(f"Unknown split: {split}")
    
    def _load_image(self, sample_id) -> np.ndarray:
        """
        Load CT volume and extract random slice.
        
        Preprocessing per paper: HU clipped to [-125, 275] then normalized to [0,1].
        No flips to preserve kidney laterality.
        
        Returns:
            (H, W) normalized CT slice
        """
        case_id, slice_idx = self._unpack_sample(sample_id)

        # Load volume
        vol_path = os.path.join(self.data_dir, 'train_vol', f"{case_id}.nii.gz")
        vol = nib.load(vol_path).get_fdata()
        
        # Clip HU to [-125, 275] as per paper
        vol = np.clip(vol, -125, 275)
        
        # Normalize to [0, 1]
        vol = (vol - (-125)) / (275 - (-125) + 1e-6)
        
        image = vol[:, :, slice_idx]
        
        # Resize to target size (bilinear as per paper)
        from skimage.transform import resize
        image = resize(image, self.image_size, order=1, mode='constant', anti_aliasing=True)
        
        return image
    
    def _load_label(self, sample_id) -> np.ndarray:
        """
        Load label volume and extract corresponding slice.
        
        Returns:
            (H, W) label slice
        """
        case_id, slice_idx = self._unpack_sample(sample_id)

        # Load label volume
        label_path = os.path.join(self.data_dir, 'train_label', f"{case_id}.nii.gz")
        label_vol = nib.load(label_path).get_fdata()
        
        label = label_vol[:, :, slice_idx].astype(np.int64)
        
        # Resize to target size (use nearest neighbor for labels)
        from skimage.transform import resize
        label = resize(label, self.image_size, order=0, mode='constant', anti_aliasing=False)
        
        return label

    def sample_metadata(self, idx: int) -> dict:
        """Metadata needed to restore a 224x224 prediction to the native CT grid."""
        case_id, slice_idx = self._unpack_sample(self.case_ids[idx])
        image_nii = nib.load(
            os.path.join(self.data_dir, 'train_vol', f"{case_id}.nii.gz")
        )
        return {
            'case_id': case_id,
            'volume_id': case_id,
            'slice_index': slice_idx,
            'native_shape': tuple(int(v) for v in image_nii.shape[:3]),
            'spacing': tuple(float(v) for v in image_nii.header.get_zooms()[:3]),
        }

    def native_label_slice(self, idx: int) -> np.ndarray:
        case_id, slice_idx = self._unpack_sample(self.case_ids[idx])
        label_nii = nib.load(
            os.path.join(self.data_dir, 'train_label', f"{case_id}.nii.gz")
        )
        return np.asarray(label_nii.dataobj[:, :, slice_idx], dtype=np.int64)
