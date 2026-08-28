"""
HierMedKD package initialization.
"""

from .models.teacher import SwinUNETeacher
from .models.student import LiteMedFormerS
from .models.adapters import StageAdapters
from .losses.__init__ import HierMedKDLoss, NormalizedFeatureLoss, ActivationEnergyLoss, BoundaryWeightedKLLoss
from .losses.supervised_loss import SupervisedLoss, DiceLoss

__version__ = "1.0.0"
__all__ = [
    "SwinUNETeacher",
    "LiteMedFormerS",
    "StageAdapters",
    "HierMedKDLoss",
    "NormalizedFeatureLoss",
    "ActivationEnergyLoss",
    "BoundaryWeightedKLLoss",
    "SupervisedLoss",
    "DiceLoss",
]