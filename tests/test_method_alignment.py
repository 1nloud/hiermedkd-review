import unittest

import numpy as np

try:
    import torch
    from losses import ActivationEnergyLoss, BoundaryWeightedKLLoss, NormalizedFeatureLoss
    from models.student import LiteMedFormerS
    from models.teacher import SwinUNETeacher
    from utils import compute_nsd
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for method-alignment tests")
class LossEquationTests(unittest.TestCase):
    def test_feature_loss_uses_channelwise_squared_l2(self):
        teacher = [torch.tensor([[[[1.0]], [[0.0]]]])]
        student = [torch.tensor([[[[0.0]], [[1.0]]]])]
        loss = NormalizedFeatureLoss()(teacher, student)
        self.assertAlmostEqual(loss.item(), 2.0, places=6)

    def test_activation_loss_has_no_spatial_divisor(self):
        teacher = [torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])]
        student = [torch.tensor([[[[0.0, 1.0], [0.0, 0.0]]]])]
        loss = ActivationEnergyLoss()(teacher, student)
        self.assertAlmostEqual(loss.item(), 2.0, places=6)

    def test_binary_morphology_finds_nonzero_boundary(self):
        labels = torch.zeros((1, 7, 7), dtype=torch.long)
        labels[:, 2:5, 2:5] = 1
        criterion = BoundaryWeightedKLLoss(kernel_size=3)
        boundary = criterion.extract_boundary_mask(labels, num_classes=2)
        self.assertGreater(boundary.sum().item(), 0)
        self.assertEqual(boundary[0, 3, 3].item(), 0.0)
        self.assertEqual(boundary[0, 2, 2].item(), 1.0)

    def test_zero_padding_marks_edge_touching_object_boundary(self):
        labels = torch.ones((1, 5, 5), dtype=torch.long)
        criterion = BoundaryWeightedKLLoss(kernel_size=3)
        boundary = criterion.extract_boundary_mask(labels, num_classes=2)
        self.assertEqual(boundary[0, 0, 0].item(), 1.0)
        self.assertEqual(boundary[0, 2, 2].item(), 0.0)

    def test_kl_uses_bhw_denominator(self):
        teacher_logits = torch.tensor(
            [[[[2.0, 2.0]], [[0.0, 0.0]]]], dtype=torch.float32
        )
        student_logits = torch.tensor(
            [[[[0.0, 0.0]], [[2.0, 2.0]]]], dtype=torch.float32
        )
        labels = torch.tensor([[[0, 255]]], dtype=torch.long)
        criterion = BoundaryWeightedKLLoss(
            temperature=1.0,
            beta=0.0,
            confidence_gating=False,
        )
        actual = criterion(teacher_logits, student_logits, labels)

        p = torch.softmax(teacher_logits[:, :, :, :1], dim=1)
        q = torch.softmax(student_logits[:, :, :, :1], dim=1)
        per_valid_pixel = torch.sum(p * torch.log(p / q))
        expected = per_valid_pixel / 2.0
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_models_expose_four_aligned_encoder_stages(self):
        image = torch.randn(1, 1, 224, 224)
        student = LiteMedFormerS(in_channels=1, num_classes=8).eval()
        teacher = SwinUNETeacher(in_channels=1, num_classes=8).eval()
        with torch.no_grad():
            student_features = student(image)['features']
            teacher_features = teacher(image)['features']
        self.assertEqual(len(student_features), 4)
        self.assertEqual(len(teacher_features), 4)
        self.assertEqual(
            [feature.shape[-2:] for feature in student_features],
            [feature.shape[-2:] for feature in teacher_features],
        )


@unittest.skipIf(torch is None, "PyTorch is required for metric tests")
class SurfaceMetricTests(unittest.TestCase):
    def test_nsd_identical_masks_is_one(self):
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[2:7, 2:7] = 1
        self.assertAlmostEqual(compute_nsd(mask, mask, tolerance=1.0), 1.0)


if __name__ == '__main__':
    unittest.main()
