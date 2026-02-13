"""
Pure PyTorch replacements for PyTorch3D operations.
These implementations avoid CUDA-only dependencies so the code
can run on Google Cloud TPU via torch_xla.
"""

import torch
import math


# ---------------------------------------------------------------------------
# KNN
# ---------------------------------------------------------------------------

class _KNNResult:
    """Mimics the pytorch3d KNN result namedtuple."""
    def __init__(self, dists, idx, knn):
        self.dists = dists   # (B, N, K)
        self.idx = idx       # (B, N, K)
        self.knn = knn       # (B, N, K, D)


def knn_points(p1, p2, K=1):
    """
    Pure-PyTorch K-nearest-neighbours.

    Args:
        p1: (B, N, D) query points
        p2: (B, M, D) reference points
        K:  number of neighbours

    Returns:
        _KNNResult with .dists (squared), .idx, .knn
    """
    # (B, N, M)
    dists = torch.cdist(p1, p2, p=2.0).pow(2)
    # topk smallest
    knn_dists, knn_idx = dists.topk(K, dim=-1, largest=False)
    # gather actual points  (B, N, K, D)
    knn_points_out = knn_gather(p2, knn_idx)
    return _KNNResult(knn_dists, knn_idx, knn_points_out)


def knn_gather(x, idx):
    """
    Gather points from x according to idx.

    Args:
        x:   (B, M, D)
        idx: (B, N, K)

    Returns:
        (B, N, K, D)
    """
    B, M, D = x.shape
    _, N, K = idx.shape
    idx_expanded = idx.unsqueeze(-1).expand(B, N, K, D)
    return x.unsqueeze(1).expand(B, N, M, D).gather(2, idx_expanded)


# ---------------------------------------------------------------------------
# 3-D Transforms
# ---------------------------------------------------------------------------

def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle (Rodrigues) representation to 3x3 rotation matrices.

    Args:
        axis_angle: (..., 3)  angle * axis

    Returns:
        (..., 3, 3) rotation matrices
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)           # (..., 1)
    safe_angles = torch.clamp(angles, min=1e-8)
    axis = axis_angle / safe_angles                                       # (..., 3)

    cos_a = torch.cos(angles).unsqueeze(-1)   # (..., 1, 1)
    sin_a = torch.sin(angles).unsqueeze(-1)   # (..., 1, 1)

    # skew-symmetric matrix K
    zero = torch.zeros_like(axis[..., 0])
    K = torch.stack([
        zero,          -axis[..., 2],  axis[..., 1],
        axis[..., 2],  zero,          -axis[..., 0],
        -axis[..., 1], axis[..., 0],  zero,
    ], dim=-1).reshape(axis.shape[:-1] + (3, 3))

    # outer product  axis . axis^T
    outer = axis.unsqueeze(-1) * axis.unsqueeze(-2)   # (..., 3, 3)

    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    R = cos_a * eye + sin_a * K + (1.0 - cos_a) * outer

    # For zero-angle inputs, R should be identity
    small = (angles.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    R = torch.where(small, eye.expand_as(R), R)
    return R


class Transform3d:
    """
    Minimal replacement for pytorch3d.transforms.Transform3d.

    Supports .rotate(R) and .transform_points(points).
    Operates on CPU or any device (including XLA / TPU).
    """

    def __init__(self, device=None):
        self._matrix = None
        self._device = device

    def rotate(self, R):
        """
        Set the rotation matrix.

        Args:
            R: (..., 3, 3) rotation matrix or (3, 3)
        """
        self._matrix = R
        return self

    def transform_points(self, points):
        """
        Apply the rotation to a set of points.

        Args:
            points: (..., N, 3)

        Returns:
            (..., N, 3)
        """
        R = self._matrix
        # points @ R^T
        return torch.matmul(points, R.transpose(-1, -2))
