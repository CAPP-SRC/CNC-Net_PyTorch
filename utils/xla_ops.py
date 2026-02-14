"""Pure PyTorch replacements for PyTorch3D operations.

These implementations avoid CUDA-only custom kernels so they can run on
any device backend supported by PyTorch, including XLA/TPU.
"""

import torch
from collections import namedtuple


# Named tuple matching the fields used from pytorch3d.ops.knn.knn_points
KNN = namedtuple("KNN", ["dists", "idx", "knn"])


def axis_angle_to_matrix(axis_angle):
    """Convert axis-angle representation to rotation matrix (Rodrigues).

    Args:
        axis_angle: (*, 3) axis-angle vectors where direction is the rotation
            axis and magnitude is the rotation angle in radians.

    Returns:
        (*, 3, 3) rotation matrices.
    """
    batch_shape = axis_angle.shape[:-1]
    dtype = axis_angle.dtype
    device = axis_angle.device

    angle = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # (*, 1)
    safe_angle = torch.clamp(angle, min=1e-8)
    axis = axis_angle / safe_angle  # (*, 3)

    cos_a = torch.cos(angle)  # (*, 1)
    sin_a = torch.sin(angle)  # (*, 1)

    x = axis[..., 0:1]
    y = axis[..., 1:2]
    z = axis[..., 2:3]
    zero = torch.zeros_like(x)

    # Skew-symmetric matrix K
    K = torch.stack([
        torch.cat([zero, -z, y], dim=-1),
        torch.cat([z, zero, -x], dim=-1),
        torch.cat([-y, x, zero], dim=-1),
    ], dim=-2)  # (*, 3, 3)

    # K^2
    K_flat = K.reshape(-1, 3, 3)
    K_sq = (K_flat @ K_flat).reshape(*batch_shape, 3, 3)

    eye = torch.eye(3, dtype=dtype, device=device).expand(
        *batch_shape, 3, 3
    ).contiguous()

    # Rodrigues: R = I + sin(θ)K + (1 - cos(θ))K²
    R = eye + sin_a.unsqueeze(-1) * K + (1 - cos_a).unsqueeze(-1) * K_sq
    return R


def transform_points(points, rotation_matrix):
    """Apply a rotation matrix to a set of points.

    Equivalent to ``Transform3d().rotate(R).transform_points(pts)`` from
    PyTorch3D, which computes ``pts @ R``.

    Args:
        points: (*, N, 3) point coordinates.
        rotation_matrix: (*, 3, 3) rotation matrix.

    Returns:
        (*, N, 3) rotated points.
    """
    return torch.matmul(points, rotation_matrix)


def knn_points(p1, p2, K=1):
    """K-nearest-neighbour search using squared Euclidean distances.

    Pure PyTorch implementation replacing ``pytorch3d.ops.knn.knn_points``.

    Args:
        p1: (B, N, D) first point cloud.
        p2: (B, M, D) second point cloud.
        K:  number of nearest neighbours.

    Returns:
        A named tuple ``KNN(dists, idx, knn)`` where
        - *dists* has shape (B, N, K) — **squared** distances,
        - *idx*   has shape (B, N, K) — indices into *p2*,
        - *knn*   has shape (B, N, K, D) — the K nearest points from *p2*.
    """
    # ||p1_i - p2_j||^2 = ||p1_i||^2 + ||p2_j||^2 - 2 * p1_i . p2_j
    p1_sq = (p1 * p1).sum(dim=-1, keepdim=True)          # (B, N, 1)
    p2_sq = (p2 * p2).sum(dim=-1, keepdim=True)          # (B, M, 1)
    cross = torch.bmm(p1, p2.transpose(1, 2))            # (B, N, M)
    dists_sq = p1_sq + p2_sq.transpose(1, 2) - 2 * cross # (B, N, M)
    dists_sq = dists_sq.clamp(min=0.0)

    knn_dists, knn_idx = dists_sq.topk(K, dim=-1, largest=False)  # (B,N,K)

    # Gather the K nearest neighbour points from p2
    B, M, D = p2.shape
    idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, D)  # (B,N,K,D)
    knn_pts = p2.unsqueeze(1).expand(-1, p1.shape[1], -1, -1)  # (B,N,M,D)
    knn_pts = knn_pts.gather(2, idx_expanded)                   # (B,N,K,D)

    return KNN(dists=knn_dists, idx=knn_idx, knn=knn_pts)
