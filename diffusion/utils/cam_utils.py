# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch


def generate_random_c2w_poses(N, max_translation_range=10.0, dtype=torch.float32, device="cpu"):
    """
    Generates N random 4x4 Camera-to-World (c2w) homogeneous transformation matrices.

    The rotation (R) is generated using unit quaternions for uniform sampling
    of the 3D rotation space.
    The translation (t) is generated randomly within a specified range.

    Args:
        N (int): The number of poses to generate.
        max_translation_range (float): The maximum absolute value for the
                                       x, y, and z translation components.
        dtype (torch.dtype): Data type of the output tensor.
        device (torch.device or str): Device for the output tensor.

    Returns:
        torch.Tensor: A tensor of shape (N, 4, 4) containing the c2w poses.
    """

    # 1. Generate N random unit quaternions for Rotation (R)

    # Generate N random quaternion components (N, 4)
    q = torch.randn(N, 4, dtype=dtype, device=device)

    # Normalize to get N unit quaternions (q / ||q||)
    q = q / torch.linalg.norm(q, dim=1, keepdim=True)

    # Extract components
    a, b, c, d = q.unbind(dim=1)  # a, b, c, d are now (N,) tensors

    # Pre-calculate squared terms
    a2, b2, c2, d2 = a * a, b * b, c * c, d * d

    # Pre-calculate double products
    bc, bd, cd = b * c, b * d, c * d
    ad, ac, ab = a * d, a * c, a * b

    # Construct the (N, 3, 3) rotation matrix batch from quaternions
    #
    R_batch = torch.stack(
        [
            torch.stack([a2 + b2 - c2 - d2, 2 * (bc - ad), 2 * (bd + ac)], dim=1),
            torch.stack([2 * (bc + ad), a2 - b2 + c2 - d2, 2 * (cd - ab)], dim=1),
            torch.stack([2 * (bd - ac), 2 * (cd + ab), a2 - b2 - c2 + d2], dim=1),
        ],
        dim=1,
    )  # (N, 3, 3)

    # 2. Generate N random translation vectors (t)

    # Generate N random numbers for t_x, t_y, t_z in [-range, +range]
    # torch.rand(N, 3) generates uniform random numbers in [0, 1)
    t_batch = (torch.rand(N, 3, dtype=dtype, device=device) * 2 * max_translation_range) - max_translation_range
    # t_batch is now (N, 3)

    # 3. Assemble the (N, 4, 4) homogeneous poses

    # Create the base (N, 4, 4) tensor (identity matrix padded)
    poses = torch.eye(4, dtype=dtype, device=device).repeat(N, 1, 1)

    # Insert the rotation R_batch
    poses[:, :3, :3] = R_batch

    # Insert the translation t_batch
    poses[:, :3, 3] = t_batch

    return poses


def random_rotation_matrix_quaternion(dtype=torch.float32, device="cpu"):
    """
    Generates a random 3x3 rotation matrix using a random unit quaternion.
    This provides a uniform distribution of rotations.
    """
    # 1. Generate four random numbers (components of a random quaternion)
    q = torch.randn(4, dtype=dtype, device=device)

    # 2. Normalize to get a unit quaternion
    q = q / torch.linalg.norm(q)

    # Extract components
    a, b, c, d = q[0], q[1], q[2], q[3]

    # 3. Convert unit quaternion to 3x3 rotation matrix
    # Based on the standard quaternion-to-matrix formula
    R = torch.tensor(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a - b * b + c * c - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a - b * b - c * c + d * d],
        ],
        dtype=dtype,
        device=device,
    )

    # The sum of squared components (a*a + b*b + c*c + d*d) is 1.0,
    # so we don't need to divide the matrix by the norm, R is already correct.

    return R


def get_pose_inverse(T):
    """
    Computes the inverse of a batch of 4x4 homogeneous transformation matrices T
    using the R^T = R^-1 property for rotation matrices.
    T: (..., 4, 4) tensor
    """
    # Extract R and t
    R = T[..., :3, :3]  # (..., 3, 3)
    t = T[..., :3, 3]  # (..., 3)

    # Compute R_inv = R.T
    R_inv = R.transpose(-1, -2)  # (..., 3, 3)

    # Compute t_inv = -R_inv @ t
    # torch.matmul handles the batch dimension (...)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)  # (..., 3)

    # Construct the inverse matrix T_inv
    T_inv = torch.eye(4, dtype=T.dtype, device=T.device).repeat(T.shape[:-2] + (1, 1))
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv

    return T_inv


def compute_raymap(intrinsics, poses, H, W, use_plucker=True):
    """
    Computes a geometry raymap (directions/moments or origins/directions).

    Args:
        intrinsics: (T, 4) tensor [fx, fy, cx, cy]
        poses: (T, 4, 4) tensor [Camera-to-World]
        H, W: int, spatial resolution of the raymap
        use_plucker: bool, if True returns Plucker coords (d, m),
                     else returns (o, d).
    Returns:
        raymap: (T, H, W, 6) tensor
    """
    T = intrinsics.shape[0]
    device = intrinsics.device
    dtype = intrinsics.dtype

    # 1. Create Pixel Grid (T, H, W)
    # indexing='ij' -> y (rows), x (cols)
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    x_grid = x_grid[None, ...].expand(T, -1, -1)
    y_grid = y_grid[None, ...].expand(T, -1, -1)

    # 2. Parse Intrinsics (T, 1, 1)
    fx = intrinsics[:, 0].view(T, 1, 1)
    fy = intrinsics[:, 1].view(T, 1, 1)
    cx = intrinsics[:, 2].view(T, 1, 1)
    cy = intrinsics[:, 3].view(T, 1, 1)

    # 3. Unproject to Camera Frame Directions
    # OpenCV convention: +Z forward, +X right, +Y down
    x_cam = (x_grid - cx) / fx
    y_cam = (y_grid - cy) / fy
    z_cam = torch.ones_like(x_cam)

    # Stack to (T, H, W, 3)
    dirs_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

    # 4. Transform to World Frame
    # R: (T, 3, 3), t: (T, 3)
    R = poses[:, :3, :3]
    t = poses[:, :3, 3]

    # Rotate: d_world = R @ d_cam
    # einsum: t=batch, i=row, j=col, h=height, w=width
    dirs_world = torch.einsum("tij,thwj->thwi", R, dirs_cam)

    # Normalize Direction vectors
    dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

    # 5. Prepare Origins
    # Expand translation t to (T, H, W, 3)
    origins = t.view(T, 1, 1, 3).expand_as(dirs_world)

    if use_plucker:
        # Plucker Moments: m = o x d
        moments = torch.cross(origins, dirs_world, dim=-1)
        # Return (Direction, Moment) -> 6 channels
        return torch.cat([dirs_world, moments], dim=-1)
    else:
        # Standard Ray: (Origin, Direction) -> 6 channels
        return torch.cat([origins, dirs_world], dim=-1)


def _normalize_poses_identity_unit_distance(
    in_c2ws: torch.Tensor,
    ref0_idx: int,
    ref1_idx: int,
):
    """
    Normalize the poses such that the ref0 camera is the identity
    and the ref1 camera is unit distance to the ref0 camera.
    """

    ref0_c2w = in_c2ws[ref0_idx]
    c2ws = torch.einsum("ij,njk->nik", torch.linalg.inv(ref0_c2w), in_c2ws)

    ref1_c2w = c2ws[ref1_idx]
    dist = torch.linalg.norm(ref1_c2w[:3, 3] - ref0_c2w[:3, 3])
    if dist > 1e-2:  # numerically stable
        c2ws[:, :3, 3] /= dist

    return c2ws
