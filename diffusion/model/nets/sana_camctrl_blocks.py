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

"""Camera-control utility helpers for the Sana-WM bidirectional path.

Only the helpers needed by ``BidirectionalGDNUCPESinglePathLiteLABothTriton``
and ``BidirectionalSoftmaxUCPESinglePathLiteLA`` are kept here.  The model uses
the UCPE (Unified Camera Pose Embedding) formulation, which builds per-pixel
ray transformation matrices from camera poses + intrinsics and applies them to
Q/K/V via block-diagonal projection.

Public surface
--------------
* ``_maybe_drop_cam_branch`` -- inference / training-time camera dropout helper.
* ``_process_camera_conditions_ucpe`` -- builds raymats + 3-channel absmap from
  the raw (B, F, 20) camera-condition tensor.
* ``prepare_prope_fns`` -- precomputes Q/K/V apply functions to share across
  blocks. Only the UCPE branch is implemented.
* ``_prepare_ray_apply_fns`` -- inner helper used by the fused Triton kernels.
* ``compute_fov_from_fx_xi``, ``ucm_unproject_grid_fov``, ``world_to_ray_mats``
  -- imported by fused-camera-GDN ops.
"""

import os
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat

_COMPILE_DISABLE = os.environ.get("GDN_DISABLE_COMPILE", "0") not in ("0", "false")


# ---------------------------------------------------------------------------
# Camera-branch dropout
# ---------------------------------------------------------------------------


def _maybe_drop_cam_branch(camera_conditions, cam_branch_drop_prob, training, device):
    """Optionally zero-out the camera branch during training (drop-path style)."""
    if camera_conditions is None:
        return None
    if not training:
        return camera_conditions
    if not cam_branch_drop_prob:
        return camera_conditions
    if cam_branch_drop_prob >= 1.0:
        return None
    if torch.rand((), device=device) < cam_branch_drop_prob:
        return None
    return camera_conditions


# ---------------------------------------------------------------------------
# UCM (Unified Camera Model) projection / unprojection
# ---------------------------------------------------------------------------


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a pixel coordinate grid of shape ``(H, W, 3)`` or ``(B, H, W, 3)``."""
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\n" "If device is `cpu`, use float32 or float64"
        )
    _xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    _ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([_ys, _xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    if batch is not None:
        grid = repeat(grid, "... -> b ...", b=batch)
    return grid


def ucm_unproject_grid(
    height: int,
    width: int,
    fx: Union[float, torch.Tensor],
    fy: Union[float, torch.Tensor],
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    """Unproject pixel grid into a camera-frame direction vector using the UCM."""
    fx_, fy_, cx_, cy_, xi_ = fx, fy, cx, cy, xi

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).reshape(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    fx, fy, cx, cy, xi = map(to_tensor_flatten, (fx, fy, cx, cy, xi))
    B = max(fx.shape[0], fy.shape[0], cx.shape[0], cy.shape[0], xi.shape[0])
    fx = fx.expand(B)
    fy = fy.expand(B)
    cx = cx.expand(B)
    cy = cy.expand(B)
    xi = xi.expand(B)

    grid = create_grid(height=height, width=width, batch=B, dtype=dtype, device=device)
    u = grid[..., 0]
    v = grid[..., 1]
    fx = fx[:, None, None]
    fy = fy[:, None, None]
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    xi = xi[:, None, None]
    x = (u - cx) / fx
    y = (v - cy) / fy
    if not y_down:
        y = -y
    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)
    X = gamma * x
    Y = gamma * y
    Z = gamma - xi
    d_cam = torch.stack([X, Y, Z], dim=-1)
    is_scalar_input = all(not torch.is_tensor(p) for p in (fx_, fy_, cx_, cy_, xi_))
    if is_scalar_input:
        return d_cam[0]
    else:
        return d_cam


def compute_fx_from_fov_xi(
    x_fov: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Recover focal length ``fx`` from horizontal FoV (degrees) + UCM xi."""

    def to_tensor_flatten(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype).view(-1)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_flatten(x_fov)
    xi = to_tensor_flatten(xi)
    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.expand(B)
    xi = xi.expand(B)
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def compute_fov_from_fx_xi(
    fx: Union[torch.Tensor, float],
    xi: Union[torch.Tensor, float],
    width: int,
    device="cpu",
    dtype=torch.float32,
):
    """Inverse of :func:`compute_fx_from_fov_xi`."""

    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx = to_tensor_1d(fx).reshape(-1)
    xi = to_tensor_1d(xi).reshape(-1)
    B = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(B)
    xi = xi.expand(B)
    A = 2.0 * fx / width
    phi = torch.atan(1.0 / A)
    denom = torch.sqrt(A * A + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi
    x_fov = torch.rad2deg(2.0 * theta)
    return x_fov


def ucm_unproject_grid_fov(
    x_fov: Union[float, torch.Tensor],
    y_fov: Union[float, torch.Tensor],
    xi: Union[float, torch.Tensor],
    height: int,
    width: int,
    cx: Union[float, torch.Tensor],
    cy: Union[float, torch.Tensor],
    device: Union[torch.device, str] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Unproject grid with intrinsics expressed as FoV (degrees) + xi."""
    is_batched = any(torch.is_tensor(p) and p.numel() > 1 for p in [x_fov, y_fov, xi, cx, cy])
    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, device, dtype)
    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        xi=xi if torch.is_tensor(xi) else torch.tensor([xi], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        y_down=True,
    )
    if not is_batched:
        d_cam = d_cam[0]
    return d_cam


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """Project 3D points in camera frame to UCM image plane."""
    r = torch.sqrt(X * X + Y * Y + Z * Z)

    def reshape_param(p, target):
        if torch.is_tensor(p):
            if p.numel() == 1:
                return p
            if p.ndim == 1 and target.ndim == 4:
                return p.view(target.shape[0], target.shape[1], 1, 1)
            while p.ndim < target.ndim:
                p = p.unsqueeze(-1)
        return p

    xi = reshape_param(xi, X)
    fx = reshape_param(fx, X)
    fy = reshape_param(fy, X)
    cx = reshape_param(cx, X)
    cy = reshape_param(cy, X)

    alpha = Z + xi * r
    du = fx * (X / alpha) + cx
    dv = fy * (Y / alpha) + cy
    return du, dv


def project_ucm_points_fov(X, Y, Z, x_fov, y_fov, xi, height, width, cx, cy):
    """Project 3D points in camera frame to UCM image plane using FoV-based intrinsics."""
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    fy = compute_fx_from_fov_xi(y_fov, xi, height, X.device, X.dtype)
    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


# ---------------------------------------------------------------------------
# Per-pixel ray transformation (world <-> ray) used by UCPE
# ---------------------------------------------------------------------------


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [H, W, 3], [B, H, W, 3], or [B, T, H, W, 3]
    c2w: torch.Tensor,  # [B, T, 4, 4]
) -> torch.Tensor:
    """Build per-pixel ``ray<-world`` transforms from camera unit rays + C2W poses."""
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    if d_cam.ndim == 4:
        B, H, W, _ = d_cam.shape
        T = c2w.shape[1]
        d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)
    elif d_cam.ndim == 5:
        B, T, H, W, _ = d_cam.shape
    else:
        raise ValueError(f"Unsupported d_cam shape: {d_cam.shape}")

    device = d_cam.device
    dtype = d_cam.dtype
    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = torch.einsum("btij,bthwj->bthwi", R_cam, d_cam)
    cam_y = R_cam[..., :, 1]
    cam_y = repeat(cam_y, "b t c -> b t h w c", h=H, w=W)
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")
    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)
    t_w2l = -torch.einsum("bthwij,bthwj->bthwi", R_w2l, t_world)
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    y_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    cx: torch.Tensor,
    cy: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """Compute UCPE absolute embedding maps ``(up_map, lat_map)``.

    ``up_map`` is a 2-channel projected up-direction; ``lat_map`` is a 1-channel
    latitude. Concatenated they form the 3-channel absmap consumed by the
    camera branch.
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        y_fov=y_fov,
        xi=xi,
        height=height,
        width=width,
        cx=cx,
        cy=cy,
        device=device,
        dtype=torch.float32,
    )

    if d_cam.ndim == 3:
        d_cam_exp = repeat(d_cam, "H W C -> B T H W C", B=B, T=T)
    elif d_cam.ndim == 4:
        if d_cam.shape[0] == B * T:
            d_cam_exp = d_cam.view(B, T, height, width, 3)
        else:
            d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)
    else:
        d_cam_exp = d_cam

    mask_exp = d_cam_exp.isnan().any(dim=-1, keepdim=True)
    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)
    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)
    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)
    delta_t = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta_t)
    sin_eps = torch.sin(delta_t)
    v_rot = (
        v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    )
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]
    du, dv = project_ucm_points_fov(
        Xs,
        Ys,
        Zs,
        x_fov=x_fov.float(),
        y_fov=y_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
        cx=cx.float(),
        cy=cy.float(),
    )
    grid = create_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )
    grid_x = grid[..., 0].unsqueeze(1)
    grid_y = grid[..., 1].unsqueeze(1)
    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)
    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)
    up_map = up_map.masked_fill(mask_exp, 0.0)
    lat_map = lat_map.masked_fill(mask_exp, 0.0)
    return up_map, lat_map


def _process_camera_conditions_ucpe(camera_conditions, B, HW, patch_size):
    """Convert ``(B, F, 20)`` camera conditions (C2W flat + fx,fy,cx,cy) into
    ``(raymats, absmap)``.

    ``raymats`` is ``(B, F, H, W, 4, 4)`` ``ray<-world`` transforms; ``absmap``
    is ``(B, F, H, W, 3)`` (up_map 2-ch + lat_map 1-ch).
    """
    F_dim = camera_conditions.shape[1]
    c2w_flat = camera_conditions[..., :16]
    C_to_W = c2w_flat.view(B, F_dim, 4, 4)

    fx = camera_conditions[..., 16]
    fy = camera_conditions[..., 17]
    cx = camera_conditions[..., 18]
    cy = camera_conditions[..., 19]
    H_dim, W_dim = HW[1], HW[2]
    image_width = W_dim * patch_size[2]
    image_height = H_dim * patch_size[1]

    # xi is fixed at 0 (pinhole) in this stack.
    xi = torch.zeros((B, F_dim), device=camera_conditions.device, dtype=camera_conditions.dtype)
    x_fov = compute_fov_from_fx_xi(
        fx, xi, image_width, device=camera_conditions.device, dtype=camera_conditions.dtype
    ).view(B, F_dim)
    y_fov = compute_fov_from_fx_xi(
        fy, xi, image_height, device=camera_conditions.device, dtype=camera_conditions.dtype
    ).view(B, F_dim)

    d_cam = ucm_unproject_grid_fov(
        x_fov,
        y_fov,
        xi,
        H_dim,
        W_dim,
        cx / patch_size[2],
        cy / patch_size[1],
        device=camera_conditions.device,
        dtype=camera_conditions.dtype,
    )
    if d_cam.ndim == 4 and d_cam.shape[0] == B * F_dim:
        d_cam = d_cam.view(B, F_dim, H_dim, W_dim, 3)

    raymats = world_to_ray_mats(d_cam, C_to_W)  # [B, F, H, W, 4, 4]

    up_map, lat_map = compute_up_lat_map(
        R=C_to_W[..., :3, :3],
        x_fov=x_fov,
        y_fov=y_fov,
        xi=xi,
        height=image_height,
        width=image_width,
        cx=cx,
        cy=cy,
        device=camera_conditions.device,
    )
    absmap = torch.cat([up_map, lat_map], dim=-1)  # (B, F, H, W, 3)

    return raymats, absmap


# ---------------------------------------------------------------------------
# Block-diagonal apply primitives shared by camera and main branches
# ---------------------------------------------------------------------------


@torch.compile(disable=_COMPILE_DISABLE)
def _apply_ray_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, seqlen, 4, 4)
) -> torch.Tensor:
    """Apply a per-token 4x4 projection matrix to feature channels grouped by 4."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    D = matrix.shape[-1]
    return torch.einsum(
        "bnij,bhnkj->bhnki",
        matrix,
        feats.reshape(batch, num_heads, seqlen, -1, D),
    ).reshape(feats.shape)


@torch.compile(disable=_COMPILE_DISABLE)
def _apply_tiled_projmat(
    feats: torch.Tensor,  # (batch, num_heads, seqlen, feat_dim)
    matrix: torch.Tensor,  # (batch, cameras, D, D)
) -> torch.Tensor:
    """Apply a per-camera projection matrix tiled across the spatial axis."""
    (batch, num_heads, seqlen, feat_dim) = feats.shape
    D = matrix.shape[-1]
    assert feat_dim % D == 0, f"feat_dim={feat_dim} must be divisible by D={D}"
    if matrix.shape[1] == seqlen:
        feats_ = feats.view(batch, num_heads, seqlen, feat_dim // D, D)
        out = torch.einsum("btij,bntpj->bntpi", matrix, feats_)
        return out.reshape(feats.shape)

    cameras = matrix.shape[1]
    assert seqlen >= cameras and seqlen % cameras == 0
    return torch.einsum(
        "bcij,bncpkj->bncpki",
        matrix,
        feats.reshape((batch, num_heads, cameras, -1, feat_dim // D, D)),
    ).reshape(feats.shape)


@torch.compile(disable=_COMPILE_DISABLE)
def _apply_complex_rope(
    hidden_states: torch.Tensor,
    freqs: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply complex RoPE (compiled: fuses fp64 cast + view_as_complex + multiply chain)."""
    x_real = hidden_states.to(torch.float64)
    if x_real.stride(-1) != 1:
        x_real = x_real.contiguous()
    x_complex = torch.view_as_complex(x_real.unflatten(-1, (-1, 2)))
    if inverse:
        freqs = freqs.conj()
    x_out = torch.view_as_real(x_complex * freqs).flatten(-2, -1)
    return x_out.type_as(hidden_states)


def _apply_block_diagonal(
    feats: torch.Tensor,  # (..., dim)
    func_size_pairs: List[Tuple[Callable[[torch.Tensor], torch.Tensor], int]],
) -> torch.Tensor:
    """Apply a block-diagonal function: split features by sizes, transform each, concat."""
    funcs, block_sizes = zip(*func_size_pairs)
    assert feats.shape[-1] == sum(block_sizes)
    x_blocks = torch.split(feats, block_sizes, dim=-1)
    out = torch.cat(
        [f(x_block) for f, x_block in zip(funcs, x_blocks)],
        dim=-1,
    )
    assert out.shape == feats.shape, "Input/output shapes should match."
    return out


def _invert_SE3(transforms: torch.Tensor) -> torch.Tensor:
    """Closed-form inverse of a 4x4 SE(3) batch."""
    assert transforms.shape[-2:] == (4, 4)
    Rinv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = Rinv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", Rinv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


# ---------------------------------------------------------------------------
# UCPE apply-fn preparation
# ---------------------------------------------------------------------------


def _prepare_ray_apply_fns(
    head_dim: int,
    P: torch.Tensor,  # (batch, seqlen, 4, 4) P = ray<-world
    P_T: torch.Tensor,  # (batch, seqlen, 4, 4) P_T = world<-ray
    P_inv: torch.Tensor,  # (batch, seqlen, 4, 4) P_inv = world<-ray
    rotary_emb: Optional[torch.Tensor] = None,
    apply_vo: bool = True,
) -> Tuple[Callable, Callable, Callable]:
    """Build ``(apply_q, apply_kv, apply_o)`` block-diagonal callables for UCPE."""
    if rotary_emb is not None:
        rope_fn = partial(_apply_complex_rope, freqs=rotary_emb, inverse=False)
        rope_fn_inv = partial(_apply_complex_rope, freqs=rotary_emb, inverse=True)
    else:
        rope_fn = lambda x: x
        rope_fn_inv = lambda x: x

    transforms_q = [
        (partial(_apply_ray_projmat, matrix=P_T), head_dim // 2),
        (rope_fn, head_dim // 2),
    ]
    transforms_kv = [
        (partial(_apply_ray_projmat, matrix=P_inv), head_dim // 2),
        (rope_fn, head_dim // 2),
    ]
    if apply_vo:
        transforms_o = [
            (partial(_apply_ray_projmat, matrix=P), head_dim // 2),
            (rope_fn_inv, head_dim // 2),
        ]
    else:
        transforms_o = lambda x: x

    apply_fn_q = partial(_apply_block_diagonal, func_size_pairs=transforms_q)
    apply_fn_kv = partial(_apply_block_diagonal, func_size_pairs=transforms_kv)
    apply_fn_o = partial(_apply_block_diagonal, func_size_pairs=transforms_o) if apply_vo else transforms_o

    return apply_fn_q, apply_fn_kv, apply_fn_o


def _slice_rope_for_cam(
    rotary_emb: Optional[torch.Tensor],
    head_dim: int,
    rope_dim: int,
) -> Optional[torch.Tensor]:
    """Re-slice WAN RoPE frequencies for a smaller rope_dim using the same (T, H, W) split."""
    if rotary_emb is None:
        return None
    orig_t_size = head_dim // 2 - 2 * (head_dim // 6)
    orig_h_size = head_dim // 6
    new_t_size = rope_dim // 2 - 2 * (rope_dim // 6)
    new_h_size = rope_dim // 6
    new_w_size = rope_dim // 6
    t_part = rotary_emb[..., :new_t_size]
    h_part = rotary_emb[..., orig_t_size : orig_t_size + new_h_size]
    w_part = rotary_emb[..., orig_t_size + orig_h_size : orig_t_size + orig_h_size + new_w_size]
    return torch.cat([t_part, h_part, w_part], dim=-1)


def prepare_prope_fns(
    camctrl_type: str,
    head_dim: int,
    camera_conditions: torch.Tensor,
    HW: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    rotary_emb: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[Callable, Callable, Callable]:
    """Precompute UCPE apply functions once for a batch (shared across all blocks).

    Only ``camctrl_type == "UCPE"`` is supported.  Accepts either precomputed
    matrices (``cam_pos_embeds`` dict with ``P``, ``P_inv``, ``pos_embeds_cam``)
    or raw camera conditions + optional raymats.
    """
    if camctrl_type != "UCPE":
        raise ValueError(f"Unsupported camctrl_type for prepare_prope_fns: {camctrl_type}")

    B = camera_conditions.shape[0]

    # Priority 1: use precomputed matrices.
    if "cam_pos_embeds" in kwargs and kwargs["cam_pos_embeds"] is not None:
        cam_pos_embeds = kwargs["cam_pos_embeds"]
        P = cam_pos_embeds.get("P")
        P_inv = cam_pos_embeds.get("P_inv")
        rotary_emb_cam = cam_pos_embeds.get("pos_embeds_cam")

        if P is not None and P_inv is not None:
            if P.ndim == 3:
                P = P.unsqueeze(0).repeat(B, 1, 1, 1)
            if P_inv.ndim == 3:
                P_inv = P_inv.unsqueeze(0).repeat(B, 1, 1, 1)

            P_T = P.transpose(-1, -2)

            if rotary_emb_cam is not None and rotary_emb_cam.ndim == 3:
                rotary_emb_cam = rotary_emb_cam.unsqueeze(0).repeat(B, 1, 1, 1)
            elif rotary_emb_cam is None and rotary_emb is not None:
                rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2)
            elif rotary_emb_cam is None:
                rotary_emb_cam = rotary_emb

            return _prepare_ray_apply_fns(head_dim, P, P_T, P_inv, rotary_emb=rotary_emb_cam)

    # Priority 2: online path.
    if "raymats" in kwargs and kwargs["raymats"] is not None:
        raymats = kwargs["raymats"]
    else:
        raymats, _ = _process_camera_conditions_ucpe(camera_conditions, B, HW, patch_size)
    raymats = raymats.reshape(B, -1, 4, 4)

    P = raymats
    P_T = P.transpose(-1, -2)
    P_inv = _invert_SE3(P)

    rotary_emb_cam = _slice_rope_for_cam(rotary_emb, head_dim, head_dim // 2) if rotary_emb is not None else None

    return _prepare_ray_apply_fns(head_dim=head_dim, P=P, P_T=P_T, P_inv=P_inv, rotary_emb=rotary_emb_cam)
