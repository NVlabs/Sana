"""Pose utility functions for SANA-WM benchmark camera accuracy."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch import Tensor

try:
    from pi3.utils.basic import load_images_as_tensor
except ImportError:
    load_images_as_tensor = None


def closed_form_inverse_se3(se3, R=None, T=None):
    """Compute the inverse of each 4x4 or 3x4 SE3 matrix in a batch."""
    is_numpy = isinstance(se3, np.ndarray)
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must have shape (N,4,4) or (N,3,4), got {se3.shape}.")
    if R is None:
        R = se3[:, :3, :3]
    if T is None:
        T = se3[:, :3, 3:]
    if is_numpy:
        r_transposed = np.transpose(R, (0, 2, 1))
        top_right = -np.matmul(r_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        r_transposed = R.transpose(1, 2)
        top_right = -torch.bmm(r_transposed, T)
        inverted_matrix = torch.eye(4, 4, dtype=R.dtype, device=R.device)[None].repeat(len(R), 1, 1)
    inverted_matrix[:, :3, :3] = r_transposed
    inverted_matrix[:, :3, 3:] = top_right
    return inverted_matrix


def align_camera_extrinsics(
    cameras_src: torch.Tensor,
    cameras_tgt: torch.Tensor,
    estimate_scale: bool = True,
    eps: float = 1e-9,
):
    """Align source camera extrinsics to target extrinsics."""
    r_src = cameras_src[:, :, :3]
    r_tgt = cameras_tgt[:, :, :3]
    rr_cov = torch.bmm(r_tgt.transpose(2, 1), r_src).mean(0)
    u, _, v = torch.svd(rr_cov)
    align_t_r = v @ u.t()

    t_src = cameras_src[:, :, 3]
    t_tgt = cameras_tgt[:, :, 3]
    a = torch.bmm(t_src[:, None], r_src)[:, 0]
    b = torch.bmm(t_tgt[:, None], r_src)[:, 0]
    a_mu = a.mean(0, keepdim=True)
    b_mu = b.mean(0, keepdim=True)
    if estimate_scale and a.shape[0] > 1:
        a_centered = a - a_mu
        b_centered = b - b_mu
        align_t_s = (a_centered * b_centered).mean() / (a_centered**2).mean().clamp(eps)
    else:
        align_t_s = 1.0
    align_t_t = b_mu - align_t_s * a_mu
    return align_t_r[None], align_t_t, align_t_s


def apply_transformation(
    cameras_src: torch.Tensor,
    align_t_r: torch.Tensor,
    align_t_t: torch.Tensor,
    align_t_s: float,
    return_extri: bool = True,
) -> torch.Tensor:
    """Apply camera alignment to source extrinsics."""
    r_src = cameras_src[:, :, :3]
    t_src = cameras_src[:, :, 3]
    aligned_r = torch.bmm(r_src, align_t_r.expand(r_src.shape[0], 3, 3))
    align_t_t_expanded = align_t_t[..., None].repeat(r_src.shape[0], 1, 1)
    transformed_t = torch.bmm(r_src, align_t_t_expanded)[..., 0]
    aligned_t = transformed_t + t_src * align_t_s
    if return_extri:
        return torch.cat([aligned_r, aligned_t.unsqueeze(-1)], dim=-1)
    return aligned_r, aligned_t


def calc_roterr_rad(r1: Tensor, r2: Tensor) -> Tensor:
    """Return geodesic rotation error in radians."""
    return (((r1.transpose(-1, -2) @ r2).diagonal(dim1=-1, dim2=-2).sum(-1) - 1) / 2).clamp(-1, 1).acos()


def calc_roterr(r1: Tensor, r2: Tensor) -> Tensor:
    """Return geodesic rotation error in degrees."""
    return torch.rad2deg(calc_roterr_rad(r1, r2))


def calc_transerr(t1: Tensor, t2: Tensor) -> Tensor:
    return (t2 - t1).norm(p=2, dim=-1)


def calc_cammc(rt1: Tensor, rt2: Tensor) -> Tensor:
    return (rt2 - rt1).reshape(-1, 12).norm(p=2, dim=-1)


def metric(c2w_1: Tensor, c2w_2: Tensor) -> tuple[float, float, float]:
    """Compute aligned rotation, translation, and camera-motion errors."""
    w2c_1 = closed_form_inverse_se3(c2w_1)
    w2c_2 = closed_form_inverse_se3(c2w_2)
    align_t_r, align_t_t, align_t_s = align_camera_extrinsics(w2c_2[:, :3, :4], w2c_1[:, :3, :4], estimate_scale=True)
    r_tgt, t_tgt = apply_transformation(w2c_2[:, :3, :4], align_t_r, align_t_t, align_t_s, return_extri=False)
    w2c_2_align = torch.cat([r_tgt, t_tgt.unsqueeze(-1)], dim=-1)
    c2w_2 = closed_form_inverse_se3(w2c_2_align)

    rot_err = calc_roterr(c2w_1[:, :3, :3], c2w_2[:, :3, :3]).mean().item()
    trans_err_rel = calc_transerr(c2w_1[:, :3, 3], c2w_2[:, :3, 3]).mean().item()
    cam_mc_rel = calc_cammc(c2w_1[:, :3, :4], c2w_2[:, :3, :4]).mean().item()
    return rot_err, trans_err_rel, cam_mc_rel


def relative_pose(rt: Tensor, mode: Literal["left", "right"]) -> Tensor:
    if mode == "left":
        return torch.cat([torch.eye(4, device=rt.device).unsqueeze(0), rt[:1].inverse() @ rt[1:]], dim=0)
    if mode == "right":
        return torch.cat([torch.eye(4, device=rt.device).unsqueeze(0), rt[1:] @ rt[:1].inverse()], dim=0)
    raise ValueError(f"Unsupported relative pose mode: {mode}")


def run_pi3_inference_batch(model, video_paths, device, interval=1):
    """Run Pi3 on a batch of videos and return per-video pose outputs."""
    if load_images_as_tensor is None:
        raise ImportError("Pi3 image-loading utilities are unavailable. Install Pi3 or place it on PYTHONPATH.")

    imgs_list = []
    valid_indices = []
    for i, video_path in enumerate(video_paths):
        try:
            img = load_images_as_tensor(video_path, interval=interval).to(device)
            imgs_list.append(img)
            valid_indices.append(i)
        except Exception as exc:
            print(f"Error loading {video_path}: {exc}")

    if not valid_indices:
        return [None] * len(video_paths)

    first_shape = imgs_list[0].shape
    can_batch = all(img.shape == first_shape for img in imgs_list)
    results_map = {}
    dtype = (
        torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    )

    if can_batch:
        try:
            batch_imgs = torch.stack(imgs_list)
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=dtype):
                    res = model(batch_imgs)
            for idx_in_valid, (pose, points) in enumerate(zip(res["camera_poses"], res["points"])):
                results_map[valid_indices[idx_in_valid]] = {
                    "pose": pose,
                    "points": points,
                    "img": imgs_list[idx_in_valid],
                }
        except Exception as exc:
            print(f"Batch inference failed, falling back to sequential: {exc}")
            can_batch = False

    if not can_batch:
        for valid_index, img in zip(valid_indices, imgs_list):
            try:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=dtype):
                        res = model(img[None])
                results_map[valid_index] = {
                    "pose": res["camera_poses"][0],
                    "points": res["points"][0],
                    "img": img,
                }
            except Exception as exc:
                print(f"Error inferencing {video_paths[valid_index]}: {exc}")

    return [results_map.get(i) for i in range(len(video_paths))]
