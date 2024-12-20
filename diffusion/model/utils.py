# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math
import os
import random
import re
import sys
from collections.abc import Iterable
from itertools import repeat

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torchvision import transforms as T


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def set_grad_checkpoint(model, gc_step=1):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.grad_checkpointing = True
        module.grad_checkpointing_step = gc_step

    model.apply(set_attr)


def set_fp32_attention(model):
    assert isinstance(model, nn.Module)

    def set_attr(module):
        module.fp32_attention = True

    model.apply(set_attr)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


def checkpoint_sequential(functions, step, input, *args, **kwargs):

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)


def window_partition(x, window_size):
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w
    )

    return attn


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, tensor.ndim)))


#################################################################################
#                          Token Masking and Unmasking                          #
#################################################################################
def get_mask(batch, length, mask_ratio, device, mask_type=None, data_info=None, extra_len=0):
    """
    Get the binary mask for the input sequence.
    Args:
        - batch: batch size
        - length: sequence length
        - mask_ratio: ratio of tokens to mask
        - data_info: dictionary with info for reconstruction
    return:
        mask_dict with following keys:
        - mask: binary mask, 0 is keep, 1 is remove
        - ids_keep: indices of tokens to keep
        - ids_restore: indices to restore the original order
    """
    assert mask_type in ["random", "fft", "laplacian", "group"]
    mask = torch.ones([batch, length], device=device)
    len_keep = int(length * (1 - mask_ratio)) - extra_len

    if mask_type == "random" or mask_type == "group":
        noise = torch.rand(batch, length, device=device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        ids_removed = ids_shuffle[:, len_keep:]

    elif mask_type in ["fft", "laplacian"]:
        if "strength" in data_info:
            strength = data_info["strength"]

        else:
            N = data_info["N"][0]
            img = data_info["ori_img"]
            # 获取原图的尺寸信息
            _, C, H, W = img.shape
            if mask_type == "fft":
                # 对图片进行reshape，将其变为patch (3, H/N, N, W/N, N)
                reshaped_image = img.reshape((batch, -1, H // N, N, W // N, N))
                fft_image = torch.fft.fftn(reshaped_image, dim=(3, 5))
                # 取绝对值并求和获取频率强度
                strength = torch.sum(torch.abs(fft_image), dim=(1, 3, 5)).reshape(
                    (
                        batch,
                        -1,
                    )
                )
            elif type == "laplacian":
                laplacian_kernel = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float32).reshape(
                    1, 1, 3, 3
                )
                laplacian_kernel = laplacian_kernel.repeat(C, 1, 1, 1)
                # 对图片进行reshape，将其变为patch (3, H/N, N, W/N, N)
                reshaped_image = img.reshape(-1, C, H // N, N, W // N, N).permute(0, 2, 4, 1, 3, 5).reshape(-1, C, N, N)
                laplacian_response = F.conv2d(reshaped_image, laplacian_kernel, padding=1, groups=C)
                strength = laplacian_response.sum(dim=[1, 2, 3]).reshape(
                    (
                        batch,
                        -1,
                    )
                )

        # 对频率强度进行归一化，然后使用torch.multinomial进行采样
        probabilities = strength / (strength.max(dim=1)[0][:, None] + 1e-5)
        ids_shuffle = torch.multinomial(probabilities.clip(1e-5, 1), length, replacement=False)
        ids_keep = ids_shuffle[:, :len_keep]
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_removed = ids_shuffle[:, len_keep:]

    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return {"mask": mask, "ids_keep": ids_keep, "ids_restore": ids_restore, "ids_removed": ids_removed}


def mask_out_token(x, ids_keep, ids_removed=None):
    """
    Mask out the tokens specified by ids_keep.
    Args:
        - x: input sequence, [N, L, D]
        - ids_keep: indices of tokens to keep
    return:
        - x_masked: masked sequence
    """
    N, L, D = x.shape  # batch, length, dim
    x_remain = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    if ids_removed is not None:
        x_masked = torch.gather(x, dim=1, index=ids_removed.unsqueeze(-1).repeat(1, 1, D))
        return x_remain, x_masked
    else:
        return x_remain


def mask_tokens(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unmask_tokens(x, ids_restore, mask_token):
    # x: [N, T, D] if extras == 0 (i.e., no cls token) else x: [N, T+1, D]
    mask_tokens = mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
    x = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
    return x


# Parse 'None' to None and others to float value
def parse_float_none(s):
    assert isinstance(s, str)
    return None if s == "None" else float(s)


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def init_processes(fn, args):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = str(random.randint(2000, 6000))
    print(f'MASTER_ADDR = {os.environ["MASTER_ADDR"]}')
    print(f'MASTER_PORT = {os.environ["MASTER_PORT"]}')
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=args.global_rank, world_size=args.global_size)
    fn(args)
    if args.global_size > 1:
        cleanup()


def mprint(*args, **kwargs):
    """
    Print only from rank 0.
    """
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def cleanup():
    """
    End DDP training.
    """
    dist.barrier()
    mprint("Done!")
    dist.barrier()
    dist.destroy_process_group()


# ----------------------------------------------------------------------------
# logging info.
class Logger:
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name=None, file_mode="w", should_flush=True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, text):
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self):
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self):
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def prepare_prompt_ar(prompt, ratios, device="cpu", show=True):
    # get aspect_ratio or ar
    aspect_ratios = re.findall(r"--aspect_ratio\s+(\d+:\d+)", prompt)
    ars = re.findall(r"--ar\s+(\d+:\d+)", prompt)
    custom_hw = re.findall(r"--hw\s+(\d+:\d+)", prompt)
    if show:
        print("aspect_ratios:", aspect_ratios, "ars:", ars, "hws:", custom_hw)
    prompt_clean = prompt.split("--aspect_ratio")[0].split("--ar")[0].split("--hw")[0]
    if len(aspect_ratios) + len(ars) + len(custom_hw) == 0 and show:
        print(
            "Wrong prompt format. Set to default ar: 1. change your prompt into format '--ar h:w or --hw h:w' for correct generating"
        )
    if len(aspect_ratios) != 0:
        ar = float(aspect_ratios[0].split(":")[0]) / float(aspect_ratios[0].split(":")[1])
    elif len(ars) != 0:
        ar = float(ars[0].split(":")[0]) / float(ars[0].split(":")[1])
    else:
        ar = 1.0
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - ar))
    if len(custom_hw) != 0:
        custom_hw = [float(custom_hw[0].split(":")[0]), float(custom_hw[0].split(":")[1])]
    else:
        custom_hw = ratios[closest_ratio]
    default_hw = ratios[closest_ratio]
    prompt_show = f"prompt: {prompt_clean.strip()}\nSize: --ar {closest_ratio}, --bin hw {ratios[closest_ratio]}, --custom hw {custom_hw}"
    return (
        prompt_clean,
        prompt_show,
        torch.tensor(default_hw, device=device)[None],
        torch.tensor([float(closest_ratio)], device=device)[None],
        torch.tensor(custom_hw, device=device)[None],
    )


def resize_and_crop_tensor(samples: torch.Tensor, new_width: int, new_height: int) -> torch.Tensor:
    orig_height, orig_width = samples.shape[2], samples.shape[3]

    # Check if resizing is needed
    if orig_height != new_height or orig_width != new_width:
        ratio = max(new_height / orig_height, new_width / orig_width)
        resized_width = int(orig_width * ratio)
        resized_height = int(orig_height * ratio)

        # Resize
        samples = F.interpolate(samples, size=(resized_height, resized_width), mode="bilinear", align_corners=False)

        # Center Crop
        start_x = (resized_width - new_width) // 2
        end_x = start_x + new_width
        start_y = (resized_height - new_height) // 2
        end_y = start_y + new_height
        samples = samples[:, :, start_y:end_y, start_x:end_x]

    return samples


def resize_and_crop_img(img: Image, new_width, new_height):
    orig_width, orig_height = img.size

    ratio = max(new_width / orig_width, new_height / orig_height)
    resized_width = int(orig_width * ratio)
    resized_height = int(orig_height * ratio)

    img = img.resize((resized_width, resized_height), Image.LANCZOS)

    left = (resized_width - new_width) / 2
    top = (resized_height - new_height) / 2
    right = (resized_width + new_width) / 2
    bottom = (resized_height + new_height) / 2

    img = img.crop((left, top, right, bottom))

    return img


def mask_feature(emb, mask):
    if emb.shape[0] == 1:
        keep_index = mask.sum().item()
        return emb[:, :, :keep_index, :], keep_index
    else:
        masked_feature = emb * mask[:, None, :, None]
        return masked_feature, emb.shape[2]


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


def get_weight_dtype(mixed_precision):
    if mixed_precision in ["fp16", "float16"]:
        return torch.float16
    elif mixed_precision in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif mixed_precision in ["fp32", "float32"]:
        return torch.float32
    else:
        raise ValueError(f"weigh precision {mixed_precision} is not defined")
