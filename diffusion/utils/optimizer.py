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
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from came_pytorch import CAME
from mmcv import Config
from mmcv.runner import OPTIMIZER_BUILDERS, OPTIMIZERS, DefaultOptimizerConstructor
from mmcv.runner import build_optimizer as mm_build_optimizer
from mmcv.utils import _BatchNorm, _InstanceNorm
from torch.nn import GroupNorm, LayerNorm
from torch.optim.optimizer import Optimizer

from .logger import get_root_logger


def auto_scale_lr(effective_bs, optimizer_cfg, rule="linear", base_batch_size=256):
    assert rule in ["linear", "sqrt"]
    logger = get_root_logger()
    # scale by world size
    if rule == "sqrt":
        scale_ratio = math.sqrt(effective_bs / base_batch_size)
    elif rule == "linear":
        scale_ratio = effective_bs / base_batch_size
    optimizer_cfg["lr"] *= scale_ratio
    logger.info(f'Automatically adapt lr to {optimizer_cfg["lr"]:.5f} (using {rule} scaling rule).')
    return scale_ratio


@OPTIMIZER_BUILDERS.register_module()
class MyOptimizerConstructor(DefaultOptimizerConstructor):
    def add_params(self, params, module, prefix="", is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module

        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        # sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", 1.0)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", 1.0)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

        for name, param in module.named_parameters(recurse=False):
            base_lr = self.base_lr
            if name == "bias" and not (is_norm or is_dcn_module):
                base_lr *= bias_lr_mult

            # apply weight decay policies
            base_wd = self.base_wd
            if self.base_wd is not None:
                # norm decay
                if is_norm:
                    base_wd *= norm_decay_mult
                # bias lr and decay
                elif name == "bias" and not is_dcn_module:
                    # TODO: current bias_decay_mult will have affect on DCN
                    base_wd *= bias_decay_mult

            param_group = {"params": [param]}
            if not param.requires_grad:
                param_group["requires_grad"] = False
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                logger = get_root_logger()
                logger.warn(f"{prefix} is duplicate. It is skipped since " f"bypass_duplicate={bypass_duplicate}")
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in custom_keys:
                if isinstance(key, tuple):
                    scope, key_name = key
                else:
                    scope, key_name = None, key
                if scope is not None and scope not in f"{prefix}":
                    continue
                if key_name in f"{prefix}.{name}":
                    is_custom = True
                    if "lr_mult" in custom_keys[key]:
                        # if 'base_classes' in f'{prefix}.{name}' or 'attn_base' in f'{prefix}.{name}':
                        #     param_group['lr'] = self.base_lr
                        # else:
                        param_group["lr"] = self.base_lr * custom_keys[key]["lr_mult"]
                    elif "lr" not in param_group:
                        param_group["lr"] = base_lr
                    if self.base_wd is not None:
                        if "decay_mult" in custom_keys[key]:
                            param_group["weight_decay"] = self.base_wd * custom_keys[key]["decay_mult"]
                        elif "weight_decay" not in param_group:
                            param_group["weight_decay"] = base_wd

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if base_lr != self.base_lr:
                    param_group["lr"] = base_lr
                if base_wd != self.base_wd:
                    param_group["weight_decay"] = base_wd
            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module)


def build_optimizer(model, optimizer_cfg):
    # default parameter-wise config
    logger = get_root_logger()

    if hasattr(model, "module"):
        model = model.module
    # set optimizer constructor
    optimizer_cfg.setdefault("constructor", "MyOptimizerConstructor")
    # parameter-wise setting: cancel weight decay for some specific modules
    custom_keys = dict()
    for name, module in model.named_modules():
        if hasattr(module, "zero_weight_decay"):
            custom_keys.update({(name, key): dict(decay_mult=0) for key in module.zero_weight_decay})

    paramwise_cfg = Config(dict(cfg=dict(custom_keys=custom_keys)))
    given_cfg = optimizer_cfg.get("paramwise_cfg")
    if given_cfg:
        paramwise_cfg.merge_from_dict(dict(cfg=given_cfg))
    optimizer_cfg["paramwise_cfg"] = paramwise_cfg.cfg
    # build optimizer
    optimizer = mm_build_optimizer(model, optimizer_cfg)

    weight_decay_groups = dict()
    lr_groups = dict()
    for group in optimizer.param_groups:
        if not group.get("requires_grad", True):
            continue
        lr_groups.setdefault(group["lr"], []).append(group)
        weight_decay_groups.setdefault(group["weight_decay"], []).append(group)

    learnable_count, fix_count = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            learnable_count += 1
        else:
            fix_count += 1
    fix_info = f"{learnable_count} are learnable, {fix_count} are fix"
    lr_info = "Lr group: " + ", ".join([f"{len(group)} params with lr {lr:.5f}" for lr, group in lr_groups.items()])
    wd_info = "Weight decay group: " + ", ".join(
        [f"{len(group)} params with weight decay {wd}" for wd, group in weight_decay_groups.items()]
    )
    opt_info = f"{optimizer.__class__.__name__} Optimizer: total {len(optimizer.param_groups)} param groups, {fix_info}. {lr_info}; {wd_info}."
    logger.info(opt_info)

    return optimizer


@OPTIMIZERS.register_module()
class Lion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)

        super().__init__(params, defaults)

    @staticmethod
    def update_fn(p, grad, exp_avg, lr, wd, beta1, beta2):
        # stepweight decay
        p.data.mul_(1 - lr * wd)

        # weight update
        update = exp_avg.clone().lerp_(grad, 1 - beta1).sign_()
        p.add_(update, alpha=-lr)

        # decay the momentum running average coefficient
        exp_avg.lerp_(grad, 1 - beta2)

    @staticmethod
    def exists(val):
        return val is not None

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):

        loss = None
        if self.exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: self.exists(p.grad), group["params"]):

                grad, lr, wd, beta1, beta2, state = (
                    p.grad,
                    group["lr"],
                    group["weight_decay"],
                    *group["betas"],
                    self.state[p],
                )

                # init state - exponential moving average of gradient values
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                self.update_fn(p, grad, exp_avg, lr, wd, beta1, beta2)

        return loss


@OPTIMIZERS.register_module()
class CAMEWrapper(torch.optim.Optimizer):
    """Implements CAME algorithm.
    This implementation is based on:
    `CAME: Confidence-guided Adaptive Memory Efficient Optimization`
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
        update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return True

    @property
    def supports_flat_params(self):
        return False

    def _get_options(self, param_shape):
        if len(param_shape) == 4:  # Conv layer
            if param_shape[2] == 1 and param_shape[3] == 1:  # 1x1 conv
                return True, "1x1_conv"
            else:  # 3x3 conv or others
                return False, "conv"
        elif len(param_shape) == 2:  # Linear layer, exactly 2D
            return True, "linear"
        return False, "other"

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape

                # factored = self._get_options(grad_shape)
                factored, layer_type = self._get_options(grad_shape)
                # State Initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(grad)
                    if factored:
                        if layer_type == "1x1_conv" or layer_type == "linear":
                            # 1x1 conv and linear layers can be handled the same way
                            state["exp_avg_sq_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_sq_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                            state["exp_avg_res_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_res_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(grad)

                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        # Handle dimensions
                        if len(grad_shape) == 4:  # 1x1 conv
                            update_reshaped = update.squeeze(-1).squeeze(-1)  # Remove last two dimensions
                        else:
                            update_reshaped = update

                        exp_avg_sq_row.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=1), alpha=1.0 - group["betas"][1]
                        )
                        exp_avg_sq_col.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=0), alpha=1.0 - group["betas"][1]
                        )

                    # Approximate calculation
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    if layer_type == "1x1_conv":
                        # Need to reshape back to 4D
                        update = update.view(grad_shape[0], grad_shape[1], 1, 1)
                    update.mul_(grad)
                else:
                    # 3x3 conv or other cases: use standard AdamW approach
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                exp_avg = state["exp_avg"]
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Confidence-guided strategy
                # Calculation of instability
                res = (update - exp_avg) ** 2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]
                    exp_avg_res_col = state["exp_avg_res_col"]

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        # Handle dimensions
                        if len(grad_shape) == 4:  # 1x1 conv
                            res_reshaped = res.squeeze(-1).squeeze(-1)  # Remove last two dimensions
                        else:
                            res_reshaped = res

                        # Update residual statistics
                        exp_avg_res_row.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=1), alpha=1.0 - group["betas"][2]
                        )
                        exp_avg_res_col.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=0), alpha=1.0 - group["betas"][2]
                        )

                    # Approximate calculation
                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    if layer_type == "1x1_conv":
                        # 需要reshape回4D
                        res_approx = res_approx.view(grad_shape[0], grad_shape[1], 1, 1)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                update.mul_(group["lr"])
                p.data.add_(-update)

        return loss


@OPTIMIZERS.register_module()
class CAME8BitWrapper(torch.optim.Optimizer):
    """Implements 8bit-CAME algorithm.

    Args:
        params (iterable): parameters to optimize or dicts defining parameter groups
        lr (float, optional): external learning rate (default: None)
        eps (tuple[float, float]): regularization constants for square gradient
            and instability respectively (default: (1e-30, 1e-16))
        clip_threshold (float): threshold of root-mean-square of
            final gradient update (default: 1.0)
        betas (tuple[float, float, float]): coefficient used for computing running averages of
            update, square gradient and instability (default: (0.9, 0.999, 0.9999)))
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        block_size (int): quantization block size, larger memory efficiency, but may reduce accuracy
        min_8bit_size (int): minimum parameter size for using 8bit quantization, only layers larger than this value will be quantized

    Note:
        1. Only use 8bit quantization for large Linear layers and 1x1 Conv layers
        2. Keep all statistics (exp_avg_sq_row, etc.) in 32bit to ensure stability
        3. Use simple min-max quantization strategy, quantize each block separately
    """

    def __init__(
        self,
        params,
        lr=None,
        eps=(1e-30, 1e-16),
        clip_threshold=1.0,
        betas=(0.9, 0.999, 0.9999),
        weight_decay=0.0,
        block_size=2048,
        min_8bit_size=16384,
    ):
        assert lr > 0.0
        assert all([0.0 <= beta <= 1.0 for beta in betas])

        logger = get_root_logger()
        logger.info(f"Initializing CAME8bit with block_size={block_size}, min_8bit_size={min_8bit_size}")

        defaults = dict(
            lr=lr,
            eps=eps,
            clip_threshold=clip_threshold,
            betas=betas,
            weight_decay=weight_decay,
            block_size=block_size,
            min_8bit_size=min_8bit_size,
        )
        super().__init__(params, defaults)

    def print_layer_info(self, param_shape, use_8bit):
        """Print layer information, including parameter size and whether 8bit quantization is used

        Args:
            param_shape (tuple): parameter shape
            use_8bit (bool): whether 8bit quantization is used
        """
        size = np.prod(param_shape)
        layer_type = "unknown"
        if len(param_shape) == 1:
            layer_type = "1D Layer"
        elif len(param_shape) == 2:
            layer_type = "Linear"
        elif len(param_shape) == 4:
            if param_shape[2] == 1 and param_shape[3] == 1:
                layer_type = "1x1 Conv"
            else:
                layer_type = "Conv"

        status = "8bit" if use_8bit else "32bit"
        print(f"{layer_type} layer with shape {param_shape}: {size:,} params -> using {status}")

    def _should_use_8bit(self, param_shape):
        """Determine if a parameter should be quantized to 8bit

        Rules:
        1. linear layers: parameter size > min_8bit_size
        2. 1x1 conv layers: parameter size > min_8bit_size
        3. other layers: use 32bit
        """
        if len(param_shape) == 2:  # linear layer
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        elif len(param_shape) == 4 and param_shape[2] == 1 and param_shape[3] == 1:
            return param_shape[0] * param_shape[1] > self.defaults["min_8bit_size"]
        return False  # other layers are not quantized

    def _quantize_state(self, state_tensor, block_size=2048):
        """Quantize a state tensor to 8bit

        Args:
            state_tensor: tensor to be quantized
            block_size: quantization block size

        Returns:
            list of quantized data blocks, each block contains:
            - data: uint8 data
            - scale: quantization scale
            - min: minimum value
        """
        if state_tensor.numel() <= 1:
            return state_tensor

        quantized_chunks = []
        for chunk in state_tensor.split(block_size):
            # Calculate quantization parameters
            chunk_min = chunk.min()
            chunk_max = chunk.max()
            scale = (chunk_max - chunk_min) / 255

            # Quantize to 0-255 range
            quantized_chunk = ((chunk - chunk_min) / scale).round().byte()
            quantized_chunks.append({"data": quantized_chunk, "scale": scale, "min": chunk_min})
        return quantized_chunks

    def _dequantize_state(self, quantized_chunks):
        """Dequantize 8bit quantized data to 32bit float

        Args:
            quantized_chunks: list of quantized data blocks

        Returns:
            dequantized 32bit float tensor
        """
        if not isinstance(quantized_chunks, list):
            return quantized_chunks

        chunks = []
        for chunk_dict in quantized_chunks:
            # Dequantize: value = data * scale + min
            chunk = chunk_dict["data"].float() * chunk_dict["scale"] + chunk_dict["min"]
            chunks.append(chunk)
        return torch.cat(chunks)

    def _dequantize_state_first_step(self, quantized_chunks):
        """Efficient dequantization for the first step"""
        if not isinstance(quantized_chunks, list):
            return quantized_chunks

        # 1. Dequantize all chunks to CPU
        dequantized_chunks = []
        for chunk_dict in quantized_chunks:
            chunk = chunk_dict["data"].float() * chunk_dict["scale"] + chunk_dict["min"]
            dequantized_chunks.append(chunk)
            del chunk_dict["data"]
            torch.cuda.empty_cache()

        # 2. Concatenate all chunks
        result = torch.cat(dequantized_chunks)

        del dequantized_chunks
        torch.cuda.empty_cache()

        return result

    def _get_options(self, param_shape):
        if len(param_shape) == 4:
            if param_shape[2] == 1 and param_shape[3] == 1:
                return True, "1x1_conv"
            else:
                return False, "conv"
        elif len(param_shape) == 2:
            return True, "linear"
        return False, "other"

    def _rms(self, tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def step(self, closure=None):
        """Perform a single optimization step

        Main steps:
        1. Determine if 8bit quantization is needed
        2. Update first and second moment estimates
        3. Compute update step
        4. Apply confidence-guided strategy
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                if grad.is_sparse:
                    raise RuntimeError("CAME8bit does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape
                factored, layer_type = self._get_options(grad_shape)

                # Determine if 8bit quantization is used
                use_8bit = self._should_use_8bit(grad_shape)

                # State Initialization
                if len(state) == 0:
                    self.print_layer_info(grad_shape, use_8bit)

                    state["step"] = 0
                    # Only use 8bit quantization for large matrices
                    if use_8bit:
                        state["exp_avg"] = self._quantize_state(torch.zeros_like(grad), group["block_size"])
                    else:
                        state["exp_avg"] = torch.zeros_like(grad)

                    if factored:
                        if layer_type == "1x1_conv" or layer_type == "linear":
                            # Keep row and column statistics in 32bit
                            state["exp_avg_sq_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_sq_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                            state["exp_avg_res_row"] = torch.zeros(grad_shape[0]).type_as(grad)
                            state["exp_avg_res_col"] = torch.zeros(grad_shape[1]).type_as(grad)
                        else:
                            if use_8bit:
                                state["exp_avg_sq"] = self._quantize_state(torch.zeros_like(grad), group["block_size"])
                            else:
                                state["exp_avg_sq"] = torch.zeros_like(grad)
                    else:
                        if use_8bit:
                            state["exp_avg_sq"] = self._quantize_state(torch.zeros_like(grad), group["block_size"])
                        else:
                            state["exp_avg_sq"] = torch.zeros_like(grad)
                    state["RMS"] = 0

                state["step"] += 1
                state["RMS"] = self._rms(p.data)

                exp_avg = self._dequantize_state(state["exp_avg"]) if use_8bit else state["exp_avg"]

                update = (grad**2) + group["eps"][0]
                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]  # 32bit
                    exp_avg_sq_col = state["exp_avg_sq_col"]  # 32bit

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        if len(grad_shape) == 4:
                            update_reshaped = update.squeeze(-1).squeeze(-1)
                        else:
                            update_reshaped = update

                        # Update row and column statistics
                        exp_avg_sq_row.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=1), alpha=1.0 - group["betas"][1]
                        )
                        exp_avg_sq_col.mul_(group["betas"][1]).add_(
                            update_reshaped.mean(dim=0), alpha=1.0 - group["betas"][1]
                        )

                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    if layer_type == "1x1_conv":
                        update = update.view(grad_shape[0], grad_shape[1], 1, 1)
                    update.mul_(grad)
                else:
                    exp_avg_sq = self._dequantize_state(state["exp_avg_sq"]) if use_8bit else state["exp_avg_sq"]
                    exp_avg_sq.mul_(group["betas"][1]).add_(update, alpha=1.0 - group["betas"][1])
                    if use_8bit:
                        state["exp_avg_sq"] = self._quantize_state(exp_avg_sq, group["block_size"])
                    else:
                        state["exp_avg_sq"] = exp_avg_sq
                    update = exp_avg_sq.rsqrt().mul_(grad)

                # Gradient clipping
                update.div_((self._rms(update) / group["clip_threshold"]).clamp_(min=1.0))

                # Update first moment
                exp_avg.mul_(group["betas"][0]).add_(update, alpha=1 - group["betas"][0])

                # Re-quantize (if needed)
                if use_8bit:
                    state["exp_avg"] = self._quantize_state(exp_avg, group["block_size"])
                else:
                    state["exp_avg"] = exp_avg

                # Confidence-guided strategy
                res = (update - exp_avg) ** 2 + group["eps"][1]

                if factored:
                    exp_avg_res_row = state["exp_avg_res_row"]  # 32bit
                    exp_avg_res_col = state["exp_avg_res_col"]  # 32bit

                    if layer_type == "1x1_conv" or layer_type == "linear":
                        if len(grad_shape) == 4:
                            res_reshaped = res.squeeze(-1).squeeze(-1)
                        else:
                            res_reshaped = res

                        # Update residual statistics
                        exp_avg_res_row.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=1), alpha=1.0 - group["betas"][2]
                        )
                        exp_avg_res_col.mul_(group["betas"][2]).add_(
                            res_reshaped.mean(dim=0), alpha=1.0 - group["betas"][2]
                        )

                    res_approx = self._approx_sq_grad(exp_avg_res_row, exp_avg_res_col)
                    if layer_type == "1x1_conv":
                        res_approx = res_approx.view(grad_shape[0], grad_shape[1], 1, 1)
                    update = res_approx.mul_(exp_avg)
                else:
                    update = exp_avg.clone()

                # Weight decay
                if group["weight_decay"] != 0:
                    p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])

                # Apply update
                update.mul_(group["lr"])
                p.data.add_(-update)

        return loss

    def load_state_dict(self, state_dict):
        """Load state dict and convert relevant states to 8bit"""
        super().load_state_dict(state_dict)

        for state in self.state.values():
            for key in [
                "exp_avg",
                "exp_avg_sq",
                "exp_avg_sq_row",
                "exp_avg_sq_col",
                "exp_avg_res_row",
                "exp_avg_res_col",
            ]:
                if key in state:
                    if isinstance(state[key], list):
                        state[key] = [
                            {
                                "data": exp["data"].byte(),  # Convert data to 8bit directly
                                "scale": exp["scale"],  # Keep scale unchanged
                                "min": exp["min"],  # Keep min unchanged
                            }
                            for exp in state[key]
                        ]
                    elif isinstance(state[key], torch.Tensor):
                        # If tensor, keep as 32bit
                        state[key] = state[key].float()  # Ensure 32bit

        del state_dict
        torch.cuda.empty_cache()
