# Copyright 2024 MIT Han Lab
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

import time
from dataclasses import dataclass

import torch
from modules.flash_attn import FlashAttention
from modules.lite_mla import LiteMLA
from modules.triton_lite_mla import TritonLiteMLA
from modules.triton_lite_mla_fwd import TritonLiteMLAFwd
from modules.utils.dtype import get_dtype_from_str
from modules.utils.export_onnx import export_onnx
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F
from torchprofile import profile_macs


@dataclass
class DevelopTritonLiteMLAConfig:
    batch_size: int = 16
    input_size: int = 1024 // 8 // 2
    num_channels: int = 1152
    num_heads: int = 36
    attn_type: str = "LiteMLA"

    device: str = "cuda"
    dtype: str = "fp16"

    profile_macs: bool = False
    test_correctness: bool = False
    warmup_iterations: int = 50
    iterations: int = 1000
    random_weight: bool = True
    backward: bool = False
    autocast: bool = False
    use_cuda_graph: bool = False

    export_model: bool = False
    opset: int = 17
    export_path: str = ""
    export_dtype: str = "fp32"
    export_device: str = "cuda"


def simulate_litemla(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    proj_weight: torch.Tensor,
    proj_bias: torch.Tensor,
    num_heads: int,
    head_dim: int,
    eps: float,
    backward: bool,
):
    B, N, C = x.shape
    qkv = F.linear(x, qkv_weight).reshape(B, N, 3, C).permute(0, 2, 3, 1)
    q, k, v = qkv.unbind(1)  # B, 3, C, N --> B, C, N

    q = q.reshape(B, C // head_dim, head_dim, N)  # b, h, h_d, N
    k = k.reshape(B, C // head_dim, head_dim, N).transpose(-1, -2)  # b, h, N, h_d
    v = v.reshape(B, C // head_dim, head_dim, N)  # b, h, h_d, N

    q = F.relu(q)  # B, h, h_d, N
    k = F.relu(k)

    q, k, v = q.float(), k.float(), v.float()
    if backward:
        k.retain_grad()
        v.retain_grad()
        q.retain_grad()
    v_pad = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
    vk = torch.matmul(v_pad, k)
    if backward:
        vk.retain_grad()
    vk_q = torch.matmul(vk, q)
    vk_q_numerator, vk_q_denominator = vk_q[:, :, :-1], vk_q[:, :, -1:]
    if backward:
        vk_q_numerator.retain_grad()
        vk_q_denominator.retain_grad()
    vk_q_divide = (vk_q_numerator / (vk_q_denominator + eps)).to(x.dtype)

    proj_input = vk_q_divide.view(B, C, N).permute(0, 2, 1)  # B, N, C
    if backward:
        proj_input.retain_grad()
    y = F.linear(proj_input, proj_weight, proj_bias)
    output_dict = {
        "q": q,
        "k": k,
        "v": v,
        "vk": vk,
        "proj_input": proj_input,
        "vk_q_numerator": vk_q_numerator,
        "vk_q_denominator": vk_q_denominator,
        "vk_q_divide": vk_q_divide,
        "y": y,
    }
    return output_dict


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    LiteMLA.fp32_attention = True
    torch.cuda.manual_seed(0)
    torch.manual_seed(0)

    cfg = OmegaConf.structured(DevelopTritonLiteMLAConfig)
    cli_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg, OmegaConf.masked_copy(cli_cfg, cfg.keys()))
    cfg: DevelopTritonLiteMLAConfig = OmegaConf.to_object(cfg)

    torch.set_grad_enabled(cfg.backward)

    device = torch.device("cuda")
    if cfg.autocast:
        dtype = torch.float32
        autocast_dtype = get_dtype_from_str(cfg.dtype)
    else:
        dtype = get_dtype_from_str(cfg.dtype)
        autocast_dtype = None

    if cfg.attn_type == "LiteMLA":
        block = LiteMLA(cfg.num_channels, cfg.num_channels, dim=cfg.num_channels // cfg.num_heads, eps=1e-8)
    elif cfg.attn_type == "TritonLiteMLA":
        block = TritonLiteMLA(cfg.num_channels, cfg.num_heads, eps=1e-8)
    elif cfg.attn_type == "TritonLiteMLAFwd":
        block = TritonLiteMLAFwd(cfg.num_channels, cfg.num_heads, eps=1e-8)
    elif cfg.attn_type == "FlashAttention":
        block = FlashAttention(cfg.num_channels, cfg.num_heads)
    else:
        raise NotImplementedError

    if not cfg.backward:
        block = block.eval()
    block = block.to(device=device, dtype=dtype, memory_format=torch.channels_last)

    if cfg.random_weight:
        for param in block.parameters():
            nn.init.trunc_normal_(param, std=0.001)

    if cfg.profile_macs:
        macs = profile_macs(block, x)
        print(f"macs: {macs}")

    if cfg.export_model:
        export_dtype = get_dtype_from_str(cfg.export_dtype)
        export_device = torch.device(cfg.export_device)
        assert cfg.export_path != ""
        export_onnx(
            block.to(device=export_device, dtype=export_dtype),
            (1, cfg.input_size**2, cfg.num_channels),
            cfg.export_path,
            cfg.opset,
            export_dtype,
            export_device,
        )
    if cfg.test_correctness:
        ref_block = (
            LiteMLA(cfg.num_channels, cfg.num_channels, dim=cfg.num_channels // cfg.num_heads, eps=1e-8)
            .eval()
            .to(device=device, memory_format=torch.channels_last)
        )
        block.load_state_dict(ref_block.state_dict())
        correct = True
        for i in range(10):
            ref_x = torch.randn(
                cfg.batch_size, cfg.input_size**2, cfg.num_channels, device=device, requires_grad=cfg.backward
            )
            x = ref_x.clone().detach().to(dtype=dtype).requires_grad_(cfg.backward)
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.autocast):
                output = block(x)
            ref_output_dict = simulate_litemla(
                ref_x,
                ref_block.qkv.weight,
                ref_block.proj.weight,
                ref_block.proj.bias,
                ref_block.in_dim // ref_block.dim,
                ref_block.dim,
                ref_block.eps,
                cfg.backward,
            )
            ref_output = ref_output_dict["y"]
            if cfg.backward:
                dy = 0.1 * torch.randn_like(output)
                output.backward(dy)
                ref_output.backward(dy.float())
            ref_output_1 = ref_block(ref_x)
            assert torch.allclose(ref_output, ref_output_1)
            output_float = output.float()
            if not torch.allclose(output_float, ref_output):
                correct = False
                max_error_pos = (output_float - ref_output).abs().view(-1).argmax()
                print(f"comparing forward results")
                print(
                    f"max error: {(output_float - ref_output).abs().max()}, mean error: {(output_float - ref_output).abs().mean()}"
                )
                print(f"max error pos: {ref_output.view(-1)[max_error_pos]} {output_float.view(-1)[max_error_pos]}")
            if cfg.backward:
                for name, grad, ref_grad in [
                    ("proj_weight", block.proj.weight.grad, ref_block.proj.weight.grad),
                    ("proj_bias", block.proj.bias.grad, ref_block.proj.bias.grad),
                    ("qkv_weight", block.qkv.weight.grad, ref_block.qkv.weight.grad),
                    ("x", x.grad, ref_x.grad),
                ]:
                    print(f"comparing {name}")
                    grad_float = grad.float()
                    max_error_pos = (grad_float - ref_grad).abs().view(-1).argmax()
                    print(
                        f"max error: {(grad_float - ref_grad).abs().max()}, mean error: {(grad_float - ref_grad).abs().mean()}"
                    )
                    print(f"max error pos: {ref_grad.view(-1)[max_error_pos]} {grad_float.view(-1)[max_error_pos]}")
        if correct:
            print("correct!")
    elif cfg.use_cuda_graph:
        x = torch.randn(
            cfg.batch_size,
            cfg.input_size**2,
            cfg.num_channels,
            device=device,
            dtype=dtype,
            requires_grad=cfg.backward,
        )
        grad_y = 0.1 * torch.randn_like(x)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(cfg.warmup_iterations):
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.autocast):
                    y = block(x)
                if cfg.backward:
                    y.backward(grad_y)
        torch.cuda.current_stream().wait_stream(s)

        g = torch.cuda.CUDAGraph()
        # Sets grads to None before capture, so backward() will create
        # .grad attributes with allocations from the graph's private pool
        with torch.cuda.graph(g):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.autocast):
                y = block(x)
            if cfg.backward:
                y.backward(grad_y)

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(cfg.iterations):
            g.replay()
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"using cuda graph:")
        print(f"each step takes {(end_time-start_time)*1000/cfg.iterations:.2f} ms")
        print(f"max memory allocated: {torch.cuda.max_memory_allocated()/1024**3:.4f} GB")
    else:
        x = torch.randn(
            cfg.batch_size,
            cfg.input_size**2,
            cfg.num_channels,
            device=device,
            dtype=dtype,
            requires_grad=cfg.backward,
        )
        grad_y = 0.1 * torch.randn_like(x)
        for i in range(cfg.warmup_iterations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.autocast):
                y = block(x)
            if cfg.backward:
                y.backward(grad_y)

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(cfg.iterations):
            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=cfg.autocast):
                y = block(x)
            if cfg.backward:
                y.backward(grad_y)
        torch.cuda.synchronize()
        end_time = time.time()
        print(f"each step takes {(end_time - start_time) * 1000 / cfg.iterations:.2f} ms")
        print(f"max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 3:.4f} GB")


if __name__ == "__main__":
    main()
