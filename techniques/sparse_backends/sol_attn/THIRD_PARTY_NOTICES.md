# Third-party notices

The files under `sol_attn/_vendor/flash_attn/cute/` and portions of the SM89,
SM90, and SM100 design scaffold derive from the FlashAttention project. Its
BSD-3-Clause license is included at
`sol_attn/sm100/LICENSE.flash-attention`.

The runtime also depends on NVIDIA CUTLASS / CuTe DSL, cuda-python, PyTorch,
and Triton. Those dependencies are not redistributed by this repository and
remain subject to their respective licenses.

The SM120 warp-MMA/TMA execution skeleton and online-softmax helpers are
adapted from NVIDIA cuDNN Frontend's block-sparse-attention reference at commit
`74785165de2da954a2c879a5e3e6f95411c2292d`. That source is licensed under the
Apache License 2.0; adapted files retain the corresponding SPDX header.
