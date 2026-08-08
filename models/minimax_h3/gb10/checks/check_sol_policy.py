"""Does the released prefix policy recover the quality the bare reference loses?

Two variants on real captured q/k/v against dense flash: the bare threshold, and the released
951-row exact KV sink.

A third was measured and dropped — a local kernel that also handed the prefix's query rows to
flash instead of the sparse grid. It bought +0.0005 of cosine for 7.8% of the time, so the
shipped path is the second row, which the released entry point provides through `sink_tokens`
with no fork at all.
"""
import sys

import paths

paths.setup(need_sol_engine=False)
import torch


from bench_sparse_attn import bench, capture_qkv, released_sol_attn

sol_attn = released_sol_attn()
PREFIX = 951

captured = capture_qkv([25], "832x480", 24)
q, k, v = captured[25]
print(f"q/k/v {tuple(q.shape)}  prefix=951 rows\n")

sdpa = torch.nn.functional.scaled_dot_product_attention
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    dense_fn = lambda: sdpa(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)).transpose(1,2)
    dense_ms = bench(dense_fn)
    reference = dense_fn().float()

variants = {
    "reference (no prefix policy)": lambda t: sol_attn(q, k, v, tau=t),
    "+ exact KV sink":              lambda t: sol_attn(q, k, v, tau=t, sink_tokens=PREFIX,
                                                       sink_start=0),
}
print(f"{'variant':32s} {'tau':>5s} {'ms':>8s} {'vs dense':>9s} {'cos':>10s} {'prefix cos':>11s}")
print(f"{'dense flash':32s} {'-':>5s} {dense_ms:8.2f} {'1.000x':>9s}")
for tau in (1.0, 2.0, 4.0):
    for name in ("reference (no prefix policy)", "+ exact KV sink", "+ dense prefix queries"):
        fn = variants[name]
        out = fn(tau).float()
        ms = bench(lambda: fn(tau))
        cos = torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0)
        pcos = torch.nn.functional.cosine_similarity(
            out[:, :951].flatten(), reference[:, :951].flatten(), dim=0)
        print(f"{name:32s} {tau:5.1f} {ms:8.2f} {dense_ms/ms:8.3f}x {cos:10.6f} {pcos:11.6f}")
    print()
