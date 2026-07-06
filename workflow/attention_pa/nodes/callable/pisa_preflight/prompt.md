Use this contract before the first PISA recipe candidate and rerun it after any
material attention-path change.

Read the authoritative local implementation at
`/lustre/fs1/portfolios/nvr/projects/nvr_elm_llm/users/yitongl/code/Sol-LTX-Infer/python/sglang/multimodal_gen/runtime/layers/attention/backends/piecewise_attn.py`.
Copy and adapt the required implementation only inside the experiment-local
target model source; never modify the shared file.

Write `runs/pisa_preflight/attention_map.json` with stable layer ids, attention
types, 50-step call map, Q/K/V and token-layout metadata, dense per-layer and
per-step timing, and sensitive-path hypotheses. Write
`runs/pisa_preflight/backend_probe.json` with the authoritative source path,
commit and SHA-256, copied/adapted source hashes, actual kernel dispatch, block
size, exact/approximate phase counters, density/sparsity, dense fallbacks, OFF
identity, and Q/K/GQA/mask/RoPE compatibility.

The probe must demonstrate the local implementation's
`chunk_reduce_qkv -> taylor_error_block_indices -> piecewise_attn_fwd` path with
both exact selected blocks and `approx_remainder=True`. A keep-or-drop mask,
dense fallback, unused environment configuration, or new implementation based
on external material is `needs_backend_port`, not a successful PISA preflight.
Screening can use the existing GB200 microbenchmark as reference and can run a
single DiT or isolated module, but no result from this node may populate a final
visual recipe without the full diffusion and aligned assessment nodes.
