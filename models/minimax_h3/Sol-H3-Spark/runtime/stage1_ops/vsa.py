"""Native VSA routing/compression with SM121 cuDNN BSA and fused BSHD merge."""
import copy
import hashlib
import importlib
import importlib.metadata
import math
import os
from pathlib import Path

FASTVIDEO_COMMIT = "3d8ac9d14bd697a89ede8f170cbfbca012a9edcc"
VSA_SOURCE_SHA = "b7c6d4e470ee232b50d16d71c2b21806f414f9baa4447434b2da66c94dc0fab9"
COUNTERS = ("vsa_forward_calls", "compression_branch_calls", "block_mask_calls",
            "triton_sparse_kernel_calls", "dense_text_calls", "cudnn_bsa_calls",
            "warmup_reference_calls")
WARMUP_REFERENCE_CALLS = 200
BSA_ATOL = BSA_RTOL = 0.03


def configure_profile(profile, backend="cudnn_bsa"):
    """Call before original basic.configure_environment/build_generator_config."""
    if backend != "cudnn_bsa":
        raise ValueError(f"unsupported selected-attention backend: {backend}")
    profile.vsa = True
    profile.vsa_sparsity = 0.9
    profile.vsa_tile_size = 64
    profile.vsa_kernel = "triton"
    # The official wrapper honors these switches; eliminate inherited SM90
    # choices and its existing mask compaction. A BSA selection replaces only
    # the final forward entry at install_model; native eager policy is unchanged.
    os.environ["FASTVIDEO_VSA_TRITON"] = "1"
    os.environ["FASTVIDEO_KERNEL_VSA_FORCE_TRITON"] = "1"
    os.environ["FASTVIDEO_VSA_TK"] = "0"
    return {"attention_backend": "VIDEO_SPARSE_ATTN_H3", "sparsity": 0.9,
            "tile_size": 64, "prefix_mode": "exempt", "dense_layers": [],
            "dense_first_n_steps": 0, "selected_attention_backend": backend,
            "kernel": "official_triton_tile64" if backend == "triton" else "cudnn_BSA_sm12x_tile64",
            "compression_gates": "all_50_published_replacements_preserved",
            "hardware_adaptation": "SM121 uses " + backend + ", not SM100a",
            "compile_policy": "52 native fullgraph regions around opaque VSA attention",
            "full_official_precision": False}


def metadata_contract(metadata, layer):
    """Scalar/shape checks only: no per-call GPU reduction or synchronization."""
    if (metadata is None or metadata.VSA_sparsity != 0.9 or metadata.tile_elems != 64
            or metadata.exempt is not True or tuple(metadata.dense_layers)
            or not 0 <= int(layer) < 50 or int(metadata.current_timestep) not in range(4)):
        raise RuntimeError("official VSA sparsity/tile/prefix/layer/step policy changed")
    video = int(metadata.num_video_tiles)
    prefix = int(metadata.num_prefix_tiles)
    topk = max(1, min(math.ceil((1 - metadata.VSA_sparsity) * video), video))
    if not 0 < topk < video or prefix < 1:
        raise RuntimeError("VSA must select a strict subset of video tiles with a real AV/text prefix")
    return {"step": int(metadata.current_timestep), "num_prefix_tiles": prefix,
            "num_video_tiles": video, "selected_video_tiles_per_query": topk,
            "video_tile_keep_fraction": topk / video, "prefix_mode": "exempt",
            "tile_size": 64, "sparsity": 0.9}


def snapshot(state):
    """Copy counters before a request, including the first lazy load."""
    return copy.deepcopy(state.get("official_vsa"))


def warmup_reference_required(completed_bsa_calls):
    """The first full request only; no reference execution in formal requests."""
    if completed_bsa_calls < 0:
        raise ValueError("negative BSA call count")
    return completed_bsa_calls < WARMUP_REFERENCE_CALLS


def reverse_active_indices(torch, indices, counts):
    """GPU-only metadata permutation; preserve inactive slots and selected set.

    Native map_to_index emits ascending block IDs, but BSA consumes the active
    list backwards. Reverse only that prefix to match Triton's execution order.
    No tensor item()/CPU synchronization, new mask, or kernel math modification.
    """
    positions = torch.arange(indices.shape[-1], device=indices.device, dtype=torch.int64)
    active_count = counts.unsqueeze(-1)
    gather_positions = torch.where(positions < active_count, active_count - 1 - positions, positions)
    return indices.gather(-1, gather_positions)


def validate_request(before, after):
    if after is None or after.get("active_compression_gates") != 50:
        raise RuntimeError("actual official VSA installation/gates missing")
    first_materialization = before is None
    if first_materialization:
        # arm_worker runs before the first lazy model materialization. The
        # install below initializes every counter to zero; only the completed
        # native calls in `after` are accepted as execution evidence.
        before = {**{key: 0 for key in COUNTERS}, "layer_calls": [0] * 50, "step_calls": [0] * 4}
    backend = after.get("selected_attention_backend", "triton")
    if (backend != "cudnn_bsa"
            or before.get("selected_attention_backend", backend) != backend):
        raise RuntimeError("selected-attention backend changed across a request")
    if (len(after["layer_calls"]) != 50 or len(before["layer_calls"]) != 50
            or len(after["step_calls"]) != 4 or len(before["step_calls"]) != 4):
        raise RuntimeError("official VSA counter geometry changed")
    previous_bsa = before.get("cudnn_bsa_calls", 0)
    expected_reference = 200 if backend == "cudnn_bsa" and previous_bsa == 0 else 0
    expected = dict(zip(COUNTERS, (200, 200, 200, 200 if backend == "triton" else 0,
                                   2, 200 if backend == "cudnn_bsa" else 0, expected_reference)))
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in COUNTERS}
    per_layer = [a - b for a, b in zip(after["layer_calls"], before["layer_calls"])]
    per_step = [a - b for a, b in zip(after["step_calls"], before["step_calls"])]
    if delta != expected or per_layer != [4] * 50 or per_step != [50] * 4:
        raise RuntimeError(f"real VSA request count mismatch: {delta}; layers={per_layer}; steps={per_step}")
    if backend == "cudnn_bsa":
        audit = after.get("warmup_selected_validation", {})
        if (previous_bsa % 200 or audit.get("passed_calls") != WARMUP_REFERENCE_CALLS
                or audit.get("status") != "PASS"
                or after.get("warmup_reference_calls") != WARMUP_REFERENCE_CALLS):
            raise RuntimeError("first full warmup must validate all 200 actual BSA calls; later references forbidden")
        order = after.get("bsa_index_order", {})
        before_order = before.get("bsa_index_order", {})
        match_order = order.get("match_triton_order", False)
        reorder_delta = order.get("metadata_reorder_calls", 0) - before_order.get("metadata_reorder_calls", 0)
        if (before_order.get("match_triton_order", match_order) != match_order
                or reorder_delta != (200 if match_order else 0)):
            raise RuntimeError("BSA active-index order/counter drift")
    bandwidth = after.get("bandwidth_optimization", {})
    previous_bandwidth = before.get("bandwidth_optimization", {})
    bandwidth_delta = {}
    if previous_bandwidth.get("enabled", bandwidth.get("enabled", False)) != bandwidth.get("enabled", False):
        raise RuntimeError("VSA bandwidth optimized changed across a request")
    if bandwidth.get("enabled", False):
        bandwidth_delta = {key: bandwidth.get(key, 0) - previous_bandwidth.get(key, 0)
                           for key in ("bshd_selected_calls", "fused_merge_calls", "warmup_layout_copies")}
        if (backend != "cudnn_bsa" or not after["bsa_index_order"]["match_triton_order"]
                or bandwidth_delta != {"bshd_selected_calls": 200, "fused_merge_calls": 200,
                                       "warmup_layout_copies": 4 * expected_reference}):
            raise RuntimeError(f"VSA bandwidth optimized execution/copy boundary drift: {bandwidth_delta}")
    return {"status": "PASS", "first_materialization": first_materialization,
            "selected_attention_backend": backend,
            "request_counts": delta, "request_layer_calls": per_layer,
            "request_step_calls": per_step, "request_attention_calls": 202,
            "active_compression_gates": 50, "observed_metadata": after["observed_metadata"],
            "count_boundary": "native forwards and actual selected backend entries, excluding fake/meta",
            "warmup_selected_validation": copy.deepcopy(after.get("warmup_selected_validation")),
            "bsa_index_order": copy.deepcopy(after.get("bsa_index_order")),
            "bandwidth_optimization": copy.deepcopy(bandwidth),
            "request_bandwidth_counts": bandwidth_delta,
            "validation_scope": "runtime call counts and selected attention numerical checks"}


def install_model(model, state, backend="cudnn_bsa"):
    """Install after native BF16 LoRA merge and FP8 conversion, before compile."""
    import torch
    from fastvideo.attention.backends import video_sparse_attn_h3 as native
    sparse = importlib.import_module("fastvideo_kernel.triton_kernels.block_sparse_attn_triton")
    if backend != "cudnn_bsa":
        raise ValueError(f"unsupported selected-attention backend: {backend}")
    if ("official_vsa" in state or getattr(native, "_sol_h3_vsa_installed", False)
            or state.get("compression_gates_unused", 0) != 0):
        raise RuntimeError("fresh real-VSA worker required; stripped gates or repeated installation")
    if (os.environ.get("FASTVIDEO_ATTENTION_BACKEND") != "VIDEO_SPARSE_ATTN_H3"
            or os.environ.get("FASTVIDEO_VSA_SM100A") != "0"
            or os.environ.get("FASTVIDEO_VSA_CUTEDSL") != "0"
            or os.environ.get("FASTVIDEO_VSA_TRITON") != "1"
            or os.environ.get("FASTVIDEO_H3_VSA_PROBE")):
        raise RuntimeError("official Triton VSA environment drift")
    source = Path(native.__file__).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != VSA_SOURCE_SHA:
        raise RuntimeError("native VSA source differs from the verified installed official commit")
    bodies, texts = [], []
    gates = []
    for module in model.modules():
        if type(module).__name__ != "MiniMaxH3Attention":
            continue
        impl = module.distributed_attention.attn_impl
        if module.to_gate_compress is None:
            if type(impl).__name__ == "MiniMaxH3VSAImpl":
                raise RuntimeError("VSA implementation missing its published compression gate")
            texts.append(impl)
            continue
        if type(impl).__name__ != "MiniMaxH3VSAImpl":
            raise RuntimeError("compression gate attached to non-VSA attention")
        module._resolve_gate_compress_for_compile()
        weight = module.to_gate_compress.weight
        if module._gate_compress_active is not True or weight.dtype != torch.bfloat16:
            raise RuntimeError("published compression gate missing/zero or unexpectedly quantized")
        bodies.append(impl)
        gates.append({"layer": int(impl.layer_idx), "dtype": str(weight.dtype),
                      "shape": list(weight.shape), "active": True})
    if len(bodies) != 50 or len(texts) != 2 or sorted(i.layer_idx for i in bodies) != list(range(50)):
        raise RuntimeError("expected original 50 VSA body blocks and two dense text blocks")
    record = {key: 0 for key in COUNTERS}
    record.update(active_compression_gates=50, gates=sorted(gates, key=lambda x: x["layer"]),
                  selected_attention_backend=backend,
                  layer_calls=[0] * 50, step_calls=[0] * 4, observed_metadata={},
                  vsa_source=str(source), vsa_source_sha256=VSA_SOURCE_SHA,
                  fastvideo_commit=FASTVIDEO_COMMIT,
                  kernel_entry=("fastvideo_kernel.triton_kernels.block_sparse_attn_triton.triton_block_sparse_attn_forward"
                                if backend == "triton" else "cudnn.BSA.block_sparse_attention_forward"),
                  kernel_source=str(Path(sparse.__file__).resolve()),
                  kernel_source_sha256=hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest(),
                  compile_policy="original loader falls back to eager on SM121 Triton",
                  compression_gate_policy="published BF16 replacements retained; never stripped")
    bsa_forward = None
    match_triton_order = True
    bandwidth_opt = True
    if bandwidth_opt and (backend != "cudnn_bsa" or not match_triton_order):
        raise RuntimeError("bandwidth optimized requires cuDNN BSA and verified Triton-order metadata")
    record["bandwidth_optimization"] = {
        "enabled": bandwidth_opt, "bshd_selected_calls": 0, "fused_merge_calls": 0,
        "warmup_layout_copies": 0,
        "selected_layout": "bshd" if bandwidth_opt else "bhsd",
        "merge": "fused; BF16 product rounded before BF16 add" if bandwidth_opt else "original eager BF16",
        "reference_layout": "bhsd contiguous; Q/K/V/output copies only during first 200 warmup calls",
        "formal_layout_copy_policy": "no BSHD-to-BHSD reference copies",
        "routing_or_compression_math_changed": False}
    if backend == "cudnn_bsa":
        if torch.cuda.get_device_capability() != (12, 1):
            raise RuntimeError("this BSA integration gate is specifically for actual SM121")
        from cudnn import BSA
        bsa_forward = BSA.block_sparse_attention_forward
        bsa_sources = {}
        for name in ("api", "_interface", "csrc.fwd.sm120_blk64.bsa_fwd_sm120"):
            path = Path(importlib.import_module("cudnn.block_sparse_attention." + name).__file__).resolve()
            bsa_sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        record.update(cudnn_frontend_version=importlib.metadata.version("nvidia-cudnn-frontend"),
                      bsa_sources=bsa_sources, triton_source=record["kernel_source"],
                      triton_source_sha256=record["kernel_source_sha256"],
                      kernel_source=bsa_sources["csrc.fwd.sm120_blk64.bsa_fwd_sm120"]["path"],
                      kernel_source_sha256=bsa_sources["csrc.fwd.sm120_blk64.bsa_fwd_sm120"]["sha256"],
                      lse_api="BSA natural log divided by ln(2), preserving original Triton log2 return",
                      bsa_index_order={"match_triton_order": match_triton_order,
                          "metadata_reorder_calls": 0,
                          "original_metadata_order": "ascending KV block IDs from native map_to_index",
                          "bsa_kernel_consumption": "reverse active index prefix",
                          "optimized_metadata_order": "active prefix reversed; inactive slots preserved" if match_triton_order else "original unchanged",
                          "actual_KV_traversal": "ascending matching Triton" if match_triton_order else "descending",
                          "conversion_timing": "GPU arange/where/gather inside each BSA call, included in real Stage1 latency",
                          "kernel_math_or_mask_change": False},
                      warmup_selected_validation={"status": "PENDING", "passed_calls": 0,
                          "max_abs": 0.0, "max_relative_l2": 0.0, "per_call": [],
                          "atol": BSA_ATOL, "rtol": BSA_RTOL,
                          "scope": "first 200 actual model selected-attention calls, first full warmup only",
                          "reference": ("saved unchanged original Triton; identical Q/K/V/counts/block_sizes and selected set; BSA active index prefix reversed"
                                        if match_triton_order else "saved unchanged original Triton, identical Q/K/V/indices/counts/block_sizes"),
                          "tolerance_source": "NVIDIA cudnn-frontend v1.28.0 test_BSA_attention_forward.py:50"})
    state["official_vsa"] = record
    state["attention_module_count"] = 52
    state["compression_gates_active"] = 50
    original_sparse = sparse.triton_block_sparse_attn_forward
    def counted_sparse(*args, _layout="bhsd", **kwargs):
        if backend == "triton":
            result = original_sparse(*args, **kwargs)
            record["triton_sparse_kernel_calls"] += 1
            return result
        # Existing native custom-op has already compacted the bool mask and
        # normalized all sparse metadata to contiguous int32 at this seam.
        if kwargs or len(args) != 6:
            raise RuntimeError("unexpected original selected-attention call signature")
        q, k, v, indices, counts, block_sizes = args
        if _layout not in ("bhsd", "bshd") or (_layout == "bshd" and not bandwidth_opt):
            raise RuntimeError("unapproved selected-attention layout")
        if (torch.is_grad_enabled() or any(x.requires_grad for x in (q, k, v))
                or any(x.dtype != torch.bfloat16 for x in (q, k, v))
                or q.shape[-1] != 128 or q.shape != k.shape or k.shape != v.shape):
            raise RuntimeError("BSA integration requires inference-only BF16 MHA/head128")
        needs_reference = warmup_reference_required(record["cudnn_bsa_calls"])
        bsa_indices = indices
        if match_triton_order:
            bsa_indices = reverse_active_indices(torch, indices, counts)
            record["bsa_index_order"]["metadata_reorder_calls"] += 1
        out, lse_ln = bsa_forward(q, k, v, bsa_indices, q2k_block_nums=counts,
                                 block_sizes=block_sizes, sparse_block_size=64,
                                 softmax_scale=128 ** -0.5, layout=_layout)
        record["cudnn_bsa_calls"] += 1
        if _layout == "bshd":
            record["bandwidth_optimization"]["bshd_selected_calls"] += 1
        if needs_reference:
            # Selected BSA remains BSHD. Only the excluded first-request
            # reference/diagnostic needs the original contiguous BHSD API.
            q_ref, k_ref, v_ref, out_ref = q, k, v, out
            if _layout == "bshd":
                q_ref, k_ref, v_ref, out_ref = (x.transpose(1, 2).contiguous() for x in (q, k, v, out))
                record["bandwidth_optimization"]["warmup_layout_copies"] += 4
            args = (q_ref, k_ref, v_ref, indices, counts, block_sizes)
            reference, reference_lse_log2 = original_sparse(*args, **kwargs)
            record["warmup_reference_calls"] += 1
            actual_f32, ref_f32 = out_ref.float(), reference.float()
            diff = actual_f32 - ref_f32
            max_abs = diff.abs().max().item()
            relative_l2 = (diff.norm() / ref_f32.norm().clamp_min(1e-30)).item()
            passed = bool(torch.isfinite(actual_f32).all() and torch.isfinite(ref_f32).all()
                          and torch.allclose(actual_f32, ref_f32, atol=BSA_ATOL, rtol=BSA_RTOL))
            audit = record["warmup_selected_validation"]
            audit["per_call"].append({"call": record["cudnn_bsa_calls"], "pass": passed,
                                      "shape_bhsd": list(q_ref.shape), "dtype": str(q.dtype),
                                      "optimized_layout": _layout, "optimized_shape": list(q.shape),
                                      "max_abs": max_abs, "relative_l2": relative_l2})
            audit["max_abs"] = max(audit["max_abs"], max_abs)
            audit["max_relative_l2"] = max(audit["max_relative_l2"], relative_l2)
            if not passed:
                audit["status"] = "FAIL"
                raise RuntimeError(f"actual BSA warmup reference FAIL: {audit['per_call'][-1]}")
            audit["passed_calls"] += 1
            if audit["passed_calls"] == WARMUP_REFERENCE_CALLS:
                audit["status"] = "PASS"
        return out, lse_ln / math.log(2.0)
    sparse.triton_block_sparse_attn_forward = counted_sparse
    original_mask = native._build_block_mask
    def counted_mask(scores, num_prefix_tiles, num_video_tiles, VSA_sparsity, exempt):
        if VSA_sparsity != 0.9 or exempt is not True:
            raise RuntimeError("official sparse mask unexpectedly became dense/competing")
        result = original_mask(scores, num_prefix_tiles, num_video_tiles, VSA_sparsity, exempt)
        record["block_mask_calls"] += 1
        return result
    native._build_block_mask = counted_mask
    optimized_forward = None
    if bandwidth_opt:
        from . import fused as fused_ops
        def selected_bshd(q, k, v, block_mask, block_sizes):
            # Same native mask compaction as the original selected wrapper.
            indices, counts = native.map_to_index(block_mask)
            return counted_sparse(q, k, v, indices.to(torch.int32).contiguous(),
                                  counts.to(torch.int32).contiguous(),
                                  block_sizes.to(torch.int32).contiguous(), _layout="bshd")
        # build_forward copies native globals: install counted_mask FIRST.
        optimized_forward = fused_ops.build_forward(native, selected_bshd, bshd=True, fused_merge=True)
        original_merge = optimized_forward.__globals__["_optimized_merge"]
        def counted_merge(*args, **kwargs):
            result = original_merge(*args, **kwargs)
            record["bandwidth_optimization"]["fused_merge_calls"] += 1
            return result
        optimized_forward.__globals__["_optimized_merge"] = counted_merge
        optimized_source = Path(fused_ops.__file__).resolve()
        record["bandwidth_optimization"].update(
            optimized_source=str(optimized_source),
            optimized_sha256=hashlib.sha256(optimized_source.read_bytes()).hexdigest())
    def body_forward(original, layer):
        def observed(query, key, value, gate_compress, attn_metadata):
            if torch.compiler.is_compiling():
                raise RuntimeError("SM121 official Triton VSA must retain native eager execution")
            meta = metadata_contract(attn_metadata, layer)
            if gate_compress is None or gate_compress.dtype != torch.bfloat16:
                raise RuntimeError("real VSA forward did not receive its learned BF16 gate")
            if bandwidth_opt and query.shape[1] != attn_metadata.variable_block_sizes.numel() * 64:
                raise RuntimeError("bandwidth optimized requires logical tile64 transport; no SM100 partner branch")
            record["active_call_context"] = {"layer": layer, **meta}
            result = original(query, key, value, gate_compress, attn_metadata)
            record["vsa_forward_calls"] += 1
            record["compression_branch_calls"] += 1
            record["layer_calls"][layer] += 1
            record["step_calls"][meta["step"]] += 1
            record["observed_metadata"][str(meta["step"])] = meta
            state["attention_calls"] += 1
            return result
        return observed
    for impl in bodies:
        original = (optimized_forward.__get__(impl, type(impl)) if optimized_forward is not None else impl.forward)
        impl.forward = body_forward(original, int(impl.layer_idx))
    def text_forward(original):
        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            record["dense_text_calls"] += 1
            state["attention_calls"] += 1
            return result
        return observed
    for impl in texts:
        impl.forward = text_forward(impl.forward)
    native._sol_h3_vsa_installed = True
    return snapshot(state)
