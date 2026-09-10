"""Build the fixed Stage2 AV context cache using native INT8 Gemma/connector math.

Run offline in the Stage2 environment on one SM121 GPU. Model loading is
resident for this isolated encode; generation never loads these weights.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

from .prompt_cache import FIXED_PROMPT, RECIPE, fingerprint, load_cache, validate_payload


def prompt_stats(tokenizer, prompt):
    """Count tokens with the same native tokenizer and BOS handling as encode."""
    if tokenizer is None:
        raise RuntimeError("Gemma tokenizer is unavailable")
    hf_tokenizer = tokenizer.tokenizer
    raw_ids = hf_tokenizer(prompt, truncation=False, padding=False).input_ids
    bos_id = hf_tokenizer.bos_token_id
    raw_count = len(raw_ids) + int(not raw_ids or raw_ids[0] != bos_id)
    kept_count = sum(int(weight) for _, weight in tokenizer.tokenize_with_weights(prompt)["gemma"])
    max_tokens = int(tokenizer.max_length)
    stats = {
        "characters": len(prompt), "whitespace_words": len(prompt.split()),
        "raw_tokens_including_bos": raw_count, "kept_tokens": kept_count,
        "max_tokens": max_tokens, "truncated": raw_count > max_tokens,
    }
    if stats["truncated"]:
        raise ValueError(f"fixed prompt requires {raw_count} tokens; native limit is {max_tokens}")
    return stats


def _with_int8_policy(builder, policy):
    from ltx_core.loader.sd_ops import SDOps
    if builder.model_sd_ops is None or policy.sd_ops is None:
        raise RuntimeError("INT8 policy requires both native and quantization key operations")
    combined = SDOps(
        name=f"{builder.model_sd_ops.name}+{policy.sd_ops.name}",
        mapping=(*builder.model_sd_ops.mapping, *policy.sd_ops.mapping),
        allowed_keys=builder.model_sd_ops.allowed_keys,
    )
    return builder.with_sd_ops(combined).with_module_ops((*builder.module_ops, *policy.module_ops))


def _validate_model(model, expected, torch):
    from .cache_ops.int8_policy import DevInt8ConvRotLinear
    quantized = [module for module in model.modules() if isinstance(module, DevInt8ConvRotLinear)]
    if len(quantized) != expected:
        raise RuntimeError(f"expected {expected} INT8 linears, got {len(quantized)}")
    if any(module.weight.dtype != torch.int8 or module.weight_scale.dtype != torch.float32
           for module in quantized):
        raise RuntimeError("INT8 weight/FP32 scale layout changed")
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_meta or tensor.device.type != "cuda":
            raise RuntimeError(f"offline model tensor is not CUDA resident: {name}")
    return {"int8_linears": len(quantized), "weight_dtype": "torch.int8",
            "scale_dtype": "torch.float32", "all_tensors_cuda_resident": True}


def _text_builders(gemma_path, connector_path):
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator, get_gemma_ops, resolve_gemma_weight_paths,
    )
    from .cache_ops.int8_policy import (
        build_dev_int8_convrot_embeddings_policy, build_gemma_int8_convrot_policy,
    )

    # Native operations initialize Gemma buffers and its embedded tokenizer.
    native_sd, native_module_ops = get_gemma_ops(str(gemma_path))
    registry = ModelRegistry(cache_models=True, cache_weights=False)
    text = SingleGPUModelBuilder(
        model_path=resolve_gemma_weight_paths(str(gemma_path)),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(gemma_path)),
        model_sd_ops=native_sd, module_ops=native_module_ops, registry=registry)
    text = _with_int8_policy(text, build_gemma_int8_convrot_policy(gemma_path, native_sd))
    connector = SingleGPUModelBuilder(
        model_path=(str(connector_path), str(gemma_path)),
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(str(gemma_path)),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS, registry=registry)
    connector = _with_int8_policy(
        connector, build_dev_int8_convrot_embeddings_policy(connector_path))
    return text, connector


def build_cache(paths: dict, output):
    """Write one checked feature-only cache; fail if the destination exists."""
    output = Path(output).expanduser().resolve()
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(output)
    gemma_path = Path(paths["offline_gemma"]).expanduser().resolve(strict=True)
    connector_path = Path(paths["offline_connector"]).expanduser().resolve(strict=True)
    for name, path in (("text_encoder", gemma_path), ("connector", connector_path)):
        if path.name != RECIPE[name] or path.stat().st_size != RECIPE[name + "_bytes"]:
            raise ValueError(f"unexpected offline {name} checkpoint identity")
    if "gemma_tokenizer" in paths and Path(paths["gemma_tokenizer"]).resolve() != gemma_path:
        raise ValueError("tokenizer must come from the exact INT8 Gemma checkpoint")

    import torch
    if torch.cuda.device_count() != 1 or tuple(torch.cuda.get_device_capability()) != (12, 1):
        raise RuntimeError("offline cache generation requires one SM121 GPU")
    started = time.monotonic_ns()
    output.parent.mkdir(parents=True, exist_ok=True)
    gemma_builder, connector_builder = _text_builders(gemma_path, connector_path)
    gemma = connector = None
    try:
        # No model-wide BF16 cast: it would corrupt row-wise FP32 scales.
        gemma = gemma_builder.build(device=torch.device("cuda:0"), dtype=None).eval().requires_grad_(False)
        gemma_audit = _validate_model(gemma, 328, torch)
        connector = connector_builder.build(device=torch.device("cuda:0"), dtype=None).eval().requires_grad_(False)
        connector.feature_extractor.to(dtype=torch.bfloat16)
        connector_audit = _validate_model(connector, 96, torch)
        stats = prompt_stats(gemma.tokenizer, FIXED_PROMPT)
        with torch.inference_mode():
            outputs = gemma.encode([FIXED_PROMPT])
            if len(outputs) != 1:
                raise RuntimeError("native Gemma must return exactly one prompt")
            hidden_states, mask = outputs[0]
            processed = connector.process_hidden_states(hidden_states, mask)
            contexts = {name: getattr(processed, name + "_encoding").detach().cpu().contiguous()
                        for name in ("video", "audio")}
        torch.cuda.synchronize()
        payload = {
            "schema_version": 1, "prompt": FIXED_PROMPT, "recipe": dict(RECIPE),
            "fingerprint": fingerprint(FIXED_PROMPT), "contexts": contexts,
            "prompt_stats": stats,
            "tensor_manifest": {name: {"shape": list(t.shape), "dtype": str(t.dtype)}
                                for name, t in contexts.items()},
            "generation": {
                "gemma_build_count": 1, "gemma_encode_calls": 1,
                "connector_process_calls": 1, "video_model_load_count": 0,
                "loading_strategy": "isolated resident INT8 offline encode",
                "gemma_input_mask_shape": list(mask.shape),
                "gemma_input_mask_dtype": str(mask.dtype),
                "connector_output_mask_shape": list(processed.attention_mask.shape),
                "connector_output_mask_dtype": str(processed.attention_mask.dtype),
                "mask_reapplied_to_contexts": False,
                "gemma": gemma_audit, "connector": connector_audit,
                "checkpoints": {name: {"path": str(path), "bytes": path.stat().st_size}
                                for name, path in (("text_encoder", gemma_path), ("connector", connector_path))},
            },
        }
        validate_payload(payload, prompt=FIXED_PROMPT, torch_module=torch)
        with output.open("xb") as stream:
            torch.save(payload, stream)
        cache = load_cache(output, prompt=FIXED_PROMPT, torch_module=torch)
        if any(not torch.equal(cache.payload["contexts"][name], tensor)
               for name, tensor in contexts.items()):
            raise RuntimeError("serialized cache differs from the native connector outputs")
        receipt = {
            "status": "PASS", "output": str(output), "prompt_cache_path": str(output),
            "bytes": output.stat().st_size, "elapsed_s": (time.monotonic_ns() - started) / 1e9,
            "saved_contexts_bitwise_equal_fresh": True, "video_model_load_count": 0,
            "gemma_encode_calls": 1, "connector_process_calls": 1,
            "fingerprint": fingerprint(FIXED_PROMPT), "tensor_manifest": payload["tensor_manifest"],
            "loading_strategy": "isolated resident INT8 offline encode",
            "numerical_reference": "fresh native INT8 encoder plus native INT8 AV connector",
            "streaming_reference_compared": False,
        }
        with output.with_suffix(".json").open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        return receipt
    finally:
        # Only these two locally constructed model shells are disposed.
        for model in (gemma, connector):
            if model is not None:
                model.dispose()
                model.to_empty(device="meta")
        del gemma, connector, gemma_builder, connector_builder
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = json.loads(args.paths.read_text())
    if "ltx_root" in paths:
        root = Path(paths["ltx_root"]).expanduser().resolve(strict=True)
        sys.path[:0] = [str(root / "packages" / package / "src")
                       for package in ("ltx-core", "ltx-pipelines", "ltx-kernels")]
    print(json.dumps(build_cache(paths, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
