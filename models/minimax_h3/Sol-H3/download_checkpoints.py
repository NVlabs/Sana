#!/usr/bin/env python3
"""Download MiniMax-H3 and the selected four-step adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument(
        "--task",
        choices=("t2v", "ref2va", "both"),
        default="t2v",
        help="Checkpoint partition to download (default: t2v)",
    )
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    model_patterns = [
        "model_index.json",
        "modular_model_index.json",
        "audio_scheduler/*",
        "audio_vae/*",
        "processor/*",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "vae/*",
    ]
    if args.task in {"t2v", "both"}:
        model_patterns.append("transformer/*")
    if args.task in {"ref2va", "both"}:
        model_patterns.append("transformer_ref/*")

    snapshot_download(
        "MiniMaxAI/MiniMax-H3",
        local_dir=root / "MiniMax-H3",
        allow_patterns=model_patterns,
    )
    if args.task in {"t2v", "both"}:
        snapshot_download(
            "FastVideo/FastH3-4-step-Preview-v1-LoRA",
            local_dir=root / "FastH3-4-step-Preview-v1-LoRA",
            allow_patterns=["dense-datafree/adapter_model.safetensors"],
        )
    if args.task in {"ref2va", "both"}:
        snapshot_download(
            "lightx2v/Minimax-h3-Turbo",
            local_dir=root / "Minimax-h3-Turbo",
            allow_patterns=["minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"],
        )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
