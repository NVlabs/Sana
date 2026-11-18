#!/usr/bin/env python3

"""Launch a vLLM server for Qwen3-Omni captioning.
Runs the actual server via ``uvx`` so that vLLM and its CUDA-tied
dependencies live in their own isolated environment (no impact on this
package's dependency tree).
The captioning script (``caption_videos.py``) talks to the server over its
OpenAI-compatible HTTP API. Once the server is up it stays loaded across
captioning runs; no per-script model warmup cost.
Typical usage::
    # Default: dynamic FP8 quantization, listen on 127.0.0.1:8001
    uv run python scripts/serve_captioner.py
    # Just print the chosen `uvx vllm serve ...` command without running it
    uv run python scripts/serve_captioner.py --print-cmd
    # Use full bf16 on a GPU with >= 66 GiB free VRAM (slightly more reliable
    # numerics but 2x the weight memory)
    uv run python scripts/serve_captioner.py --quantization bf16
    # Use a different port or expose on all interfaces
    uv run python scripts/serve_captioner.py --port 9000 --host 0.0.0.0
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

console = Console()

# Model identifier we serve. The captioner client must use the same string.
DEFAULT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

# Pinned vLLM version known to support Qwen3-Omni on CUDA 12.x.
# vLLM 0.20+ requires CUDA 13. Update both as the environment evolves.
# The ``[audio]`` extra is required for Qwen3-Omni to decode audio at all.
DEFAULT_VLLM_SPEC = "vllm[audio]==0.11.2"

# Approximate disk needed for the model download (HF cache structure).
MODEL_DISK_GIB = 65.0


app = typer.Typer(
    pretty_exceptions_enable=False,
    no_args_is_help=False,
    help="Launch a local vLLM server for Qwen3-Omni captioning.",
)


def _query_disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(str(path)).free / 1024**3


def _build_vllm_args(
    *,
    model: str,
    host: str,
    port: int,
    quantization: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    extra_args: list[str],
) -> list[str]:
    """Construct the `vllm serve ...` argv."""
    args = [
        "vllm",
        "serve",
        model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        # Let the server accept ``file://`` URLs pointing at local videos.
        "--allowed-local-media-path",
        "/",
        # The model is a multimodal MoE; cap each input to one of each
        # modality to match what our captioner sends.
        "--limit-mm-per-prompt",
        '{"image": 1, "video": 1, "audio": 1}',
        # Small concurrent-sequence cap so KV cache headroom isn't fragmented.
        "--max-num-seqs",
        "4",
    ]
    if quantization == "fp8":
        args += ["--quantization", "fp8"]
    args += extra_args
    return args


@app.command()
def main(
    model: str = typer.Option(DEFAULT_MODEL, "--model", help="Model identifier to serve."),
    host: str = typer.Option("127.0.0.1", "--host", help="Listen address. Use 0.0.0.0 for remote access."),
    port: int = typer.Option(8001, "--port", help="HTTP port."),
    quantization: str = typer.Option(
        "fp8",
        "--quantization",
        "-q",
        help=(
            "Weight precision. 'fp8' (default, dynamic FP8 -- ~31 GiB weights) is "
            "the recommended choice; it fits on 40 GiB GPUs and runs at the same "
            "speed as bf16 on H100. 'bf16' uses ~60 GiB of weights -- pick it if "
            "you have abundant VRAM and want minimal numerical drift."
        ),
    ),
    max_model_len: int = typer.Option(
        32768,
        "--max-model-len",
        help="Maximum context length the server accepts (must fit input video tokens + max_tokens).",
    ),
    gpu_memory_utilization: float = typer.Option(
        0.9,
        "--gpu-memory-utilization",
        help="Fraction of GPU memory vLLM may reserve (model + KV cache).",
    ),
    hf_home: Path | None = typer.Option(  # noqa: B008
        None,
        "--hf-home",
        help=(
            "Override HF_HOME (where the model is downloaded). The model is ~65 GB; "
            "by default this follows your environment's HF_HOME or HuggingFace's default."
        ),
    ),
    vllm_spec: str = typer.Option(
        DEFAULT_VLLM_SPEC,
        "--vllm-spec",
        help="pip-style spec passed to `uvx --from`. Pin a version that matches your CUDA.",
    ),
    print_cmd: bool = typer.Option(
        False,
        "--print-cmd",
        help="Print the chosen command without running it.",
    ),
    extra_args: list[str] | None = typer.Argument(  # noqa: B008
        None,
        help="Additional args passed through to `vllm serve` after `--`.",
    ),
) -> None:
    """Launch the vLLM server for Qwen3-Omni."""
    extra = extra_args or []

    if quantization not in ("bf16", "fp8"):
        console.print(f"[red]--quantization must be 'bf16' or 'fp8'; got {quantization!r}.[/]")
        raise typer.Exit(code=1)

    # Disk check (only meaningful before first download).
    cache_root = hf_home or Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache_root.mkdir(parents=True, exist_ok=True)
    free_disk = _query_disk_free_gib(cache_root)
    if free_disk < MODEL_DISK_GIB:
        console.print(
            f"[yellow]\u26a0 Only {free_disk:.1f} GiB free on disk under {cache_root} but the "
            f"model needs ~{MODEL_DISK_GIB:.0f} GiB. Either free up space, set --hf-home "
            f"to a larger volume, or expect the download to fail mid-way.[/]"
        )

    vllm_args = _build_vllm_args(
        model=model,
        host=host,
        port=port,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        extra_args=extra,
    )

    # Use ``uvx --from vllm==...`` so vLLM lives in its own throwaway venv
    # (or a cached tool venv). The `--` separates uvx args from the command's.
    uvx_cmd = ["uvx", "--from", vllm_spec, *vllm_args]

    env = os.environ.copy()
    # vLLM 0.11.x requires the V0 engine for Qwen3-Omni's multimodal pipeline.
    env.setdefault("VLLM_USE_V1", "0")
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)

    console.print("\n[bold]Command:[/]")
    console.print("  " + " ".join(uvx_cmd))
    if hf_home is not None:
        console.print(f"  [dim](with HF_HOME={hf_home})[/]")

    if print_cmd:
        return

    console.print("\n[dim]Launching... (first run downloads the model -- ~5 min on a fast link)[/]\n")
    try:
        completed = subprocess.run(uvx_cmd, env=env, check=False)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/]")
        return
    sys.exit(completed.returncode)


if __name__ == "__main__":
    app()
