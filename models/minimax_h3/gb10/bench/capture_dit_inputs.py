#!/usr/bin/env python3
"""Capture real DiT inputs from the official cell, to benchmark one forward against.

Every later measurement — lossless kernel work first, sparse attention after — is a single
`MiniMaxH3Transformer3DModel.forward` on a *recorded* latent rather than on random data.
Random tensors would misstate two things at once: attention sparsity is a property of the
real activations, and any numerical check against the baseline is only meaningful on inputs
the model actually produces.

The geometry is the published cell and is not reduced: 1344x768, 124 frames at 24 fps
(5.167 s), 50 steps, `shift` 12.0 / 3.0 from the released scheduler configs, seed 0, the
official `t2va_example_1` prompt. That packs to 38,247 rows — 37,296 target video plus a
951-row text+audio prefix — which this asserts.

The whole capture is ~10 MB: `hidden_states` is `(1, 37296, 96)` patchified video, and
everything else is either the small audio/text streams or index vectors. Steps are captured
across the trajectory because the noise level changes what sparse attention can skip; the
run aborts after the last one rather than paying for the remaining steps.
"""

from __future__ import annotations

import paths

paths.setup()

import argparse
import inspect
import os
import sys


import torch

import gpu_infer

CAPTURE_DIR = str(paths.CAPTURE_DIR)
EXPECTED_SEQ_LEN = 38247
EXPECTED_VIDEO_TOKENS = 37296


class _CaptureComplete(Exception):
    """Raised to unwind the pipeline once the last requested step has been recorded."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps-to-capture", type=int, nargs="+", default=[0, 12, 24, 36],
                        help="DiT call indices to record (50 scheduler steps = 49 calls)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1344)
    parser.add_argument("--output-dir", default=None,
                        help="defaults to dit_inputs/<width>x<height>")
    args = parser.parse_args()

    wanted = sorted(set(args.steps_to_capture))
    last = wanted[-1]
    official = (args.height, args.width) == (768, 1344)
    if args.output_dir is None:
        args.output_dir = os.path.join(CAPTURE_DIR, f"{args.width}x{args.height}")
    os.makedirs(args.output_dir, exist_ok=True)

    # The published cell, verbatim — `build_pipeline` only reads `device`, the rest drives the
    # request below.
    run_args = args
    # Frames are not a knob: `17 * n + 5` between 5 and 15 seconds at 24 fps makes 124 both the
    # floor and the published value. Only the canvas shrinks, and only for iteration speed —
    # attention is the one O(S^2) term, so a smaller canvas *under*-states its share and any
    # result has to be confirmed at 1344x768.
    run_args.num_frames, run_args.steps = 124, 50
    run_args.seed = 0
    run_args.prompt_file = str(paths.PROMPT_FILE)
    import json
    with open(run_args.prompt_file) as handle:
        run_args.prompt = json.load(handle)["prompt"]

    pipe, placement = gpu_infer.build_pipeline(run_args)
    print(f"[capture] loaded in {placement['load_s']:.1f}s", flush=True)

    transformer = pipe.transformer
    signature = inspect.signature(type(transformer).forward)
    original = transformer.forward
    call_index = 0
    captured: list[int] = []

    def wrapped(*call_args, **call_kwargs):
        nonlocal call_index
        index = call_index
        call_index += 1

        if index in wanted:
            bound = signature.bind(transformer, *call_args, **call_kwargs)
            bound.apply_defaults()
            recorded = {
                name: value.detach().to("cpu").clone()
                for name, value in bound.arguments.items()
                if isinstance(value, torch.Tensor)
            }

            seq_len = recorded["position_ids"].shape[0]
            video_tokens = recorded["hidden_states"].shape[1]
            if index == wanted[0]:
                if official and (seq_len != EXPECTED_SEQ_LEN
                                 or video_tokens != EXPECTED_VIDEO_TOKENS):
                    raise ValueError(
                        f"geometry does not match the published cell: {seq_len} rows / "
                        f"{video_tokens} video tokens, expected {EXPECTED_SEQ_LEN} / "
                        f"{EXPECTED_VIDEO_TOKENS}"
                    )
                print(f"[capture] packed sequence {seq_len} rows, {video_tokens} video tokens "
                      f"(+{seq_len - video_tokens} text/audio prefix)", flush=True)

            path = os.path.join(args.output_dir, f"step_{index:02d}.pt")
            torch.save(
                {
                    "tensors": recorded,
                    "call_index": index,
                    "num_scheduler_steps": 50,
                    "height": run_args.height, "width": run_args.width,
                    "num_frames": 124, "fps": 24,
                    "seed": 0,
                    "prompt_file": run_args.prompt_file,
                    "timestep": recorded["timestep"].tolist(),
                },
                path,
            )
            captured.append(index)
            size_mb = os.path.getsize(path) / 2**20
            print(f"[capture] step {index:02d}: timestep={recorded['timestep'].tolist()} "
                  f"-> {path} ({size_mb:.1f} MB)", flush=True)

        result = original(*call_args, **call_kwargs)
        if index == last:
            raise _CaptureComplete
        return result

    transformer.forward = wrapped

    generator = torch.Generator().manual_seed(run_args.seed)
    try:
        pipe(
            prompt=run_args.prompt,
            height=run_args.height,
            width=run_args.width,
            num_frames=run_args.num_frames,
            num_inference_steps=run_args.steps,
            generator=generator,
        )
    except Exception:
        # The modular pipeline re-raises block errors wrapped, so the sentinel type is not
        # reliably visible here; `captured` is.
        if len(captured) != len(wanted):
            raise
        print(f"[capture] stopped after step {last}; the remaining {49 - last - 1} calls are "
              f"not needed", flush=True)

    print(f"[capture] wrote {sorted(os.listdir(args.output_dir))}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
