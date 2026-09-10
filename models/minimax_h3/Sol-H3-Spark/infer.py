#!/usr/bin/env python3
"""Generate 768p video with audio using the frozen single-DGX-Spark recipe."""

import argparse
import json
from pathlib import Path
import signal

from runtime.config import load_paths, read_cases
from runtime.pipeline import Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="Runtime path file produced by prepare.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="MiniMax-H3 prompt, including optional soundscape/music fields")
    source.add_argument("--prompts", type=Path, help="JSONL: case_id, prompt, seed; models reused across rows")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --prompt (JSONL has per-row seeds)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    args = parser.parse_args(argv)
    if args.prompt is not None:
        if not args.prompt.strip() or not 0 <= args.seed < 2**63:
            parser.error("Use a nonempty prompt and nonnegative 63-bit seed")
        cases = [{"case_id": "generation", "prompt": args.prompt, "seed": args.seed}]
    else:
        cases = read_cases(args.prompts)
    paths = load_paths(args.paths)

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("Interrupted; closing only this pipeline's workers")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pipeline = Pipeline(paths, args.output_dir)
    try:
        print("Loading models and running one full warmup (reported separately).", flush=True)
        pipeline.start(cases[0])
        for case in cases:
            print(json.dumps(pipeline.generate(case), ensure_ascii=False), flush=True)
        pipeline.finish()
        print(f"Mean E2E: {pipeline.report['mean_e2e_s']:.2f} s; results: {pipeline.root / 'results.json'}")
    except BaseException as error:
        pipeline.report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        pipeline.save()
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
