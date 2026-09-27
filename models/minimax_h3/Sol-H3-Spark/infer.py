#!/usr/bin/env python3
"""Generate 768p video with audio using the frozen single-DGX-Spark recipe."""

import argparse
import json
from pathlib import Path
import signal

from runtime.config import (TASKS, REF_IMAGE_MATCH_CHOICES, REF_STAGE1_ATTN_CHOICES,
                            load_paths, normalize_case, read_cases)
from runtime.pipeline import Pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=Path, required=True, help="Runtime path file produced by prepare.py")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="MiniMax-H3 prompt, including optional soundscape/music fields")
    source.add_argument("--prompts", type=Path, help="JSONL: case_id, prompt, seed; models reused across rows")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --prompt (JSONL has per-row seeds)")
    parser.add_argument("--task", choices=TASKS, default=None, help="Task family; defaults to t2va")
    parser.add_argument("--first-frame", type=Path, help="FL2VA first-frame image for --prompt")
    parser.add_argument("--last-frame", type=Path, help="FL2VA last-frame image for --prompt")
    parser.add_argument("--reference", action="append", default=[], metavar="TYPE:PATH",
                        help="Ordered Ref2VA image:PATH, video:PATH or audio:PATH; repeat as needed")
    parser.add_argument("--ref-image-match", choices=REF_IMAGE_MATCH_CHOICES,
                        help="Ref2VA only: match each image to the stage1 (default) or stage2 pixel area")
    parser.add_argument("--ref-stage1-attn", choices=REF_STAGE1_ATTN_CHOICES,
                        help="Ref2VA only: dense FA4 (default) or dense-first Sol-Attn; Stage2 is unchanged")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing runs are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.prompt is not None:
            case = {"case_id": "generation", "prompt": args.prompt, "seed": args.seed,
                    "task": args.task or "t2va"}
            for key in ("first_frame", "last_frame"):
                if getattr(args, key) is not None:
                    case[key] = str(getattr(args, key))
            if args.reference:
                case["references"] = []
                for reference in args.reference:
                    kind, separator, path = reference.partition(":")
                    if not separator:
                        raise ValueError("--reference must be image:PATH, video:PATH or audio:PATH")
                    case["references"].append({"type": kind, "path": path})
            cases = [normalize_case(case)]
        else:
            if args.first_frame or args.last_frame or args.reference:
                raise ValueError("With --prompts, put frame/reference inputs in each JSONL row")
            cases = read_cases(args.prompts, default_task=args.task or "t2va")
            if args.task and cases[0]["task"] != args.task:
                raise ValueError("--task disagrees with the JSONL task")
        if cases[0]["task"] != "ref2va" and (args.ref_image_match is not None or args.ref_stage1_attn is not None):
            raise ValueError("--ref-image-match and --ref-stage1-attn require task ref2va")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    paths = load_paths(args.paths, task=cases[0]["task"])

    def interrupted(_number, _frame):
        raise KeyboardInterrupt("Interrupted; closing only this pipeline's workers")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    pipeline = Pipeline(paths, args.output_dir, task=cases[0]["task"],
                        ref_image_match=args.ref_image_match, ref_stage1_attn=args.ref_stage1_attn)
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
