#!/usr/bin/env python3
"""Build the fixed 15-video MiniMax H3 -> LTX 2.5 refiner manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence


EXPECTED_SOURCE_COUNT = 15
STUDENT_VARIANT = "t2_l3_480"
PROMPT_ID_RE = re.compile(r"p[0-9]{3}")


def _require_int(value: Any, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{description} must be an integer, got {value!r}")
    return value


def _load_teacher_runs(teacher_benchmark: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(teacher_benchmark.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot read teacher benchmark {teacher_benchmark}: {error}"
        ) from error

    if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
        raise ValueError("teacher benchmark must be a JSON object with a runs array")
    runs = document["runs"]
    if len(runs) != EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"teacher benchmark must contain exactly {EXPECTED_SOURCE_COUNT} runs; "
            f"found {len(runs)}"
        )
    if not all(isinstance(run, dict) for run in runs):
        raise ValueError("every teacher benchmark run must be a JSON object")
    return runs


def _expected_student_name(ordinal: int, prompt_id: str, seed: int) -> str:
    return f"{ordinal:03d}-{prompt_id}-seed{seed}-{STUDENT_VARIANT}.mp4"


def build_manifest(
    teacher_benchmark: Path,
    student_root: Path,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Validate the complete source set and return all or its first ``limit`` rows."""

    teacher_benchmark = teacher_benchmark.resolve()
    student_root = student_root.resolve()
    videos_root = student_root / "videos"
    if not videos_root.is_dir():
        raise ValueError(f"student video directory does not exist: {videos_root}")
    if limit is not None and not 1 <= limit <= EXPECTED_SOURCE_COUNT:
        raise ValueError(
            f"--limit must be in [1, {EXPECTED_SOURCE_COUNT}], got {limit}"
        )

    runs = _load_teacher_runs(teacher_benchmark)
    rows: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    prompt_ids: set[str] = set()
    seeds: set[int] = set()

    for row_index, run in enumerate(runs):
        ordinal = _require_int(run.get("ordinal"), f"runs[{row_index}].ordinal")
        expected_ordinal = row_index + 1
        if ordinal != expected_ordinal:
            raise ValueError(
                f"runs[{row_index}].ordinal must be {expected_ordinal}, got {ordinal}"
            )

        prompt = run.get("prompt")
        if not isinstance(prompt, dict):
            raise ValueError(f"runs[{row_index}].prompt must be an object")
        prompt_id = prompt.get("id")
        if not isinstance(prompt_id, str) or PROMPT_ID_RE.fullmatch(prompt_id) is None:
            raise ValueError(
                f"runs[{row_index}].prompt.id must match pNNN, got {prompt_id!r}"
            )
        if prompt_id in prompt_ids:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        prompt_ids.add(prompt_id)

        prompt_index = _require_int(
            prompt.get("index"), f"runs[{row_index}].prompt.index"
        )
        seed = _require_int(prompt.get("seed"), f"runs[{row_index}].prompt.seed")
        if seed in seeds:
            raise ValueError(f"duplicate seed: {seed}")
        seeds.add(seed)

        text = prompt.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"runs[{row_index}].prompt.text must be non-empty")
        prompt_sha256 = prompt.get("sha256")
        actual_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if prompt_sha256 != actual_sha256:
            raise ValueError(
                f"runs[{row_index}].prompt.sha256 does not match its prompt text"
            )

        metadata = prompt.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(f"runs[{row_index}].prompt.metadata must be an object")
        if metadata.get("id") != prompt_id or metadata.get("index") != prompt_index:
            raise ValueError(
                f"runs[{row_index}] prompt metadata id/index disagrees with prompt"
            )

        student_name = _expected_student_name(ordinal, prompt_id, seed)
        student_path = videos_root / student_name
        if not student_path.is_file():
            raise ValueError(
                f"missing student video for prompt {prompt_id}, seed {seed}: "
                f"{student_path}"
            )
        expected_names.add(student_name)
        rows.append(
            {
                "index": row_index,
                "source_ordinal": ordinal,
                "source_prompt_index": prompt_index,
                "prompt_id": prompt_id,
                "file": student_path.relative_to(student_root).as_posix(),
                "prompt": text,
                "prompt_sha256": actual_sha256,
                "seed": seed,
            }
        )

    actual_names = {path.name for path in videos_root.glob("*.mp4") if path.is_file()}
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise ValueError(
            "student videos do not exactly match the teacher benchmark; "
            f"missing={missing}, unexpected={unexpected}"
        )

    selected = rows if limit is None else rows[:limit]
    # Keep the refiner runner's strict contiguous-index contract explicit.
    return [{**row, "index": index} for index, row in enumerate(selected)]


def _write_json_atomic(output: Path, rows: list[dict[str, Any]]) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(serialized)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.replace(temporary_name, output)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-benchmark", type=Path, required=True)
    parser.add_argument("--student-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="emit only the first N rows after validating the complete 15-video set",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rows = build_manifest(
            args.teacher_benchmark,
            args.student_root,
            limit=args.limit,
        )
        _write_json_atomic(args.output, rows)
    except ValueError as error:
        raise SystemExit(f"error: {error}") from error
    print(f"wrote {len(rows)} validated rows to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
