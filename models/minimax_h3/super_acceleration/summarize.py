#!/usr/bin/env python3
"""Fail-closed summary for the two independent pipeline pairs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--hot-repeats", type=int, choices=(1, 10), required=True)
    args = parser.parse_args()
    rows: list[dict[str, Any]] = []
    sources = []
    handoff_modes: set[str] = set()
    for pair in range(2):
        path = args.run_root / f"pair_{pair}" / "benchmark.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or int(payload["hot_repeats"]) != args.hot_repeats:
            raise RuntimeError(f"incomplete pair benchmark {path}")
        if int(payload["pair_id"]) != pair or len(payload["hot"]) != args.hot_repeats:
            raise RuntimeError(f"pair identity/count mismatch in {path}")
        sources.append(str(path))
        handoff_modes.add(str(payload.get("handoff_mode", "mp4")))
        rows.extend(payload["hot"])
    if len(handoff_modes) != 1:
        raise RuntimeError(f"pair handoff modes disagree: {sorted(handoff_modes)}")
    handoff_mode = next(iter(handoff_modes))
    for row in rows:
        stage2 = row["stage2"]
        attention = stage2["attention"]
        kernel = attention.get("kernel") or {}
        if (
            int(attention.get("dense_calls", -1)) != 3
            or int(attention.get("sol_calls", -1)) != 141
            or int(kernel.get("kernel_calls", -1)) != 141
            or attention.get("selected_backend") != "cute_sm100"
        ):
            raise RuntimeError(f"invalid strict SOL telemetry: {attention}")
        output = Path(row["stage2_output"])
        if not output.is_file() or output.stat().st_size <= 0:
            raise FileNotFoundError(output)

    def summary(key: str) -> dict[str, Any]:
        values = [float(row[key]) for row in rows]
        return {
            "count": len(values),
            "median_s": statistics.median(values),
            "mean_s": statistics.fmean(values),
            "min_s": min(values),
            "max_s": max(values),
            "values_s": values,
        }

    stage2_phase_names = tuple(rows[0]["stage2"]["phases_s"])
    if any(tuple(row["stage2"]["phases_s"]) != stage2_phase_names for row in rows):
        raise RuntimeError("Stage-2 phase schema changed across hot requests")
    stage2_phases_median = {
        name: statistics.median(
            float(row["stage2"]["phases_s"][name]) for row in rows
        )
        for name in stage2_phase_names
    }

    stage1_delivery = {
        "h3_denoise": statistics.median(
            float(row["stage1"]["h3_denoise_s"]) for row in rows
        ),
        "h3_dit_and_shared": statistics.median(
            float(row["stage1"]["h3_dit_and_shared_s"]) for row in rows
        ),
        "taeh3_decode": statistics.median(
            float(row["stage1"]["stage1_decode_s"]) for row in rows
        ),
        "h3_audio_decode": statistics.median(
            float(row["stage1"]["stage1_audio_decode_s"]) for row in rows
        ),
    }
    if handoff_mode == "direct_tensor":
        stage1_delivery["tensor_preprocess_and_handoff"] = statistics.median(
            float(row["stage1"]["stage1_tensor_handoff_s"]) for row in rows
        )
    else:
        stage1_delivery["h264_aac_mux"] = statistics.median(
            float(row["stage1"]["stage1_encode_mux_s"]) for row in rows
        )

    result = {
        "schema_version": 1,
        "status": "complete",
        "kind": f"two_pair_single_gpu_per_stage_h3_ltx25_e2e_{handoff_mode}",
        "topology": "two independent pairs; each pair is one H3 GPU -> one LTX GPU",
        "handoff_mode": handoff_mode,
        "hot_samples": len(rows),
        "e2e_wall": summary("e2e_wall_s"),
        "stage1_wall": {
            "median_s": statistics.median(float(row["stage1"]["stage1_wall_s"]) for row in rows)
        },
        "stage1_gpu_and_delivery_median_s": stage1_delivery,
        "stage2_service": {
            "median_s": statistics.median(float(row["stage2"]["wall_s"]) for row in rows)
        },
        "control_rpc_residual": {
            "median_s": statistics.median(
                float(row["control_rpc_residual_s"]) for row in rows
            ),
            "definition": (
                "E2E minus Stage-1 wall minus resident Stage-2 service wall; "
                "localhost JSON request/response and serialization only"
            ),
        },
        "stage2_phases_median_s": stage2_phases_median,
        "outputs": [row["stage2_output"] for row in rows],
        "pair_benchmarks": sources,
    }
    destination = args.run_root / "summary.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(destination)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
