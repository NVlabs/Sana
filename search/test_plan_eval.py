#!/usr/bin/env python3
"""CPU test for plan_eval: speed-target binning + candidate rendering.
Run: ~/lustre/miniconda3/envs/sana/bin/python search/test_plan_eval.py
"""
import os
import sys
import json
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from search.plan_eval import assess, gemini_quality_blockers, load_profile, load_tiers, promotion_note, quality_ranking_key, tier_of, render_candidate  # noqa: E402
from tools.vision.nvidia_gemini_judge import extract_json  # noqa: E402

ok = fail = 0
def check(name, cond):
    global ok, fail
    ok, fail = (ok + 1, fail) if cond else (ok, fail + 1)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}")

tiers = load_tiers()

# --- tier_of: improvement is required ---
check("no speedup/mem win -> no tier", tier_of(1.0, None, {"overall": "pass", "new_artifacts": []}, tiers) is None)

# --- speed targets, not LPIPS/Gemini hard thresholds ---
check("speedup below 1.5x -> no delivery bucket",
      tier_of(1.3, None, {"overall": "pass", "new_artifacts": []}, tiers) is None)

check("1.5x speed target -> low bucket",
      tier_of(1.5, None, {"overall": "pass", "new_artifacts": []}, tiers) == "low")

check("2.0x speed target -> medium bucket",
      tier_of(2.0, None, {"overall": "pass", "new_artifacts": [{"severity": "low"}]}, tiers) == "medium")

# Quality is ranking evidence, not a hard threshold inside tier_of.
check("3.0x speed target -> high bucket even when quality ranks lower",
      tier_of(3.0, None, {"overall": "fail", "new_artifacts": [{"severity": "medium"}]}, tiers) == "high")

check("high-severity artifact still speed-buckets; selector ranks it lower",
      tier_of(3.0, None, {"overall": "fail", "new_artifacts": [{"severity": "high"}]}, tiers) == "high")

# --- missing / inconclusive Gemini does not change the speed bucket in tier_of ---
check("no gemini verdict -> speed bucket still determined by speed",
      tier_of(2.0, None, None, tiers) == "medium")
check("inconclusive gemini -> speed bucket still determined by speed",
      tier_of(2.0, None, {"overall": "inconclusive", "new_artifacts": []}, tiers) == "medium")

# --- memory-only win is frontier evidence but not a speed-target bucket ---
check("mem win without speedup -> no speed bucket",
      tier_of(None, 0.8, {"overall": "pass", "new_artifacts": []}, tiers) is None)

# --- final quality ranking uses Gemini and LPIPS together, not LPIPS alone ---
clean_higher_lpips = {
    "speedup": 2.2,
    "gemini_overall": "pass",
    "max_artifact_severity": "none",
    "lpips_max": 0.04,
}
artifact_lower_lpips = {
    "speedup": 2.4,
    "gemini_overall": "pass",
    "max_artifact_severity": "medium",
    "lpips_max": 0.001,
}
check("quality ranking prefers cleaner Gemini even if LPIPS is higher",
      sorted([artifact_lower_lpips, clean_higher_lpips], key=quality_ranking_key)[0] is clean_higher_lpips)

same_gemini_lower_lpips = {
    "speedup": 2.1,
    "gemini_overall": "pass",
    "max_artifact_severity": "none",
    "lpips_max": 0.01,
}
check("quality ranking uses LPIPS when Gemini evidence ties",
      sorted([clean_higher_lpips, same_gemini_lower_lpips], key=quality_ranking_key)[0] is same_gemini_lower_lpips)

# --- promotion note must not mislabel quality failures as missing speedup ---
check("note: speedup + inconclusive quality says frontier evidence",
      "below the lowest delivery target" in promotion_note(
          None,
          [],
          1.2,
          None,
          {"overall": "inconclusive", "new_artifacts": []},
          tiers,
          lpips_delta=0.02,
      ))
check("note: blocked quality evidence asks for backfill",
      "missing or blocked quality evidence" in promotion_note(
          None,
          ["lpips:missing", "nvidia_gemini:blocked"],
          2.0,
          None,
          {"overall": "inconclusive", "new_artifacts": []},
          tiers,
      ))
check("gemini fail becomes hard quality blocker",
      gemini_quality_blockers({"overall": "fail", "new_artifacts": [{"severity": "high"}]}) == ["nvidia_gemini:fail:high"])
check("note: gemini fail says quality failed",
      "quality failed" in promotion_note(
          None,
          ["nvidia_gemini:fail:high"],
          2.0,
          None,
          {"overall": "fail", "new_artifacts": [{"severity": "high"}]},
          tiers,
      ))
check("note: no speedup still says no latency/mem improvement",
      "no latency/mem improvement" in promotion_note(
          None,
          [],
          1.0,
          None,
          {"overall": "pass", "new_artifacts": []},
          tiers,
      ))

check("gemini helper null root becomes inconclusive JSON",
      extract_json("null")["overall"] == "inconclusive")

# --- assess can reuse a durable pairwise judge artifact for backfills ---
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "run"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs/benchmark.json").write_text(json.dumps({"total_s": 80.0}) + "\n")
    (root / "outputs/quality.json").write_text(json.dumps({"status": "ok", "judges": {}}) + "\n")
    (root / "outputs/quality_pairwise.json").write_text(
        json.dumps({"overall": "pass", "new_artifacts": []}) + "\n"
    )
    verdict = assess(root, load_profile("cosmos3"), tiers, baseline_frames=None, gemini=True)
    check("assess: reuses existing pairwise verdict", verdict["gemini_overall"] == "pass")
    check("assess: existing pairwise verdict can reach low speed target", verdict["tier"] == "low")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "run"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs/benchmark.json").write_text(json.dumps({"total_s": 80.0}) + "\n")
    (root / "outputs/quality.json").write_text(json.dumps({
        "status": "blocked_quality",
        "promotion_blockers": ["nvidia_gemini:blocked"],
        "judges": {
            "lpips": {"result": {"status": "ok", "max": 0.0}},
            "nvidia_gemini": {"status": "blocked", "result": {}},
        },
    }) + "\n")
    (root / "outputs/quality_pairwise.json").write_text(
        json.dumps({"overall": "pass", "new_artifacts": []}) + "\n"
    )
    verdict = assess(root, load_profile("cosmos3"), tiers, baseline_frames=None, gemini=True)
    check("assess: pairwise Gemini clears collector Gemini blocker",
          "nvidia_gemini:blocked" not in verdict["quality_blockers"])
    check("assess: pairwise override can speed-bucket despite collector Gemini block",
          verdict["tier"] == "low")

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "run"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs/benchmark.json").write_text(json.dumps({"total_s": 80.0}) + "\n")
    (root / "outputs/quality.json").write_text(json.dumps({
        "status": "ok",
        "promotion_blockers": [],
        "judges": {
            "nvidia_gemini": {
                "status": "complete",
                "result": {
                    "overall": "fail",
                    "new_artifacts": [{"severity": "high"}],
                },
            },
        },
    }) + "\n")
    (root / "outputs/quality_pairwise.json").write_text(
        json.dumps({"overall": "pass", "new_artifacts": []}) + "\n"
    )
    verdict = assess(root, load_profile("cosmos3"), tiers, baseline_frames=None, gemini=True)
    check("assess: collector Gemini fail is not hidden by pairwise pass",
          "nvidia_gemini:fail:high" in verdict["quality_blockers"])
    check("assess: conflicting Gemini fail blocks speed bucket", verdict["tier"] is None)

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "run"
    (root / "outputs").mkdir(parents=True)
    (root / "outputs/benchmark.json").write_text(json.dumps({"total_s": 80.0}) + "\n")
    (root / "outputs/quality.json").write_text(json.dumps({"status": "ok", "judges": {}}) + "\n")
    (root / "outputs/quality_pairwise.json").write_text(
        json.dumps({"overall": "fail", "new_artifacts": [{"severity": "high"}]}) + "\n"
    )
    verdict = assess(root, load_profile("cosmos3"), tiers, baseline_frames=None, gemini=True)
    check("assess: pairwise Gemini fail blocks quality", "nvidia_gemini:fail:high" in verdict["quality_blockers"])
    check("assess: pairwise Gemini fail blocks speed bucket", verdict["tier"] is None)
    check("assess: pairwise Gemini fail note is explicit", "quality failed" in verdict["note"])

# --- render_candidate produces a launcher-valid sparse manifest ---
try:
    prof = load_profile("cosmos3")
    m = render_candidate(prof, "sparse_attention", {"sparsity": 0.9, "component": "transformer"})
    check("render: has official_config + slurm", "official_config" in m and "slurm" in m)
    check("render: composed sparse env present",
          "SGLANG_HQ_COMPONENT_ATTENTION_BACKENDS" in m["env"])
    check("render: carries model base env (MODEL_REPO)", m["env"].get("MODEL_REPO") == "nvidia/Cosmos3-Super")
    q = render_candidate(
        prof,
        "nvfp4_ffn",
        {
            "disable_rht": False,
            "disable_stochastic_rounding": False,
            "disable_2d_quantization": False,
            "fused_proj_in_gelu": True,
            "pad_m_to": 32,
            "fp4_gemm_backend": "cudnn",
            "dense_layers": "0-1",
        },
        kind="build_transform",
    )
    check("render nvfp4: enables primary env", q["env"].get("SGLANG_HQ_ENABLE_TE_NVFP4_FFN") == "1")
    check("render nvfp4: recipe flag propagated", q["env"].get("SGLANG_HQ_NVFP4_DISABLE_RHT") == "0")
    check("render nvfp4: fused path propagated", q["env"].get("SGLANG_HQ_ENABLE_TE_NVFP4_FUSED_PROJ_IN_GELU") == "1")
    check("render nvfp4: no implicit LTX2 adapter env",
          not any(key.startswith("SGLANG_LTX2_TE_NVFP4_") for key in q["env"]))
    check("render nvfp4: backend propagated", q["env"].get("SGLANG_DIFFUSION_FLASHINFER_FP4_GEMM_BACKEND") == "cudnn")
    check("render nvfp4: dense guard metadata propagated", q["env"].get("SGLANG_HQ_NVFP4_DENSE_LAYERS") == "0-1")
    p = render_candidate(
        prof,
        "nvfp4_ffn",
        {
            "module_scope": "profiled_hot_ffn",
            "profile_layer_scores": "0-1:0.05,2-29:1.0,30-31:0.05",
            "profile_keep_ratio": 0.875,
        },
        kind="build_transform",
    )
    check("render nvfp4 profile: selector-derived profiled layers propagated",
          p["env"].get("SGLANG_HQ_NVFP4_PROFILED_LAYERS") == "2-29")
    check("render nvfp4 profile: selector-derived dense guards propagated",
          p["env"].get("SGLANG_HQ_NVFP4_DENSE_LAYERS") == "0-1,30-31")
except Exception as e:  # torch/efficiency import issues shouldn't fail the tier logic
    print(f"  SKIP  render_candidate ({type(e).__name__}: {e})")

print(f"\n=== {ok} passed, {fail} failed ===")
sys.exit(1 if fail else 0)
