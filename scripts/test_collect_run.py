#!/usr/bin/env python3
"""Lightweight checks for collect_run quality helper logic."""

from __future__ import annotations

from collect_run import gemini_quality_blocker, max_gemini_severity


def check(name: str, condition: bool) -> None:
    if not condition:
        raise AssertionError(name)
    print(f"PASS {name}")


def main() -> int:
    check(
        "Gemini max severity defaults to none",
        max_gemini_severity({"new_artifacts": []}) == "none",
    )
    check(
        "Gemini fail becomes promotion blocker",
        gemini_quality_blocker(
            {
                "status": "complete",
                "result": {
                    "overall": "fail",
                    "new_artifacts": [{"severity": "high"}],
                },
            }
        )
        == "nvidia_gemini:fail:high",
    )
    check(
        "Medium artifact becomes promotion blocker",
        gemini_quality_blocker(
            {
                "status": "complete",
                "result": {
                    "overall": "pass",
                    "new_artifacts": [{"severity": "medium"}],
                },
            }
        )
        == "nvidia_gemini:artifact:medium",
    )
    check(
        "Clean Gemini pass has no blocker",
        gemini_quality_blocker(
            {
                "status": "complete",
                "result": {"overall": "pass", "new_artifacts": []},
            }
        )
        is None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
