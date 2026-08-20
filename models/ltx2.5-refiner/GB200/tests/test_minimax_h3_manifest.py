#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_minimax_h3_manifest import (
    EXPECTED_SOURCE_COUNT,
    build_manifest,
    main,
)


class MiniMaxH3ManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.student_root = self.root / "student"
        (self.student_root / "videos").mkdir(parents=True)
        runs = []
        for index in range(EXPECTED_SOURCE_COUNT):
            ordinal = index + 1
            prompt_id = f"p{index * 3 + 1:03d}"
            prompt_index = index * 3
            seed = 42 + index
            text = f"Detailed prompt number {ordinal}."
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            runs.append(
                {
                    "ordinal": ordinal,
                    "prompt": {
                        "id": prompt_id,
                        "index": prompt_index,
                        "metadata": {"id": prompt_id, "index": prompt_index},
                        "seed": seed,
                        "sha256": digest,
                        "text": text,
                    },
                }
            )
            filename = (
                f"{ordinal:03d}-{prompt_id}-seed{seed}-t2_l3_480.mp4"
            )
            (self.student_root / "videos" / filename).touch()
        self.teacher_benchmark = self.root / "benchmark.json"
        self.teacher_benchmark.write_text(
            json.dumps({"runs": runs}), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_builds_complete_refiner_manifest(self) -> None:
        rows = build_manifest(self.teacher_benchmark, self.student_root)
        self.assertEqual(len(rows), EXPECTED_SOURCE_COUNT)
        self.assertEqual([row["index"] for row in rows], list(range(15)))
        self.assertEqual(rows[0]["prompt_id"], "p001")
        self.assertEqual(rows[0]["seed"], 42)
        self.assertEqual(
            rows[0]["file"], "videos/001-p001-seed42-t2_l3_480.mp4"
        )
        self.assertEqual(rows[-1]["source_ordinal"], 15)
        self.assertEqual(rows[-1]["source_prompt_index"], 42)

    def test_limit_validates_all_sources_but_emits_smoke_prefix(self) -> None:
        missing = self.student_root / "videos" / (
            "015-p043-seed56-t2_l3_480.mp4"
        )
        missing.unlink()
        with self.assertRaisesRegex(ValueError, "missing student video"):
            build_manifest(self.teacher_benchmark, self.student_root, limit=1)

        missing.touch()
        rows = build_manifest(self.teacher_benchmark, self.student_root, limit=1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["index"], 0)

    def test_rejects_prompt_seed_filename_disagreement(self) -> None:
        document = json.loads(self.teacher_benchmark.read_text(encoding="utf-8"))
        document["runs"][0]["prompt"]["seed"] = 999
        self.teacher_benchmark.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing student video.*seed 999"):
            build_manifest(self.teacher_benchmark, self.student_root)

    def test_rejects_unexpected_student_video(self) -> None:
        (self.student_root / "videos" / "unexpected.mp4").touch()
        with self.assertRaisesRegex(ValueError, "unexpected=.*unexpected.mp4"):
            build_manifest(self.teacher_benchmark, self.student_root)

    def test_cli_writes_requested_smoke_manifest(self) -> None:
        output = self.root / "manifests" / "smoke.json"
        self.assertEqual(
            main(
                [
                    "--teacher-benchmark",
                    str(self.teacher_benchmark),
                    "--student-root",
                    str(self.student_root),
                    "--output",
                    str(output),
                    "--limit",
                    "1",
                ]
            ),
            0,
        )
        self.assertEqual(len(json.loads(output.read_text(encoding="utf-8"))), 1)


if __name__ == "__main__":
    unittest.main()
