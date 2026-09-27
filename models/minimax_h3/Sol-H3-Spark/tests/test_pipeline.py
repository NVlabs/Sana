"""CPU-only tests of orchestration, output ownership and timing boundaries."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

from runtime.config import read_cases, normalize_case, load_recipe, load_paths, REQUIRED_PATHS
from runtime.pipeline import Pipeline, await_json, worker_environment


class FakeWorker:
    events = []

    def __init__(self, name, root, paths, config):
        self.name = name
        self.config = config
        self.events.append(("start", name))
        self.pending = None

    def submit(self, op="run", **kwargs):
        self.pending = (op, kwargs)
        self.events.append(("submit", self.name, op))
        return "pending"

    def wait(self, _response):
        op, kwargs = self.pending
        self.pending = None
        return self.call(op, **kwargs)

    def call(self, op="run", **kwargs):
        self.events.append(("call", self.name, op))
        if op == "release_idle_cache":
            return {}
        if op == "prepare":
            Path(kwargs["output_root"]).mkdir()
            return {}
        if self.name == "qwen":
            return {"conditioning_path": "unused", "request_wall_s": 0.01}
        if self.name == "stage1":
            Path(kwargs["outputdir"]).mkdir()
            case = kwargs["case"]
            (Path(kwargs["outputdir"]) / "request.json").write_text(json.dumps({
                "request_id": case.get("request_id", case["case_id"])}))
            return {"capture_dir": kwargs["outputdir"], "request_wall_s": 0.02}
        captured = json.loads((Path(kwargs["capture_dir"]) / "request.json").read_text())
        if captured["request_id"] != kwargs["request_id"]:
            raise ValueError("capture belongs to another request")
        output = Path(kwargs["output_root"]) / "video.mp4"
        output.write_bytes(b"test output, not a real MP4")
        return {"output": str(output), "final_mp4_complete_monotonic_ns": time.monotonic_ns(),
                "stage2_request_s": 0.03, "result": {"phases_s": {"refine": 0.02}}}

    def close(self):
        self.events.append(("close", self.name))


class PipelineTests(unittest.TestCase):
    def test_residency_order_and_real_endpoint(self):
        FakeWorker.events = []
        case = {"case_id": "example", "prompt": "A bird glides over a mountain lake.", "seed": 42}
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline({}, Path(directory) / "run", worker_factory=FakeWorker)
            try:
                pipeline.start(case)
                row = pipeline.generate(case)
                pipeline.finish()
                self.assertEqual(pipeline.report["status"], "PASS")
                self.assertEqual(len(pipeline.report["requests"]), 1)
                self.assertFalse(pipeline.report["phase_sum_used"])
                self.assertEqual(row["e2e_s"],
                    (row["final_mp4_complete_monotonic_ns"] - row["request_start_monotonic_ns"]) / 1e9)
                events = FakeWorker.events
                self.assertLess(events.index(("close", "qwen")), events.index(("start", "stage2")))
                starts = [index for index, event in enumerate(events) if event == ("start", "qwen")]
                self.assertEqual(len(starts), 2)
                self.assertGreater(starts[1], events.index(("call", "stage2", "run")))
                self.assertEqual(events.count(("start", "stage1")), 1)
                self.assertEqual(events.count(("start", "stage2")), 1)
            finally:
                pipeline.close()

    def test_does_not_overwrite_run(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                Pipeline({}, directory, worker_factory=FakeWorker)

    def test_case_ids_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.jsonl"
            for name in ("../escape", "warmup", "stage1-worker"):
                path.write_text(json.dumps({"case_id": name, "prompt": "test"}) + "\n")
                with self.assertRaises(ValueError):
                    read_cases(path)
            valid = {"case_id": "bird", "prompt": "test"}
            path.write_text(json.dumps(valid) + "\n" + json.dumps(valid) + "\n")
            with self.assertRaises(ValueError):
                read_cases(path)

    def test_recipe_has_no_experimental_arm(self):
        recipe = load_recipe()
        self.assertEqual(recipe["stage1"]["height"], 384)
        self.assertEqual(recipe["handoff"]["upscale_ratio"], 2)
        self.assertEqual(recipe["stage2"]["updates"], 3)
        self.assertFalse(recipe["stage2"]["online_text_encoder"])
        self.assertEqual(recipe["output"]["frames"], 121)

    def test_multimodal_case_inputs_are_preserved_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename in ("first.png", "last.png", "voice.wav"):
                (root / filename).write_bytes(b"input fixture")
            case = {"case_id": "guided", "prompt": "A fox turns its head.", "task": "fl2va",
                    "first_frame": "first.png", "last_frame": "last.png"}
            normalized = normalize_case(case, base_dir=root)
            self.assertEqual(normalized["first_frame"], str((root / "first.png").resolve()))
            self.assertEqual(normalized["last_frame"], str((root / "last.png").resolve()))
            references = [{"type": "audio", "path": "voice.wav"}, {"type": "image", "path": "first.png"}]
            ref = normalize_case({"case_id": "ref", "prompt": "A fox.", "task": "ref2va",
                                  "references": references}, base_dir=root)
            self.assertEqual([item["type"] for item in ref["references"]], ["audio", "image"])
            with self.assertRaisesRegex(ValueError, "audio alone"):
                normalize_case(dict(ref, references=ref["references"][:1]), base_dir=root)
            with self.assertRaisesRegex(ValueError, "T2VA"):
                normalize_case(dict(case, task="t2va"), base_dir=root)

    def test_batch_rejects_mixed_model_families_before_workers_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.png").write_bytes(b"input fixture")
            path = root / "prompts.jsonl"
            first = {"case_id": "text", "prompt": "A fox."}
            second = {"case_id": "guided", "prompt": "A fox.", "task": "fl2va", "first_frame": "first.png"}
            path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
            with self.assertRaisesRegex(ValueError, "one task"):
                read_cases(path)

    def test_ref2va_uses_its_own_adapter_without_inventing_vsa_gates(self):
        text, ref = load_recipe(), load_recipe("ref2va")
        self.assertEqual(text["stage1"]["lora"], "FastH3_VSA_DataFree")
        self.assertEqual(ref["stage1"]["lora"], "LightX2V_Ref2VA_4step")
        self.assertEqual(ref["stage1"]["attention"], "FA4_dense")
        self.assertNotIn("attention_policy", ref["stage1"])
        self.assertEqual(ref["stage1"]["reference_image_resize"], {
            "mode": "match", "pixel_budget": 672 * 384,
            "budget_source": "stage1_draft",
        })
        self.assertNotIn("sparsity", ref["stage1"])
        self.assertEqual(ref["stage2"], text["stage2"])
        self.assertEqual(ref["handoff"], text["handoff"])

    def test_ref2va_matching_and_attention_choices_leave_other_settings_unchanged(self):
        baseline = load_recipe("ref2va")
        for match in ("stage1", "stage2"):
            for attention in ("dense", "sol"):
                with self.subTest(match=match, attention=attention):
                    recipe = load_recipe("ref2va", ref_image_match=match, ref_stage1_attn=attention)
                    source = recipe["stage1"] if match == "stage1" else recipe["output"]
                    self.assertEqual(recipe["stage1"]["reference_image_resize"], {
                        "mode": "match", "pixel_budget": source["width"] * source["height"],
                        "budget_source": "stage1_draft" if match == "stage1" else "final_output",
                    })
                    self.assertEqual(recipe["stage1"]["attention"],
                                     "FA4_dense" if attention == "dense" else "FA4_Sol_text_audio_sink")
                    if attention == "sol":
                        self.assertEqual(recipe["stage1"]["attention_policy"], {
                            "dense_steps": 1, "dense_body_layers": [0],
                            "tau_by_step": [None, 1.0, 1.25, 1.5],
                            "sink_token_tags": [1, 2], "dense_sink_queries": True,
                            "reference_image_sink": False, "qwen_visual_sink": False,
                            "sink_kv_block_size": 64,
                        })
                    for key in ("attention", "attention_policy", "reference_image_resize"):
                        recipe["stage1"].pop(key, None)
                    expected = json.loads(json.dumps(baseline))
                    for key in ("attention", "attention_policy", "reference_image_resize"):
                        expected["stage1"].pop(key, None)
                    self.assertEqual(recipe, expected)
        for task in ("t2va", "fl2va"):
            for options in ({"ref_image_match": "stage1"}, {"ref_stage1_attn": "dense"}):
                with self.assertRaisesRegex(ValueError, "require task ref2va"):
                    load_recipe(task, **options)
        for options in ({"ref_image_match": "native"}, {"ref_stage1_attn": "vsa"},
                        {"ref_image_match": ""}, {"ref_stage1_attn": ""}):
            with self.assertRaises(ValueError):
                load_recipe("ref2va", **options)

    def test_ref2va_options_reach_all_workers_and_result_recipe(self):
        case = {"case_id": "reference", "task": "ref2va", "prompt": "A fox.", "seed": 42}
        with tempfile.TemporaryDirectory() as directory:
            pipeline = Pipeline({}, Path(directory) / "run", task="ref2va",
                                ref_image_match="stage2", ref_stage1_attn="sol", worker_factory=FakeWorker)
            try:
                pipeline.start(case)
                for worker in (pipeline.stage1, pipeline.stage2, pipeline.qwen):
                    self.assertEqual(worker.config, pipeline.config)
                    self.assertEqual(worker.config["stage1"]["reference_image_resize"]["pixel_budget"], 1344 * 768)
                    self.assertEqual(worker.config["stage1"]["attention"], "FA4_Sol_text_audio_sink")
                saved = json.loads((pipeline.root / "results.json").read_text())
                self.assertEqual(saved["recipe"], pipeline.config)
            finally:
                pipeline.close()

    def test_cli_ref2va_options_single_and_inferred_batch_task(self):
        import infer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "subject.png"
            image.write_bytes(b"input fixture")
            batch = root / "cases.jsonl"
            batch.write_text(json.dumps({"case_id": "reference", "task": "ref2va", "prompt": "A fox.",
                                         "references": [{"type": "image", "path": str(image)}]}) + "\n")
            for source in (["--task", "ref2va", "--prompt", "A fox.", "--reference", f"image:{image}"],
                           ["--prompts", str(batch)]):
                with mock.patch.object(infer, "load_paths", return_value={}), \
                     mock.patch.object(infer, "Pipeline") as factory, \
                     mock.patch.object(infer.signal, "signal"), contextlib.redirect_stdout(io.StringIO()):
                    factory.return_value.report = {"mean_e2e_s": 1.0}
                    factory.return_value.generate.return_value = {"status": "PASS"}
                    factory.return_value.root = root / "run"
                    infer.main(["--paths", "unused.json", "--output-dir", str(root / "run"), *source,
                                "--ref-image-match", "stage2", "--ref-stage1-attn", "sol"])
                    factory.assert_called_once_with({}, root / "run", task="ref2va",
                                                    ref_image_match="stage2", ref_stage1_attn="sol")
            for flag, value in (("--ref-image-match", "stage1"), ("--ref-stage1-attn", "dense")):
                with mock.patch.object(infer, "load_paths") as paths, \
                     mock.patch.object(infer, "Pipeline") as factory, contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        infer.main(["--paths", "unused.json", "--output-dir", str(root / "bad"),
                                    "--prompt", "A fox.", flag, value])
                    self.assertEqual(error.exception.code, 2)
                    paths.assert_not_called()
                    factory.assert_not_called()

    def test_paths_preserve_venv_interpreter_and_allow_absent_offline_weights(self):
        import sys
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            interpreter = root / "venv-python"
            interpreter.symlink_to(sys.executable)
            values = {key: str(root) for key in REQUIRED_PATHS}
            values.update({key: str(interpreter) for key in REQUIRED_PATHS if key.endswith("_python")})
            values["offline_gemma"] = str(root / "not-downloaded.safetensors")
            filename = root / "paths.json"
            filename.write_text(json.dumps(values))
            paths = load_paths(filename)
            self.assertEqual(paths["stage1_python"], str(interpreter))
            self.assertEqual(paths["offline_gemma"], str(Path(values["offline_gemma"]).resolve()))

    def test_worker_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            path.write_text(json.dumps({"status": "FAIL", "error": "expected failure"}))
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                await_json(path, None)

    def test_stage1_environment_keeps_fa4_isolated(self):
        paths = {"fa4_root": "/deps/fa4", "fastvideo_root": "/deps/FastVideo"}
        env = worker_environment("stage1", paths)
        self.assertIn("/deps/fa4/nvidia_cutlass_dsl/python_packages", env["PYTHONPATH"])
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertNotIn("HF_TOKEN", env)


if __name__ == "__main__":
    unittest.main()
