"""CPU-only task-specific download and component preparation contracts."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from download_checkpoints import MANIFEST, checkpoint_paths, selected_entries
from prepare import missing_h3_components


class CheckpointTaskTests(unittest.TestCase):
    def test_task_partitions_and_adapters(self):
        manifest = json.loads(MANIFEST.read_text())
        for task in ("t2va", "fl2va", "ref2va"):
            entries = {entry["key"]: entry for entry in selected_entries(manifest, task=task)}
            patterns = entries["h3_model"]["allow_patterns"]
            self.assertEqual("transformer/*" in patterns, task != "ref2va")
            self.assertEqual("transformer_ref/*" in patterns, task == "ref2va")
            self.assertEqual("vae/*" in patterns, task != "t2va")
            self.assertEqual("vsa_lora" in entries, task != "ref2va")
            self.assertEqual("ref2va_lora" in entries, task == "ref2va")
        # Selection never modifies the shared source manifest.
        self.assertNotIn("transformer/*", manifest["entries"][0]["allow_patterns"])

    def test_shared_root_and_offline_selection(self):
        plain = checkpoint_paths("checkpoints", include_offline=False)
        ref = checkpoint_paths("checkpoints", task="ref2va", include_offline=False)
        self.assertEqual(plain["h3_model"], ref["h3_model"])
        self.assertEqual(plain["qwen_checkpoint"], ref["qwen_checkpoint"])
        self.assertNotIn("offline_gemma", ref)
        self.assertNotIn("gemma_tokenizer", ref)
        self.assertNotIn("vsa_lora", ref)

    def test_native_input_vae_and_partition_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for component in ("transformer", "vae"):
                directory = root / component
                directory.mkdir()
                (directory / "config.json").write_text("{}")
                (directory / "diffusion_pytorch_model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": {"a": "weights.safetensors"}}))
                (directory / "weights.safetensors").touch()
            self.assertEqual(missing_h3_components(root, "fl2va"), [])
            self.assertTrue(all("transformer_ref" in name
                                for name in missing_h3_components(root, "ref2va")))
            (root / "vae/weights.safetensors").unlink()
            self.assertEqual(missing_h3_components(root, "t2va"), [])
            self.assertEqual(missing_h3_components(root, "fl2va"),
                             [str(root / "vae/weights.safetensors")])


if __name__ == "__main__":
    unittest.main()
