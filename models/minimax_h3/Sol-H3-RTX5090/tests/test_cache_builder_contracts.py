"""Offline cache construction contracts that need no model framework."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from runtime.cache_builder import build_cache, prompt_stats
from runtime.cache_ops.int8_header import (
    Int8ConvRotLayerSpec, _expected_gemma_modules, partition_dev_int8_convrot_specs,
)


def specs():
    return tuple(Int8ConvRotLayerSpec(f"{prefix}{index}.proj", (16, 256), (16, 1), 256)
                 for prefix, count in (("transformer_blocks.", 1344),
                                       ("audio_embeddings_connector.", 48),
                                       ("video_embeddings_connector.", 48))
                 for index in range(count))


class Tokenizer:
    bos_token_id = 2

    def __init__(self, ids, limit):
        self.ids, self.max_length, self.tokenizer = ids, limit, self

    def __call__(self, *args, **kwargs):
        return SimpleNamespace(input_ids=self.ids)

    def tokenize_with_weights(self, prompt):
        tokens = list(self.ids)
        if not tokens or tokens[0] != self.bos_token_id:
            tokens.insert(0, self.bos_token_id)
        return {"gemma": [(token, 1) for token in tokens[:self.max_length]]}


class CacheBuilderContracts(unittest.TestCase):
    def test_connector_keys_preserve_both_modalities(self):
        transformer, connector = partition_dev_int8_convrot_specs(specs())
        self.assertEqual(len(transformer), 1344)
        self.assertEqual(len(connector), 96)
        self.assertEqual(connector[0].module_name, "audio_connector.0.proj")
        self.assertEqual(connector[-1].module_name, "video_connector.47.proj")
        self.assertTrue(all(spec.convrot_groupsize == 256 for spec in connector))

    def test_partial_or_duplicate_checkpoint_schema_fails(self):
        rows = specs()
        with self.assertRaises(ValueError):
            partition_dev_int8_convrot_specs(rows[:-1])
        with self.assertRaises(ValueError):
            partition_dev_int8_convrot_specs((*rows, rows[0]))

    def test_gemma_shared_kv_topology_is_retained(self):
        names = _expected_gemma_modules()
        self.assertEqual(len(names), 328)
        self.assertIn("model.layers.4.self_attn.v_proj", names)
        self.assertNotIn("model.layers.5.self_attn.v_proj", names)
        self.assertIn("model.layers.5.self_attn.k_proj", names)

    def test_bos_count_matches_native_prompt_encoding(self):
        with_bos = prompt_stats(Tokenizer([2, 7, 8], 1024), "two words")
        without_bos = prompt_stats(Tokenizer([7, 8], 1024), "two words")
        self.assertEqual(with_bos, without_bos)
        self.assertEqual(with_bos["raw_tokens_including_bos"], 3)
        self.assertFalse(with_bos["truncated"])

    def test_truncated_prompt_is_rejected(self):
        with self.assertRaises(ValueError):
            prompt_stats(Tokenizer([7, 8, 9], 3), "a b c")

    def test_existing_cache_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "cache.pt"
            target.write_bytes(b"existing")
            with self.assertRaises(FileExistsError):
                build_cache({}, target)
            self.assertEqual(target.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
