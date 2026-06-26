"""
This unittest is introduced in #22360, preventing duplicate transformer safetensors variants being loaded together
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

partial_json_parser = types.ModuleType("partial_json_parser")
partial_json_parser_core = types.ModuleType("partial_json_parser.core")
partial_json_parser_exceptions = types.ModuleType("partial_json_parser.core.exceptions")
partial_json_parser_options = types.ModuleType("partial_json_parser.core.options")


class _MalformedJSON(Exception):
    pass


class _Allow:
    STR = 1
    OBJ = 2
    ARR = 4
    ALL = STR | OBJ | ARR


def _loads(input_str, _flags=None):
    return json.loads(input_str)


partial_json_parser_exceptions.MalformedJSON = _MalformedJSON
partial_json_parser_options.Allow = _Allow
partial_json_parser.loads = _loads
sys.modules.setdefault("partial_json_parser", partial_json_parser)
sys.modules.setdefault("partial_json_parser.core", partial_json_parser_core)
sys.modules.setdefault(
    "partial_json_parser.core.exceptions", partial_json_parser_exceptions
)
sys.modules.setdefault("partial_json_parser.core.options", partial_json_parser_options)

from sglang.multimodal_gen.runtime.layers.linear import UnquantizedLinearMethod
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    _prepare_nvfp4_weight_bytes,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    _filter_duplicate_precision_variant_safetensors,
    _Flux2Nvfp4FallbackAdapter,
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.models.dits.flux import FluxSingleTransformerBlock
from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _ltx2_runtime_module_name_variants,
    _matches_any_module_variant,
    _matches_any_pattern,
    _preset_patterns,
    _updated_quant_config,
    build_modelopt_nvfp4_transformer,
)


class _FakeFluxTransformer:
    pass


class _FakeQuantConfig:
    @classmethod
    def get_name(cls):
        return "modelopt_fp4"


class TestTransformerQuantHelpers(unittest.TestCase):
    def _make_server_args(self, **overrides):
        defaults = dict(
            transformer_weights_path=None,
            pipeline_config=SimpleNamespace(
                dit_precision="bf16",
                dit_config=SimpleNamespace(
                    arch_config=SimpleNamespace(param_names_mapping={})
                ),
            ),
            nunchaku_config=None,
            quantization=None,
            tp_size=1,
            dit_cpu_offload=False,
            text_encoder_cpu_offload=False,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_transformer_safetensors_to_load_uses_single_override_file(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            server_args = self._make_server_args(transformer_weights_path=f.name)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [f.name])

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_safetensors_to_load_prefers_mixed_export(
        self, _mock_download
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            mixed = f"{tmpdir}/flux2-dev-nvfp4-mixed.safetensors"
            full = f"{tmpdir}/flux2-dev-nvfp4.safetensors"
            open(mixed, "a").close()
            open(full, "a").close()

            server_args = self._make_server_args(transformer_weights_path=tmpdir)
            resolved = resolve_transformer_safetensors_to_load(
                server_args, "/unused/component/path"
            )

        self.assertEqual(resolved, [mixed])

    def test_filter_transformer_precision_variants_prefers_canonical_file(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.safetensors",
            "/tmp/transformer/other.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(
            resolved,
            [
                "/tmp/transformer/diffusion_pytorch_model.safetensors",
                "/tmp/transformer/other.safetensors",
            ],
        )

    def test_filter_transformer_precision_variants_keeps_precision_only_family(self):
        files = [
            "/tmp/transformer/diffusion_pytorch_model.bf16.safetensors",
            "/tmp/transformer/diffusion_pytorch_model.fp16.safetensors",
        ]

        resolved = _filter_duplicate_precision_variant_safetensors(files)

        self.assertEqual(resolved, files)

    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.build_nvfp4_config_from_safetensors_list",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_quant_config_from_safetensors_metadata",
        return_value=None,
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.get_metadata_from_safetensors_file"
    )
    @patch(
        "sglang.multimodal_gen.runtime.loader.transformer_load_utils.maybe_download_model",
        side_effect=lambda path, **kw: path,
    )
    def test_resolve_transformer_quant_load_spec_keeps_nunchaku_hook(
        self,
        _mock_download,
        mock_metadata,
        _mock_quant_metadata,
        mock_maybe_download,
        _mock_nvfp4,
    ):
        mock_maybe_download.side_effect = AssertionError(
            "local safetensors path should not trigger maybe_download_model"
        )
        mock_metadata.return_value = {
            "config": json.dumps({"_class_name": _FakeFluxTransformer.__name__})
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            nunchaku_config = NunchakuConfig(transformer_weights_path=f.name)
            server_args = self._make_server_args(
                transformer_weights_path=nunchaku_config.transformer_weights_path,
                nunchaku_config=nunchaku_config,
            )

            spec = resolve_transformer_quant_load_spec(
                hf_config={},
                server_args=server_args,
                safetensors_list=[nunchaku_config.transformer_weights_path],
                component_model_path="/unused/component/path",
                model_cls=_FakeFluxTransformer,
                cls_name=_FakeFluxTransformer.__name__,
            )

        self.assertIsNone(spec.quant_config)
        self.assertIs(spec.nunchaku_config, nunchaku_config)
        self.assertIsNone(spec.param_dtype)
        self.assertEqual(len(spec.post_load_hooks), 1)
        self.assertIs(nunchaku_config.model_cls, _FakeFluxTransformer)
        mock_maybe_download.assert_not_called()

    def test_flux2_mixed_nvfp4_fallback_disables_conflicting_offloads(self):
        server_args = self._make_server_args(
            transformer_weights_path="/tmp/flux2-dev-nvfp4-mixed.safetensors",
            tp_size=2,
            dit_cpu_offload=True,
            text_encoder_cpu_offload=True,
        )

        _Flux2Nvfp4FallbackAdapter._maybe_adjust_flux2_nvfp4_fallback_defaults(
            cls_name="Flux2Transformer2DModel",
            server_args=server_args,
            quant_config=_FakeQuantConfig(),
        )

        self.assertFalse(server_args.dit_cpu_offload)
        self.assertFalse(server_args.text_encoder_cpu_offload)

    def test_prepare_nvfp4_weight_bytes_swaps_nibbles(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=True)

        self.assertEqual(prepared.tolist(), [[0xBA, 0x01]])

    def test_prepare_nvfp4_weight_bytes_can_skip_nibble_swap(self):
        weight = torch.tensor([[0xAB, 0x10]], dtype=torch.uint8)

        prepared = _prepare_nvfp4_weight_bytes(weight, swap_weight_nibbles=False)

        self.assertEqual(prepared.tolist(), [[0xAB, 0x10]])

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_flat_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
                "swap_weight_nibbles": False,
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_reads_swap_weight_nibbles_from_nested_config(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "swap_weight_nibbles": False,
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertFalse(config.swap_weight_nibbles)

    def test_modelopt_fp4_config_defaults_to_swizzled_weight_scales(self):
        config = ModelOptFp4Config.from_config(
            {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "ignore": [],
            }
        )

        self.assertEqual(config.weight_scale_layout, "swizzled")

    def test_modelopt_fp4_config_reads_weight_scale_layout(self):
        config = ModelOptFp4Config.from_config(
            {
                "quantization": {
                    "quant_algo": "NVFP4",
                    "exclude_modules": [],
                    "weight_scale_layout": "linear",
                },
                "config_groups": {"default": {"weights": {"group_size": 16}}},
            }
        )

        self.assertEqual(config.weight_scale_layout, "linear")

    def test_builder_adds_diffusers_quant_type_for_nvfp4(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=["single_transformer_blocks.*.proj_mlp*"],
            swap_weight_nibbles=False,
        )

        self.assertEqual(updated["quantization_config"]["quant_type"], "NVFP4")
        self.assertEqual(
            updated["quantization_config"]["ignore"],
            ["single_transformer_blocks.*.proj_mlp*"],
        )

    def test_ltx2_nvfp4_preset_matches_expected_fallback_modules(self):
        patterns = _preset_patterns("ltx2-nvfp4")

        self.assertTrue(_matches_any_pattern("patchify_proj", patterns))
        self.assertTrue(_matches_any_pattern("audio_proj_out", patterns))
        self.assertTrue(
            _matches_any_pattern(
                "adaln_single.emb.timestep_embedder.linear_1", patterns
            )
        )
        self.assertTrue(_matches_any_pattern("audio_adaln_single.linear", patterns))
        self.assertTrue(
            _matches_any_pattern(
                "av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_2",
                patterns,
            )
        )
        self.assertTrue(
            _matches_any_pattern("transformer_blocks.0.attn1.to_q", patterns)
        )
        self.assertTrue(
            _matches_any_pattern(
                "transformer_blocks.43.audio_to_video_attn.to_out.0", patterns
            )
        )
        self.assertTrue(
            _matches_any_pattern("transformer_blocks.47.audio_ff.proj_out", patterns)
        )
        self.assertFalse(
            _matches_any_pattern("transformer_blocks.1.attn1.to_q", patterns)
        )
        self.assertTrue(
            _matches_any_module_variant(
                "proj_in",
                patterns,
                pattern_preset="ltx2-nvfp4",
            )
        )
        self.assertTrue(
            _matches_any_module_variant(
                "model.diffusion_model.time_embed.emb.timestep_embedder.linear_1",
                patterns,
                pattern_preset="ltx2-nvfp4",
            )
        )
        self.assertTrue(
            _matches_any_module_variant(
                "transformer_blocks.47.audio_ff.net.2",
                patterns,
                pattern_preset="ltx2-nvfp4",
            )
        )

    def test_ltx2_nvfp4_name_variants_include_runtime_module_names(self):
        self.assertIn(
            "patchify_proj",
            _ltx2_runtime_module_name_variants("model.diffusion_model.proj_in"),
        )
        self.assertIn(
            "adaln_single.emb.timestep_embedder.linear_2",
            _ltx2_runtime_module_name_variants(
                "time_embed.emb.timestep_embedder.linear_2"
            ),
        )
        self.assertIn(
            "transformer_blocks.43.ff.proj_in",
            _ltx2_runtime_module_name_variants("transformer_blocks.43.ff.net.0.proj"),
        )

    def test_builder_accepts_ltx2_nvfp4_preset_in_quant_config(self):
        updated = _updated_quant_config(
            {
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                }
            },
            fallback_patterns=_preset_patterns("ltx2-nvfp4"),
            swap_weight_nibbles=True,
        )

        ignore = updated["quantization_config"]["ignore"]
        self.assertIn("patchify_proj", ignore)
        self.assertIn("transformer_blocks.47.audio_ff.proj_out", ignore)
        self.assertTrue(updated["quantization_config"]["swap_weight_nibbles"])
        self.assertEqual(
            updated["quantization_config"]["weight_scale_layout"], "swizzled"
        )

    def test_ltx2_nvfp4_builder_handles_x0_single_file_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base.safetensors"
            source = root / "source-fp4.safetensors"
            output_dir = root / "out"
            patchify_key = "model.diffusion_model.proj_in"
            runtime_patchify_key = "patchify_proj"
            fallback_ff_key = (
                "model.diffusion_model.transformer_blocks.47.audio_ff.net.2"
            )
            runtime_fallback_ff_key = "transformer_blocks.47.audio_ff.proj_out"
            quantized_attn_key = "model.diffusion_model.transformer_blocks.1.attn1.to_q"
            audio_vae_key = "audio_vae.decoder.conv_in.conv.weight"
            metadata = {
                "config": json.dumps(
                    {"transformer": {"_class_name": "AVTransformer3DModel"}}
                ),
                "_quantization_metadata": json.dumps(
                    {
                        "format_version": "1.0",
                        "layers": {
                            patchify_key: {"format": "nvfp4"},
                            quantized_attn_key: {"format": "nvfp4"},
                        },
                    }
                ),
            }

            save_file(
                {
                    f"{patchify_key}.weight": torch.full((2, 8), 1, dtype=torch.uint8),
                    f"{patchify_key}.weight_scale": torch.ones(
                        (2, 1), dtype=torch.float32
                    ),
                    f"{fallback_ff_key}.weight": torch.full(
                        (2, 8), 2, dtype=torch.uint8
                    ),
                    f"{fallback_ff_key}.weight_scale": torch.ones(
                        (2, 1), dtype=torch.float32
                    ),
                    f"{quantized_attn_key}.weight": torch.full(
                        (2, 8), 3, dtype=torch.uint8
                    ),
                    f"{quantized_attn_key}.weight_scale": torch.ones(
                        (2, 1), dtype=torch.float32
                    ),
                    audio_vae_key: torch.ones((1, 1, 1, 1), dtype=torch.bfloat16),
                },
                source,
                metadata=metadata,
            )
            save_file(
                {
                    f"{runtime_patchify_key}.weight": torch.full(
                        (2, 16), 7, dtype=torch.bfloat16
                    ),
                    f"{runtime_fallback_ff_key}.weight": torch.full(
                        (2, 16), 9, dtype=torch.bfloat16
                    ),
                    f"{quantized_attn_key}.weight": torch.full(
                        (2, 16), 11, dtype=torch.bfloat16
                    ),
                },
                base,
            )

            stats = build_modelopt_nvfp4_transformer(
                base_transformer_dir=str(base),
                modelopt_hf_dir=str(source),
                output_dir=str(output_dir),
                pattern_preset="ltx2-nvfp4",
            )

            config = json.loads((output_dir / "config.json").read_text())
            quant_config = config["quantization_config"]
            tensors = load_file(output_dir / source.name)
            index = json.loads(
                (output_dir / f"{source.stem}.safetensors.index.json").read_text()
            )

            self.assertEqual(config["_class_name"], "LTX2VideoTransformer3DModel")
            self.assertEqual(quant_config["group_size"], 16)
            self.assertIn("patchify_proj", quant_config["ignore"])
            self.assertIn(
                "transformer_blocks.47.audio_ff.proj_out",
                quant_config["ignore"],
            )
            self.assertEqual(
                tensors[f"{patchify_key}.weight"].dtype,
                torch.bfloat16,
            )
            self.assertNotIn(f"{patchify_key}.weight_scale", tensors)
            self.assertEqual(
                tensors[f"{fallback_ff_key}.weight"].dtype,
                torch.bfloat16,
            )
            self.assertNotIn(f"{fallback_ff_key}.weight_scale", tensors)
            self.assertEqual(
                tensors[f"{quantized_attn_key}.weight"].dtype,
                torch.uint8,
            )
            self.assertNotIn(audio_vae_key, tensors)
            self.assertNotIn(audio_vae_key, index["weight_map"])
            self.assertEqual(stats["fallback_modules"], 2)
            self.assertEqual(stats["replaced_tensors"], 2)
            self.assertEqual(stats["removed_aux_tensors"], 2)

    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_rank", return_value=0)
    @patch("sglang.multimodal_gen.runtime.layers.linear.get_group_size", return_value=1)
    @patch(
        "sglang.multimodal_gen.runtime.layers.linear.get_tp_group", return_value=None
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.layer.get_ring_parallel_world_size",
        return_value=1,
    )
    @patch(
        "sglang.multimodal_gen.runtime.layers.attention.selector.get_global_server_args",
        return_value=SimpleNamespace(attention_backend=None),
    )
    def test_flux_single_transformer_block_modelopt_excludes_use_full_prefix(
        self,
        _mock_server_args,
        _mock_ring_world_size,
        _mock_tp_group,
        _mock_group_size,
        _mock_group_rank,
    ):
        quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            exclude_modules=[
                "single_transformer_blocks.*.proj_mlp*",
                "single_transformer_blocks.*.proj_out*",
                "single_transformer_blocks.*.attn.to_q",
            ],
        )

        block = FluxSingleTransformerBlock(
            dim=64,
            num_attention_heads=4,
            attention_head_dim=16,
            mlp_ratio=2.0,
            quant_config=quant_config,
            prefix="single_transformer_blocks.0",
        )

        self.assertEqual(block.proj_mlp.prefix, "single_transformer_blocks.0.proj_mlp")
        self.assertEqual(block.proj_out.prefix, "single_transformer_blocks.0.proj_out")
        self.assertEqual(
            block.attn.to_q.prefix, "single_transformer_blocks.0.attn.to_q"
        )
        self.assertIsInstance(block.proj_mlp.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.proj_out.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(block.attn.to_q.quant_method, UnquantizedLinearMethod)


if __name__ == "__main__":
    unittest.main()
