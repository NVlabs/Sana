from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


def _ltx2_vocoder_core_config(config: dict, prefix: str = "") -> dict:
    def get(name: str, default=None):
        return config.get(f"{prefix}{name}", default)

    act_fn = get("act_fn", "snake")
    final_act_fn = get("final_act_fn", "tanh")

    return {
        "resblock_kernel_sizes": get("resnet_kernel_sizes"),
        "upsample_rates": get("upsample_factors"),
        "upsample_kernel_sizes": get("upsample_kernel_sizes"),
        "resblock_dilation_sizes": get("resnet_dilations"),
        "upsample_initial_channel": get("hidden_channels", 1024),
        "resblock": "AMP1" if str(act_fn).lower() in ("snake", "snakebeta") else "1",
        "activation": act_fn,
        "use_tanh_at_final": str(final_act_fn).lower() == "tanh",
        "apply_final_activation": final_act_fn is not None,
        "use_bias_at_final": bool(get("final_bias", True)),
    }


def _maybe_add_ltx2_vocoder_bwe_config(config: dict, vocoder_config) -> None:
    if not any(key.startswith("bwe_") for key in config):
        return

    vocoder_config.arch_config.vocoder = {
        "vocoder": _ltx2_vocoder_core_config(config),
        "bwe": {
            **_ltx2_vocoder_core_config(config, prefix="bwe_"),
            "input_sampling_rate": config["input_sampling_rate"],
            "output_sampling_rate": config["output_sampling_rate"],
            "n_fft": config["filter_length"],
            "hop_length": config["hop_length"],
            "win_size": config.get("window_length", config["filter_length"]),
            "num_mels": config["num_mel_channels"],
        },
    }


def _remap_ltx2_vocoder_state_dict(loaded: dict, target_keys: set[str]) -> dict:
    remapped = {}
    for key, value in loaded.items():
        mapped_key = key.replace(".downsample.filter", ".downsample.lowpass.filter")
        if mapped_key != key and mapped_key in target_keys:
            remapped[mapped_key] = value
        else:
            remapped[key] = value
    return remapped


class VocoderLoader(ComponentLoader):
    component_names = ["vocoder"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        if class_name == "LTX2VocoderWithBWE":
            class_name = "LTX2Vocoder"

        server_args.model_paths[component_name] = component_model_path

        from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import (
            LTXVocoderConfig,
        )

        vocoder_config = LTXVocoderConfig()
        vocoder_config.update_model_arch(config)
        _maybe_add_ltx2_vocoder_bwe_config(config, vocoder_config)

        try:
            vocoder_precision = server_args.pipeline_config.audio_vae_precision
        except AttributeError:
            vocoder_precision = "fp32"
        vocoder_dtype = PRECISION_TO_TYPE[vocoder_precision]

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        with set_default_torch_dtype(vocoder_dtype), skip_init_modules():
            vocoder_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vocoder = vocoder_cls(vocoder_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        loaded = _remap_ltx2_vocoder_state_dict(
            loaded, target_keys=set(vocoder.state_dict().keys())
        )
        incompatible = vocoder.load_state_dict(loaded, strict=False)
        missing_keys = []
        unexpected_keys = []
        try:
            missing_keys = incompatible.missing_keys
            unexpected_keys = incompatible.unexpected_keys
        except AttributeError:
            # Best-effort fallback in case older torch returns a tuple-like.
            try:
                missing_keys = incompatible[0]
                unexpected_keys = incompatible[1]
            except Exception:
                pass

        if missing_keys or unexpected_keys:
            logger.warning(
                "Loaded vocoder with missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
        return vocoder
