from dataclasses import asdict, dataclass, field


@dataclass
class Gemma3RopeScaling:
    factor: float = 8.0
    rope_type: str = "linear"


@dataclass
class Gemma3TextConfig:
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attn_logit_softcapping: float | None = None
    cache_implementation: str = "hybrid"
    final_logit_softcapping: float | None = None
    head_dim: int = 256
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size: int = 3840
    initializer_range: float = 0.02
    intermediate_size: int = 15360
    max_position_embeddings: int = 131072
    model_type: str = "gemma3_text"
    num_attention_heads: int = 16
    num_hidden_layers: int = 48
    num_key_value_heads: int = 8
    query_pre_attn_scalar: int = 256
    rms_norm_eps: float = 1e-06
    rope_local_base_freq: int = 10000
    rope_scaling: Gemma3RopeScaling = field(default_factory=Gemma3RopeScaling)
    rope_theta: int = 1000000
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    torch_dtype: str = "float32"
    use_cache: bool = True
    vocab_size: int = 262208


@dataclass
class Gemma3VisionConfig:
    attention_dropout: float = 0.0
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_size: int = 1152
    image_size: int = 896
    intermediate_size: int = 4304
    layer_norm_eps: float = 1e-06
    model_type: str = "siglip_vision_model"
    num_attention_heads: int = 16
    num_channels: int = 3
    num_hidden_layers: int = 27
    patch_size: int = 14
    torch_dtype: str = "float32"
    vision_use_head: bool = False


@dataclass
class Gemma3ConfigData:
    architectures: list[str] = field(default_factory=lambda: ["Gemma3ForConditionalGeneration"])
    boi_token_index: int = 255999
    eoi_token_index: int = 256000
    eos_token_id: list[int] = field(default_factory=lambda: [1, 106])
    image_token_index: int = 262144
    initializer_range: float = 0.02
    mm_tokens_per_image: int = 256
    model_type: str = "gemma3"
    text_config: Gemma3TextConfig = field(default_factory=Gemma3TextConfig)
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.51.0"
    vision_config: Gemma3VisionConfig = field(default_factory=Gemma3VisionConfig)

    def to_dict(self) -> dict:
        return asdict(self)


GEMMA3_CONFIG_FOR_LTX = Gemma3ConfigData()
