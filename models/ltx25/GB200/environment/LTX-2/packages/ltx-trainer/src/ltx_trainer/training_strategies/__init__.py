"""Training strategies for different conditioning modes.
This package implements the Strategy Pattern to handle different training modes:
- Text-to-video training (standard generation, optionally with audio) [DEPRECATED]
- Video-to-video training (IC-LoRA mode with reference videos) [DEPRECATED]
- Flexible training (unified conditioning framework supporting all scenarios) [RECOMMENDED]
Each strategy encapsulates the specific logic for preparing model inputs and computing loss.
"""

import warnings

from ltx_trainer import logger
from ltx_trainer.training_strategies.base_strategy import (
    DEFAULT_FPS,
    ModelInputs,
    TrainingStrategy,
    TrainingStrategyConfigBase,
)
from ltx_trainer.training_strategies.flexible import FlexibleStrategy, FlexibleStrategyConfig
from ltx_trainer.training_strategies.text_to_video import TextToVideoConfig, TextToVideoStrategy
from ltx_trainer.training_strategies.video_to_video import VideoToVideoConfig, VideoToVideoStrategy

# Type alias for all strategy config types
TrainingStrategyConfig = TextToVideoConfig | VideoToVideoConfig | FlexibleStrategyConfig

__all__ = [
    "DEFAULT_FPS",
    "FlexibleStrategy",
    "FlexibleStrategyConfig",
    "ModelInputs",
    "TextToVideoConfig",
    "TextToVideoStrategy",
    "TrainingStrategy",
    "TrainingStrategyConfig",
    "TrainingStrategyConfigBase",
    "VideoToVideoConfig",
    "VideoToVideoStrategy",
    "get_training_strategy",
]


def get_training_strategy(config: TrainingStrategyConfig) -> TrainingStrategy:
    """Factory function to create the appropriate training strategy.
    The strategy is determined by the `name` field in the configuration.
    Args:
        config: Strategy-specific configuration with a `name` field
    Returns:
        The appropriate training strategy instance
    Raises:
        ValueError: If strategy name is not supported
    Note:
        The `text_to_video` and `video_to_video` strategies are deprecated.
        Please use the `flexible` strategy instead.
    """

    match config:
        case TextToVideoConfig():
            warnings.warn(
                "The 'text_to_video' training strategy is deprecated and will be removed "
                "in a future version. Please migrate to the 'flexible' strategy. "
                "See the migration guide in the documentation.",
                DeprecationWarning,
                stacklevel=2,
            )
            strategy = TextToVideoStrategy(config)
        case VideoToVideoConfig():
            warnings.warn(
                "The 'video_to_video' training strategy is deprecated and will be removed "
                "in a future version. Please migrate to the 'flexible' strategy. "
                "See the migration guide in the documentation.",
                DeprecationWarning,
                stacklevel=2,
            )
            strategy = VideoToVideoStrategy(config)
        case FlexibleStrategyConfig():
            strategy = FlexibleStrategy(config)
        case _:
            raise ValueError(f"Unknown training strategy config type: {type(config).__name__}")

    # Determine audio mode for logging
    if hasattr(config, "with_audio"):
        audio_mode = "(audio enabled 🔈)" if config.with_audio else "(audio disabled 🔇)"
    elif hasattr(config, "audio") and config.audio is not None:
        audio_mode = "(audio enabled 🔈)"
    else:
        audio_mode = "(audio disabled 🔇)"

    logger.debug(f"🎯 Using {strategy.__class__.__name__} training strategy {audio_mode}")
    return strategy
