"""Normalized model-component paths for monolith and split checkpoint layouts.
``ModelPaths`` is the post-parse contract for component locations. The CLI builds it
(via :func:`model_paths_from_namespace` or an explicit factory); pipelines and blocks
fetch the slots they need via typed accessors (``transformer()``, ``text_encoder()``, …)
that return ``str`` or raise. Unused slots stay ``None`` — never filled with dummy paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ModelPaths:
    """Resolved paths for LTX components.
    Construct via :meth:`from_monolith` or :meth:`from_split` — those factories set
    :attr:`embeddings_weight_paths` explicitly. ``mode`` is descriptive only; it is not
    consulted when resolving embeddings sources.
    Component slots other than those a given pipeline loads may be ``None``. Call the
    typed accessors at the use site so a missing path fails where it is needed.
    Two mutually exclusive modes:
    * ``monolith`` — legacy fat checkpoint (+ optional ``gemma_root`` directory).
    * ``split`` — Comfy-aligned layout with explicit per-component safetensors.
    """

    mode: Literal["monolith", "split"]
    transformer_path: str | None
    text_encoder_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    duration_head_path: str | None
    # Weight sources for EmbeddingsProcessor (config / gemma_source_checkpoint from first
    # path). Monolith: (checkpoint,). Split with TE: (transformer, text_encoder).
    embeddings_weight_paths: tuple[str, ...]

    def transformer(self) -> str:
        return self._require("transformer_path")

    def text_encoder(self) -> str:
        return self._require("text_encoder_path")

    def video_vae(self) -> str:
        return self._require("video_vae_path")

    def audio_vae(self) -> str:
        return self._require("audio_vae_path")

    def duration_head(self) -> str:
        return self._require("duration_head_path")

    def _require(self, attr: str) -> str:
        if attr not in _COMPONENT_ATTRS:
            raise ValueError(f"Unknown ModelPaths component {attr!r}. Expected one of {_COMPONENT_ATTRS}.")
        value = getattr(self, attr)
        if value is None:
            raise ValueError(
                f"ModelPaths.{attr} is required but missing (mode={self.mode!r}). "
                f"Pass the corresponding CLI flag or factory argument."
            )
        return value

    @classmethod
    def from_monolith(
        cls,
        checkpoint_path: str,
        gemma_root: str | None = None,
        *,
        video_vae_path: str | None = None,
    ) -> ModelPaths:
        """Build paths for a fat checkpoint.
        ``gemma_root`` may be ``None`` when the caller does not encode prompts (e.g. HDR
        with precomputed embeddings). Other components point at the checkpoint file
        itself (or ``video_vae_path`` for the video VAE override).
        """
        return cls(
            mode="monolith",
            transformer_path=checkpoint_path,
            text_encoder_path=gemma_root,
            video_vae_path=video_vae_path or checkpoint_path,
            audio_vae_path=checkpoint_path,
            duration_head_path=checkpoint_path,
            embeddings_weight_paths=(checkpoint_path,),
        )

    @classmethod
    def from_split(
        cls,
        *,
        transformer_path: str | None = None,
        text_encoder_path: str | None = None,
        video_vae_path: str | None = None,
        audio_vae_path: str | None = None,
        duration_head_path: str | None = None,
    ) -> ModelPaths:
        """Build paths for an explicit split pack; omitted components stay ``None``."""
        if transformer_path is not None and text_encoder_path is not None:
            embeddings: tuple[str, ...] = (transformer_path, text_encoder_path)
        elif transformer_path is not None:
            embeddings = (transformer_path,)
        else:
            embeddings = ()
        return cls(
            mode="split",
            transformer_path=transformer_path,
            text_encoder_path=text_encoder_path,
            video_vae_path=video_vae_path,
            audio_vae_path=audio_vae_path,
            duration_head_path=duration_head_path,
            embeddings_weight_paths=embeddings,
        )


_COMPONENT_ATTRS = frozenset(
    {
        "transformer_path",
        "text_encoder_path",
        "video_vae_path",
        "audio_vae_path",
        "duration_head_path",
    }
)

# Flags that select split pack mode. ``video_vae_path`` is intentionally omitted: it is
# a shared slot (monolith DiffVAE override *and* split component) and must not flip XOR.
_SPLIT_PACK_FLAGS = (
    "transformer_path",
    "text_encoder_path",
    "audio_vae_path",
    "duration_head_path",
)

_SPLIT_COMPONENT_FLAGS = (
    "transformer_path",
    "text_encoder_path",
    "video_vae_path",
    "audio_vae_path",
    "duration_head_path",
)


def model_paths_from_namespace(namespace: argparse.Namespace) -> ModelPaths:
    """Build :class:`ModelPaths` from a parsed CLI namespace.
    Enforces monolith XOR split. Split mode is selected by any pack-identifying
    flag (``--transformer-path``, ``--text-encoder-path``, ``--audio-vae-path``,
    ``--duration-head-path``); ``--video-vae-path`` fills the shared video VAE
    slot in either mode. Pipelines fetch components via typed accessors. Monolith
    mode still needs a checkpoint plus ``--gemma-root`` (shared prompt-encoding CLIs).
    """
    split_values = {name: getattr(namespace, name, None) for name in _SPLIT_COMPONENT_FLAGS}
    pack_set = {k: v for k, v in split_values.items() if k in _SPLIT_PACK_FLAGS and v is not None}
    video_vae = split_values["video_vae_path"]

    monolith_ckpt = getattr(namespace, "checkpoint_path", None) or getattr(namespace, "distilled_checkpoint_path", None)
    gemma_root = getattr(namespace, "gemma_root", None)

    if pack_set and (monolith_ckpt is not None or gemma_root is not None):
        raise SystemExit(
            "Split pack flags (--transformer-path, --text-encoder-path, "
            "--audio-vae-path, --duration-head-path) cannot be combined with "
            "--checkpoint-path / --distilled-checkpoint-path / --gemma-root. "
            "--video-vae-path is allowed in either mode (shared video VAE slot)."
        )

    if pack_set:
        return ModelPaths.from_split(
            transformer_path=split_values["transformer_path"],
            text_encoder_path=split_values["text_encoder_path"],
            video_vae_path=video_vae,
            audio_vae_path=split_values["audio_vae_path"],
            duration_head_path=split_values["duration_head_path"],
        )

    if monolith_ckpt is None or gemma_root is None:
        raise SystemExit(
            "Provide either monolith args (--checkpoint-path or --distilled-checkpoint-path "
            "+ --gemma-root) or one or more split pack flags "
            "(--transformer-path, --text-encoder-path, --audio-vae-path, "
            "--duration-head-path). --video-vae-path fills the video VAE slot in either mode."
        )

    # Monolith default for video VAE is the fat checkpoint; only pass an explicit override.
    override = video_vae if video_vae is not None and video_vae != monolith_ckpt else None
    return ModelPaths.from_monolith(
        monolith_ckpt,
        gemma_root,
        video_vae_path=override,
    )
