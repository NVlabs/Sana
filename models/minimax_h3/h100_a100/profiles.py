"""Locked MiniMax-H3 profiles shared by H100 and A100."""

from __future__ import annotations

from dataclasses import dataclass
import os


PINNED_IMAGE = (
    "docker://lmsysorg/sglang:nightly-dev-cu13-20260803-12eadf86"
)
PINNED_MODEL_REVISION = "bfc8ed0353f5a9733be73e6b2c98ec0948195b86"
PINNED_SGLANG_MODEL_SHA256 = (
    "5f87319969c446685ee93d422fc34a7c040defb238eff2274d664f2f8310e997"
)


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    display_name: str
    capability: tuple[int, int]
    sol_backend: str


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    sol_attention: bool
    tau: float
    threshold_type: str
    dense_steps: int
    dense_layers: int
    cache: str
    cache_threshold: float
    easycache_retain_steps: int = 0
    easycache_cooldown_steps: int = 0
    easycache_max_hits: int = 0


HARDWARE = {
    "h100": HardwareProfile(
        name="h100",
        display_name="4x NVIDIA H100 80GB HBM3",
        capability=(9, 0),
        sol_backend="cute_sm90",
    ),
    "a100": HardwareProfile(
        name="a100",
        display_name="4x NVIDIA A100-SXM4-80GB",
        capability=(8, 0),
        sol_backend="triton",
    ),
}


PROFILES = {
    "dense": RuntimeProfile(
        name="dense",
        sol_attention=False,
        tau=1.0,
        threshold_type="exact",
        dense_steps=10,
        dense_layers=2,
        cache="none",
        cache_threshold=0.08,
    ),
    "quality": RuntimeProfile(
        name="quality",
        sol_attention=True,
        tau=0.5,
        threshold_type="diag",
        dense_steps=15,
        dense_layers=2,
        cache="easycache",
        cache_threshold=0.30,
        easycache_retain_steps=10,
        easycache_cooldown_steps=4,
        easycache_max_hits=1,
    ),
    "balanced": RuntimeProfile(
        name="balanced",
        sol_attention=True,
        tau=1.0,
        threshold_type="diag",
        dense_steps=10,
        dense_layers=2,
        cache="easycache",
        cache_threshold=0.30,
        easycache_retain_steps=6,
        easycache_cooldown_steps=4,
        easycache_max_hits=2,
    ),
    "aggressive": RuntimeProfile(
        name="aggressive",
        sol_attention=True,
        tau=1.0,
        threshold_type="diag",
        dense_steps=10,
        dense_layers=2,
        cache="firstblock",
        cache_threshold=0.08,
    ),
    "fullopt_exact": RuntimeProfile(
        name="fullopt_exact",
        sol_attention=True,
        tau=1.0,
        threshold_type="exact",
        dense_steps=10,
        dense_layers=2,
        cache="firstblock",
        cache_threshold=0.08,
    ),
}


def _locked_env(name: str, value: str) -> None:
    current = os.environ.get(name)
    if current is not None and current != value:
        raise RuntimeError(
            f"{name} is locked by the selected MiniMax-H3 profile: "
            f"expected {value!r}, got {current!r}"
        )
    os.environ[name] = value


def configure_runtime() -> tuple[HardwareProfile, RuntimeProfile]:
    """Resolve one hardware/profile pair and lock its algorithm settings."""

    hardware_name = os.environ.get("H3_HARDWARE", "h100").strip().lower()
    profile_name = os.environ.get("H3_SOL_PROFILE", "dense").strip().lower()
    try:
        hardware = HARDWARE[hardware_name]
    except KeyError as exc:
        raise ValueError("H3_HARDWARE must be h100 or a100") from exc
    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(
            "H3_SOL_PROFILE must be dense, quality, balanced, aggressive, "
            "or fullopt_exact"
        ) from exc

    values = {
        "H3_NUM_GPUS": "4",
        "H3_ULYSSES_DEGREE": "4",
        "H3_MODEL_REVISION": PINNED_MODEL_REVISION,
        "H3_OFFLOAD": "0",
        "H3_TORCH_COMPILE": "0",
        "H3_REORDER": "0",
        "H3_POLICY_NAME": profile.name,
        "H3_SOL_ATTN": "1" if profile.sol_attention else "0",
        "H3_SOL_TAU": str(profile.tau),
        "H3_SOL_THRESH_TYPE": profile.threshold_type,
        "H3_SOL_DENSE_STEPS": str(profile.dense_steps),
        "H3_SOL_DENSE_LAYERS": str(profile.dense_layers),
        "H3_SOL_SINK_MODE": "prefix",
        "H3_SOL_DENSITY_MODE": "warmup",
        "H3_SOL_CORRECTNESS_GATE": "1",
        "H3_FIRSTBLOCKCACHE": "1" if profile.cache == "firstblock" else "0",
        "H3_EASYCACHE": "1" if profile.cache == "easycache" else "0",
        "H3_CACHE_THRESHOLD": str(profile.cache_threshold),
        "H3_EASYCACHE_THRESHOLD": str(profile.cache_threshold),
        "H3_EASYCACHE_RETAIN_STEPS": str(profile.easycache_retain_steps),
        "H3_EASYCACHE_COOLDOWN_STEPS": str(profile.easycache_cooldown_steps),
        "H3_EASYCACHE_MAX_HITS": str(profile.easycache_max_hits),
        "H3_EXPECTED_SOL_BACKEND": hardware.sol_backend,
        "SOL_ATTN_STRICT": "1",
    }
    for name, value in values.items():
        _locked_env(name, value)
    return hardware, profile


__all__ = [
    "HARDWARE",
    "PINNED_IMAGE",
    "PINNED_MODEL_REVISION",
    "PINNED_SGLANG_MODEL_SHA256",
    "PROFILES",
    "HardwareProfile",
    "RuntimeProfile",
    "configure_runtime",
]
