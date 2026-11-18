"""Architecture dispatch contracts; no CUDA device is required."""

import pytest

from sol_attn.interface import _backend_for_arch


@pytest.mark.parametrize(
    "arch,expected",
    [
        ((8, 0), "triton"),
        ((8, 9), "cute_sm89"),
        ((9, 0), "cute_sm90"),
        ((10, 0), "cute_sm100"),
        ((10, 3), "cute_sm100"),
        ((10, 1), "triton"),
        ((12, 0), "cute_sm120"),
    ],
)
def test_backend_for_arch(arch, expected):
    assert _backend_for_arch(arch, cute_available=True) == expected
    assert _backend_for_arch(arch, cute_available=False) == "triton"


def test_unsupported_architecture_is_rejected():
    with pytest.raises(RuntimeError, match="compute capability >= 8.0"):
        _backend_for_arch((7, 5), cute_available=True)
