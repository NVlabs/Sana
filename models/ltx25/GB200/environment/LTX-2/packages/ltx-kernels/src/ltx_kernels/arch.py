"""GPU architecture detection for blockwise kernel dispatch."""

import torch


def get_device_arch() -> str:
    """Return a coarse architecture name for the current CUDA device.
    Used to pick the FP8 GEMM kernel variant (SM89 vs SM90) at runtime.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; blockwise kernels require a CUDA device.")
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    if major == 8 and (minor >= 0 and minor < 9):
        return "ampere"
    if major == 8 and minor == 9:
        return "ada"
    if major == 9 and minor == 0:
        return "hopper"
    if major in {10, 12}:
        return "blackwell"
    raise NotImplementedError(f"Unsupported GPU compute capability sm_{major}{minor}.")
