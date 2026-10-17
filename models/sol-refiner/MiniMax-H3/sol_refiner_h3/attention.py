"""Use an installed NATTEN backend with the upstream decoder processor."""

from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
    LTX2VideoVaeNeighborhoodNattenProcessor,
)


class LocalNattenProcessor(LTX2VideoVaeNeighborhoodNattenProcessor):
    """Keep the upstream attention calculation while loading NATTEN locally."""

    def __init__(self, backend: str | None = None):
        try:
            from natten.functional import na3d
        except ImportError as exc:
            raise ImportError(
                "Install the NATTEN wheel matching your PyTorch and CUDA versions"
            ) from exc
        self._na3d = na3d
        self.backend = backend
