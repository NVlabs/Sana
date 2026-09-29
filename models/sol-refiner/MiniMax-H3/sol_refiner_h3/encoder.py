"""Encoder-only view of Diffusers' LTX video autoencoder."""

from diffusers import AutoencoderKLLTX2Video


class LTX2Encoder(AutoencoderKLLTX2Video):
    """Reuse the upstream encoder and serialization without an unused decoder.

    LTX-2.5 stores its diffusion decoder as a separate component. This class
    inherits the upstream encode, tiling, configuration and loading APIs.
    """

    @classmethod
    def from_config(cls, config=None, **kwargs):
        result = super().from_config(config, **kwargs)
        model = result[0] if isinstance(result, tuple) else result
        model.decoder = None
        return result

    def decode(self, *args, **kwargs):
        raise RuntimeError(
            "Decode with the pipeline's LTX-2.5 diffusion_decoder component"
        )
