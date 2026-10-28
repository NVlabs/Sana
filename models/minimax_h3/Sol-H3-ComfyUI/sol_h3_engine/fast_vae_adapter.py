"""Adapter for the Sol-H3-VAE decoder (Efficient-Large-Model/Sol-H3-VAE) in the FastH3 runtimes."""
import json
import os
import sys
from pathlib import Path


def settings():
    value = os.environ.get("SOL_H3_FAST_VAE")
    return json.loads(value) if value else None


class Decoder:
    def __init__(self, config):
        # Sol-H3-VAE ships its own code; no checkpoint/code is bundled here.
        sys.path.insert(0, config["code"])
        import fa4_sm120
        from sol_h3_vae import SolH3VAEDecoder
        fa4_sm120.install()
        self.decoder = SolH3VAEDecoder(
            config["vae_config"], config["weights"], engine="fast", max_engines=1)

    def decode(self, batch, args):
        import torch
        from fastvideo.pipelines.basic.minimax_h3.packing import h3_dit_patch_size, unpatchify_video_tokens
        from fastvideo.pipelines.basic.minimax_h3.stages.minimax_h3_latent_preparation import MINIMAX_H3_LAYOUT_KEY
        layout = batch.extra[MINIMAX_H3_LAYOUT_KEY]
        _, channels, frames, height, width = batch.raw_latent_shape
        latent = unpatchify_video_tokens(batch.latents[layout.num_condition_video_rows:],
                                        frames, height, width, channels, h3_dit_patch_size(args))
        # The student checkpoint already contains latent mean/std. Do not apply
        # native VAE denormalize_latents before passing this normalized tensor.
        with torch.no_grad():
            from sol_h3_vae.fast import NonCausalFast3
            engine = None
            try:
                self.decoder.model.to("cuda")
                # A full 768p decoder approaches 30 GiB. Its buffers must be
                # released before the next Qwen/DiT request on a 32 GiB GPU.
                # Reuse the delivered eager kernel path: a graph that will be
                # discarded after this decode would add capture work each time.
                engine = NonCausalFast3(self.decoder.model, frames, height, width, cuda_graph=False)
                pixels = engine.decode(latent.transpose(1, 2).float().contiguous()).cpu()
                # Convert on CPU to avoid additional full-video GPU buffers.
                batch.output = self.decoder.processor.revert_tensor(pixels).clamp_(0, 1)
            finally:
                del engine
                self.decoder.model.to("cpu")
        return batch
