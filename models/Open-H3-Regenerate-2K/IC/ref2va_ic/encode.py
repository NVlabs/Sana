"""H3 VAE and Qwen3-VL text encoding of the draft, and VAE decoding of the result.

VAE: official H3 VAE in FP32 with 256/64 tiling, ImageNet-normalized pixels, posterior sample (seeded), then
(z - latents_mean) / latents_std.  Prompt: TASK + "\n\nScene description: " + caption, with the draft frames (uint8,
24 fps, at the LQ canvas) as <Video 1>; `drop_prefix_length` marks where the text starts (an empty caption is valid input).
"""
import torch

TASK = ('Enhance <Video 1> into a high-quality version of the same video. '
        'Preserve the scene, objects, camera motion, timing, and composition of every frame. '
        'Restore fine details and remove degradation. Do not continue or change the scene.')

PIXEL_MEAN = (.485, .456, .406)
PIXEL_STD = (.229, .224, .225)


def load_vae(model_dir, device='cuda'):
    from diffusers import AutoencoderKLMiniMaxH3
    vae = AutoencoderKLMiniMaxH3.from_pretrained(str(model_dir), subfolder='vae', torch_dtype=torch.float32,
                                                 local_files_only=True).to(device).eval().requires_grad_(False)
    vae.enable_tiling(tile_sample_min_height=256, tile_sample_min_width=256,
                      tile_sample_min_overlap_height=64, tile_sample_min_overlap_width=64)
    # the FP32 VAE keeps PyTorch SDPA even after the transformer switched to FlashAttention-3 (bf16 / fp16 only)
    vae.set_attention_backend('native')
    return vae


def _stats(vae, device):
    mean = torch.tensor(vae.config.latents_mean, device=device).view(1, 24, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=device).view(1, 24, 1, 1, 1)
    pm = torch.tensor(PIXEL_MEAN, device=device).view(1, 3, 1, 1, 1)
    ps = torch.tensor(PIXEL_STD, device=device).view(1, 3, 1, 1, 1)
    return mean, std, pm, ps


@torch.inference_mode()
def vae_encode(vae, pixels, seed):
    """pixels: float [T, 3, h, w] in [0, 1] -> normalized latent [24, T', h/16, w/16] (bf16, CPU)."""
    device = next(vae.parameters()).device
    mean, std, pm, ps = _stats(vae, device)
    x = (pixels.to(device).permute(1, 0, 2, 3)[None] - pm) / ps
    z = vae.encode(x, return_dict=False)[0].sample(generator=torch.Generator().manual_seed(seed))
    latent = ((z - mean) / std)[0].bfloat16().cpu()
    assert torch.isfinite(latent).all()
    return latent


@torch.inference_mode()
def vae_decode(vae, latent):
    """normalized latent [24, T', H/16, W/16] -> uint8 frames [T, H, W, 3] (CPU)."""
    device = next(vae.parameters()).device
    mean, std, pm, ps = _stats(vae, device)
    pixels = vae.decode(latent[None].to(device).float() * std + mean, return_dict=False)[0][0]   # [3, T, H, W]
    pm, ps = pm[0], ps[0]
    return torch.cat([((pixels[:, t:t + 8] * ps + pm).clamp(0, 1) * 255).round().byte().permute(1, 2, 3, 0).cpu()
                      for t in range(0, pixels.shape[1], 8)]).numpy()


def load_text_encoder(model_dir, device='cuda'):
    from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep
    encoder = MiniMaxH3Ref2VATextEncoderStep().init_pipeline(str(model_dir))
    # the official modular_model_index.json names the Hub repo; load the components from the local copy instead
    encoder.load_components(dtype=torch.bfloat16, pretrained_model_name_or_path=str(model_dir))
    encoder.text_encoder.to(device)
    return encoder


@torch.inference_mode()
def encode_prompt(encoder, pixels, caption):
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference
    frames = (pixels * 255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).contiguous().cpu().numpy()
    prompt = TASK + '\n\nScene description: ' + (caption or '')
    out = encoder(prompt=prompt, normalized_references=[MiniMaxH3VideoReference(frames=frames, fps=24.)],
                  output=['prompt_embeds', 'text_token_tags'])
    emb = out['prompt_embeds'][0].detach().cpu().bfloat16()
    tags = out['text_token_tags'].flatten().cpu()
    prefix = len(emb) - len(encoder.tokenizer.encode(prompt, add_special_tokens=False))
    assert 0 < prefix < len(tags) and emb.shape == (len(tags), 5120) and torch.isfinite(emb).all()
    assert (tags[:prefix] == 0).any() and (tags[prefix:] == 1).all()
    return dict(prompt_embeds=emb, text_token_tags=tags, drop_prefix_length=prefix)
