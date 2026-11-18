"""Download everything infer.py needs into --out (default ./weights):

  base_model/  MiniMax-H3 @ 42ed227 (Ref2VA transformer `transformer_ref`, H3 VAE, Qwen3-VL text encoder, tokenizer,
               processor, scheduler, pipeline indices): the unmodified copy in this release
               (--base-from-minimax: the official MiniMaxAI/MiniMax-H3 at the same revision instead)
  sharp/       lms_curve.bin = the [1025, 8] FP32 adaln_t_table of Comfy-Org/MiniMax-H3 @ e5eb578 (base-derived; only
               that tensor is read from the checkpoint, by HTTP byte range)
  lora/, assets/, latent_upsampler/, vae_fast/, config.json   Efficient-Large-Model/Open-H3-Regenerate-2K-IC (IC LoRA,
               HyperFlow 8-step LoRA, LMS Sharp LoRA, silence latent, latent upscaler, fast H3 VAE decoder)

    pip install -U huggingface_hub
    python download.py --out ./weights                   # ~150 GB, mostly MiniMax-H3 (transformer_ref + text encoder)
"""
import argparse
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

BASE = 'MiniMaxAI/MiniMax-H3'
BASE_REVISION = '42ed227ee7df40d41602854ae760620d6eb651fe'
BASE_PATTERNS = ['transformer_ref/*', 'vae/*', 'text_encoder/*', 'tokenizer/*', 'processor/*', 'scheduler/*',
                 'model_index.json', 'modular_model_index.json', 'LICENSE', 'README.md']
THIS = 'Efficient-Large-Model/Open-H3-Regenerate-2K-IC'
CURVE = dict(repo='Comfy-Org/MiniMax-H3', revision='e5eb578a89295337b8ff433a035929ce0279e0b6',
             file='diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors', tensor='adaln_t_table',
             sha256='c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7')


def read_range(url, start, length, token):
    headers = {'Range': f'bytes={start}-{start + length - 1}', 'Accept-Encoding': 'identity'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if response.status != 206:
            raise RuntimeError('server ignored the Range request; refusing a full checkpoint download')
        data = response.read()
    if len(data) != length:
        raise RuntimeError('truncated range response')
    return data


def fetch_curve(out):
    """Only the adaln_t_table tensor of the checkpoint, by byte range (safetensors header, then its bytes)."""
    from huggingface_hub import get_token, hf_hub_url
    url, token = hf_hub_url(CURVE['repo'], CURVE['file'], revision=CURVE['revision']), get_token()
    header_length, = struct.unpack('<Q', read_range(url, 0, 8, token))
    tensor = json.loads(read_range(url, 8, header_length, token))[CURVE['tensor']]
    assert tensor['dtype'] == 'F32' and tensor['shape'] == [1025, 8], tensor
    start, end = tensor['data_offsets']
    data = read_range(url, 8 + header_length + start, end - start, token)
    if hashlib.sha256(data).hexdigest() != CURVE['sha256']:
        raise ValueError('adaln_t_table SHA256 mismatch')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('weights'))
    ap.add_argument('--repo', default=THIS, help='repository with the IC weights')
    ap.add_argument('--revision', default='main', help='revision of --repo')
    ap.add_argument('--base-from-minimax', action='store_true',
                    help='fetch base_model/ from MiniMaxAI/MiniMax-H3 at the pinned revision instead of --repo')
    a = ap.parse_args()
    snapshot_download(a.repo, revision=a.revision, local_dir=a.out,
                      allow_patterns=['config.json', 'lora/*', 'assets/*', 'latent_upsampler/*', 'vae_fast/*'])
    if a.base_from_minimax:
        snapshot_download(BASE, revision=BASE_REVISION, local_dir=a.out / 'base_model', allow_patterns=BASE_PATTERNS)
    else:
        snapshot_download(a.repo, revision=a.revision, local_dir=a.out,
                          allow_patterns=[f'base_model/{p}' for p in BASE_PATTERNS])
    fetch_curve(a.out / 'sharp' / 'lms_curve.bin')
    print('weights ready in', a.out.resolve())


if __name__ == '__main__':
    main()
