"""Fetch everything infer.py needs into one weights directory and verify it.

    weights/
      lora/                 merged LoRA + Sharp curve branches (this release) + adaln_t_table.bin (fetched, see below)
      latent_upsampler/     author 3D-conv latent upscaler
      base_model/           MiniMax-H3 Diffusers components, unmodified copy of revision 6818f6c
      hyperflow/            optional HyperFlow 8-step adapter, for prompt -> 768P generation (--hyperflow)

    python download.py --weights ./weights [--hyperflow]
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request

RELEASE_REPO = "Efficient-Large-Model/Open-H3-Regenerate-2K"
BASE = {"repo": "MiniMaxAI/MiniMax-H3", "revision": "6818f6c32d12b210915e44ad56a4228c2608f160"}
BASE_PATTERNS = ["LICENSE", "README.md", "model_index.json", "modular_model_index.json", "scheduler/*",
                 "audio_scheduler/*", "processor/*", "tokenizer/*", "text_encoder/*", "transformer/*", "vae/*",
                 "audio_vae/*"]
CURVE = {"repo": "Comfy-Org/MiniMax-H3", "revision": "e5eb578a89295337b8ff433a035929ce0279e0b6",
         "file": "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors", "tensor": "adaln_t_table",
         "bytes": 32800, "sha256": "c02a6c11888297688c1e6278185ea1f947023acfc69f9003bbcdcec9a229a8e7"}
HYPERFLOW = {"repo": "videorebirth/hyperflow", "revision": "5bde94d7de3e78dafa2abc43ee2f280a1cb244e8",
             "file": "minimax_h3_hyperflow_8step_v1.0.safetensors",
             "sha256": "9297f4505bfdef59c3014d11274411809c19b0abfe26161cab2b425a696df447"}
EXPECTED = {
    "lora/open_h3_regenerate_2k_lora.safetensors": "8e36a0912ab72f7f3b00f41e3c7c97c7dfdec21e66c88ab22f08b9b0c681bb04",
    "lora/sharp_curve_branches.safetensors": "26ff1c4ea50fc5b0cea252912b67ec308a65d534cef026ef56f415ae92102e4e",
    "latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors":
        "4f57821f5837f32f7142b67d815606dbd7550f194e5c769f7d6c3f83b146a5e6",
    # fast H3 VAE decoder (non-causal, 24 layers; StreamRender-H3), used by --vae fast
    "vae_fast/fast_h3_vae_noncausal_24l.safetensors": "8658f1e85b5997c4ba716406a2f148bfa36cf97d4d5fcd402c7bd341f54e77a1",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_range(url, start, length, token=None):
    headers = {"Range": f"bytes={start}-{start + length - 1}", "Accept-Encoding": "identity"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        if response.status != 206:
            raise RuntimeError("server ignored the Range request; refusing a full checkpoint download")
        data = response.read(length + 1)
    if len(data) != length:
        raise RuntimeError("truncated range response")
    return data


def fetch_curve(destination, token=None):
    """Sharp's 1025x8 adaln_t_table is base-derived and not redistributed: read just that tensor by byte range."""
    url = f"https://huggingface.co/{CURVE['repo']}/resolve/{CURVE['revision']}/{CURVE['file']}"
    header_length, = struct.unpack("<Q", read_range(url, 0, 8, token))
    header = json.loads(read_range(url, 8, header_length, token))
    tensor = header[CURVE["tensor"]]
    if tensor["dtype"] != "F32" or tensor["shape"] != [1025, 8]:
        raise ValueError("unexpected adaln_t_table format")
    start, end = tensor["data_offsets"]
    data = read_range(url, 8 + header_length + start, end - start, token)
    if len(data) != CURVE["bytes"] or hashlib.sha256(data).hexdigest() != CURVE["sha256"]:
        raise ValueError("adaln_t_table SHA256 mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def localize_text_encoder(base):
    """The official modular index points the text encoder / tokenizer / processor at the Hub repo; point them at the
    local copy instead (modified file)."""
    index_path = base / "modular_model_index.json"
    index = json.loads(index_path.read_text())
    for name, entry in index.items():
        if isinstance(entry, list) and len(entry) == 3 and isinstance(entry[2], dict) \
                and entry[2].get("pretrained_model_name_or_path") == BASE["repo"]:
            entry[2]["pretrained_model_name_or_path"] = str(base.resolve())
            entry[2]["revision"] = None
    index["_modified_by"] = "Open-H3-Regenerate-2K download.py: components resolved from this local folder"
    index_path.write_text(json.dumps(index, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--repo", default=RELEASE_REPO, help="Hugging Face repo of this release")
    parser.add_argument("--hyperflow", action="store_true", help="also fetch the optional HyperFlow adapter")
    parser.add_argument("--skip-base", action="store_true", help="base_model/ is already in place")
    parser.add_argument("--base-from-minimax", action="store_true",
                        help="fetch base_model/ from MiniMaxAI/MiniMax-H3 at the pinned revision instead of this release")
    args = parser.parse_args()
    from huggingface_hub import get_token, hf_hub_download, snapshot_download

    weights = args.weights
    snapshot_download(args.repo, local_dir=weights, allow_patterns=["lora/*", "latent_upsampler/*", "vae_fast/*", "*.md", "*.json"])
    for relative, digest in EXPECTED.items():
        if sha256(weights / relative) != digest:
            raise ValueError(f"SHA256 mismatch: {relative}")
    table = weights / "lora/adaln_t_table.bin"
    if not table.exists() or sha256(table) != CURVE["sha256"]:
        fetch_curve(table, get_token())
    if not args.skip_base:
        base = weights / "base_model"
        if args.base_from_minimax:
            snapshot_download(BASE["repo"], revision=BASE["revision"], local_dir=base, allow_patterns=BASE_PATTERNS)
        else:
            snapshot_download(args.repo, local_dir=weights, allow_patterns=[f"base_model/{p}" for p in BASE_PATTERNS])
        localize_text_encoder(base)
    if args.hyperflow:
        cached = hf_hub_download(HYPERFLOW["repo"], HYPERFLOW["file"], revision=HYPERFLOW["revision"],
                                 local_dir=weights / "hyperflow")
        if sha256(cached) != HYPERFLOW["sha256"]:
            raise ValueError("HyperFlow SHA256 mismatch")
    print(f"READY {weights}")


if __name__ == "__main__":
    main()
