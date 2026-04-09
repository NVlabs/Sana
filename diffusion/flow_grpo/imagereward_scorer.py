import os

import torch

# ---------------------------------------------------------------------------
# Compatibility shim: ImageReward's vendored BLIP imports three helpers that
# were removed from transformers >= 5.0.  Inject them back before the import.
# ---------------------------------------------------------------------------
import transformers.modeling_utils as _mu
from PIL import Image

if not hasattr(_mu, "apply_chunking_to_forward"):

    def _apply_chunking_to_forward(forward_fn, chunk_size, chunk_dim, *input_tensors):
        if chunk_size > 0:
            num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size
            input_chunks = tuple(t.chunk(num_chunks, dim=chunk_dim) for t in input_tensors)
            output_chunks = tuple(forward_fn(*chunk) for chunk in zip(*input_chunks))
            return torch.cat(output_chunks, dim=chunk_dim)
        return forward_fn(*input_tensors)

    _mu.apply_chunking_to_forward = _apply_chunking_to_forward

if not hasattr(_mu, "find_pruneable_heads_and_indices"):

    def _find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
        mask = torch.ones(n_heads, head_size)
        heads = set(heads) - already_pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in already_pruned_heads)
            mask[head] = 0
        return heads, mask.view(-1).contiguous().eq(1).nonzero().squeeze()

    _mu.find_pruneable_heads_and_indices = _find_pruneable_heads_and_indices

if not hasattr(_mu, "prune_linear_layer"):

    def _prune_linear_layer(layer, index, dim=0):
        W = layer.weight.index_select(dim, index.to(layer.weight.device)).clone().detach()
        if layer.bias is not None:
            if dim == 1:
                b = layer.bias.clone().detach()
            else:
                b = layer.bias[index].clone().detach()
        new_size = list(layer.weight.size())
        new_size[dim] = len(index)
        new_layer = torch.nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
        new_layer.weight.requires_grad_(False)
        new_layer.weight.copy_(W)
        new_layer.weight.requires_grad_(True)
        if layer.bias is not None:
            new_layer.bias.requires_grad_(False)
            new_layer.bias.copy_(b)
            new_layer.bias.requires_grad_(True)
        return new_layer

    _mu.prune_linear_layer = _prune_linear_layer
# ---------------------------------------------------------------------------

import ImageReward as RM

# Replace BLIP's init_tokenizer to avoid additional_special_tokens_ids
# (removed in transformers 5.x) and patch all_tied_weights_keys.
import ImageReward.models.BLIP.blip_pretrain as _blip_pt


def _patched_init_tokenizer():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
    tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
    return tokenizer


_blip_pt.init_tokenizer = _patched_init_tokenizer

# Patch PreTrainedModel to add all_tied_weights_keys if missing
# (renamed to _tied_weights_keys in transformers 5.x)
from transformers import PreTrainedModel as _PTM

if not hasattr(_PTM, "all_tied_weights_keys"):
    _PTM.all_tied_weights_keys = property(lambda self: getattr(self, "_tied_weights_keys", []))
# ---------------------------------------------------------------------------


class ImageRewardScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.model = (
            RM.load(
                "ImageReward-v1.0",
                device=device,
                download_root=os.path.join(os.environ.get("HF_HOME", "~/.cache/"), "ImageReward"),
            )
            .eval()
            .to(dtype=dtype)
        )
        self.model.requires_grad_(False)

    @torch.no_grad()
    def __call__(self, prompts, images):
        _, rewards = self.model.inference_rank(prompts, images)
        rewards = torch.diagonal(torch.Tensor(rewards).to(self.device).reshape(len(prompts), len(prompts)), 0)
        return rewards.contiguous()


# Usage example
def main():
    scorer = ImageRewardScorer(device="cuda", dtype=torch.float32)

    images = [
        "test_cases/nasa.jpg",
        "test_cases/hello world.jpg",
    ]
    pil_images = [Image.open(img) for img in images]
    prompts = [
        'An astronaut\'s glove floating in zero-g with "NASA 2049" on the wrist',
        'New York Skyline with "Hello World" written with fireworks on the sky',
    ]
    print(scorer(prompts, pil_images))


if __name__ == "__main__":
    main()
