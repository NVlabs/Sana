"""Reward model scorers and multi-score dispatcher for Sol-RL training."""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.transforms import Compose, InterpolationMode, Normalize

CKPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reward_ckpts")

# ---------------------------------------------------------------------------
# HPSv2 transform helpers
# ---------------------------------------------------------------------------

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


class _ResizeMaxSize(nn.Module):
    def __init__(self, max_size, interpolation=InterpolationMode.BICUBIC, fn="max", fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def forward(self, img):
        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = TF.resize(img, new_size, self.interpolation)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = TF.pad(img, padding=[pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2], fill=self.fill)
        return img


class _MaskAwareNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.normalize = Normalize(mean=mean, std=std)

    def forward(self, tensor):
        if tensor.shape[1] == 4:
            normalized_parts = []
            for i in range(tensor.shape[0]):
                img_slice = tensor[i]
                normalized_rgb = self.normalize(img_slice[:3])
                alpha_channel = img_slice[3:]
                normalized_parts.append(torch.cat([normalized_rgb, alpha_channel], dim=0))
            return torch.stack(normalized_parts, dim=0)
        else:
            return self.normalize(tensor)


def _hpsv2_image_transform(image_size, mean=None, std=None, fill_color=0):
    mean = mean or OPENAI_DATASET_MEAN
    std = std or OPENAI_DATASET_STD
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    return Compose(
        [
            _ResizeMaxSize(image_size, fill=fill_color),
            _MaskAwareNormalize(mean=mean, std=std),
        ]
    )


# ---------------------------------------------------------------------------
# CLIP transform helper
# ---------------------------------------------------------------------------


def _get_clip_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")


def _get_clip_image_transform(processor):
    config = processor.to_dict()
    resize = T.Resize(_get_clip_size(config.get("size"))) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(_get_clip_size(config.get("crop_size"))) if config.get("do_center_crop") else nn.Identity()
    normalise = (
        T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()
    )
    return T.Compose([resize, crop, normalise])


# ---------------------------------------------------------------------------
# ImageReward compatibility shim (transformers >= 5.0)
# ---------------------------------------------------------------------------

_imagereward_patched = False


def _patch_imagereward_compat():
    global _imagereward_patched
    if _imagereward_patched:
        return
    _imagereward_patched = True

    import transformers.modeling_utils as _mu

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
                b = layer.bias.clone().detach() if dim == 1 else layer.bias[index].clone().detach()
            new_size = list(layer.weight.size())
            new_size[dim] = len(index)
            new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
            new_layer.weight.requires_grad_(False)
            new_layer.weight.copy_(W)
            new_layer.weight.requires_grad_(True)
            if layer.bias is not None:
                new_layer.bias.requires_grad_(False)
                new_layer.bias.copy_(b)
                new_layer.bias.requires_grad_(True)
            return new_layer

        _mu.prune_linear_layer = _prune_linear_layer

    import ImageReward.models.BLIP.blip_pretrain as _blip_pt

    def _patched_init_tokenizer():
        from transformers import BertTokenizer

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        tokenizer.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tokenizer.enc_token_id = tokenizer.convert_tokens_to_ids("[ENC]")
        return tokenizer

    _blip_pt.init_tokenizer = _patched_init_tokenizer

    from transformers import PreTrainedModel as _PTM

    if not hasattr(_PTM, "all_tied_weights_keys"):
        _PTM.all_tied_weights_keys = property(lambda self: getattr(self, "_tied_weights_keys", []))


# =========================================================================
# Scorer classes (heavy deps lazy-loaded inside __init__)
# =========================================================================


class HPSv2Scorer(nn.Module):
    def __init__(self, dtype, device):
        super().__init__()
        from hpsv2.src.open_clip import create_model, get_tokenizer

        self.dtype = dtype
        self.device = device
        model = create_model(
            "ViT-H-14",
            os.path.join(CKPT_PATH, "open_clip_pytorch_model.bin"),
            precision="amp",
            device=device,
            jit=False,
            force_quick_gelu=False,
            force_custom_text=False,
            force_patch_dropout=False,
            force_image_size=None,
            pretrained_image=False,
            output_dict=True,
        )
        image_size = model.visual.image_size
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        self.preprocess_val = _hpsv2_image_transform(
            image_size,
            mean=getattr(model.visual, "image_mean", None),
            std=getattr(model.visual, "image_std", None),
        )
        self.model = model.to(device)
        checkpoint = torch.load(os.path.join(CKPT_PATH, "HPS_v2.1_compressed.pt"), map_location="cpu")
        self.model.load_state_dict(checkpoint["state_dict"])
        self.processor = get_tokenizer("ViT-H-14")
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts):
        image = self.preprocess_val(images.to(self.dtype).to(device=self.device, non_blocking=True))
        text = self.processor(prompts).to(device=self.device, non_blocking=True)
        outputs = self.model(image, text)
        image_features, text_features = outputs["image_features"], outputs["text_features"]
        logits_per_image = image_features @ text_features.T
        return torch.diagonal(logits_per_image, 0).contiguous()


class ClipScorer(nn.Module):
    def __init__(self, device):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.tform = _get_clip_image_transform(self.processor.image_processor)
        self.eval()

    @torch.no_grad()
    def __call__(self, pixels, prompts, return_img_embedding=False):
        texts = self.processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        pixels = self.tform(pixels).to(dtype=pixels.dtype, device=self.device)
        outputs = self.model(pixel_values=pixels, **texts)
        if return_img_embedding:
            return outputs.logits_per_image.diagonal() / 100, outputs.image_embeds
        return outputs.logits_per_image.diagonal() / 100


class PickScoreScorer(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1").eval().to(device, dtype=dtype)

    @staticmethod
    def _as_embedding(features):
        if torch.is_tensor(features):
            return features
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return features.pooler_output
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            hidden = features.last_hidden_state
            return hidden.mean(dim=1) if hidden.ndim == 3 else hidden
        raise TypeError(f"Unsupported model output type: {type(features)}")

    @torch.no_grad()
    def __call__(self, prompt, images):
        image_inputs = self.processor(images=images, padding=True, truncation=True, max_length=77, return_tensors="pt")
        image_inputs = {k: v.to(device=self.device) for k, v in image_inputs.items()}
        text_inputs = self.processor(text=prompt, padding=True, truncation=True, max_length=77, return_tensors="pt")
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}

        image_embs = self._as_embedding(self.model.get_image_features(**image_inputs))
        image_embs = image_embs / image_embs.norm(p=2, dim=-1, keepdim=True)
        text_embs = self._as_embedding(self.model.get_text_features(**text_inputs))
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)

        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ image_embs.T)
        return scores.diag() / 26


class ImageRewardScorer(nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        import ImageReward as RM

        _patch_imagereward_compat()
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


# =========================================================================
# Factory functions (image format normalisation) & dispatcher
# =========================================================================


def clip_score(device):
    scorer = ClipScorer(device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def hpsv2_score(device):
    scorer = HPSv2Scorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if not isinstance(images, torch.Tensor):
            images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
            images = torch.tensor(images, dtype=torch.uint8) / 255.0
        scores = scorer(images, prompts)
        return scores, {}

    return _fn


def pickscore_score(device):
    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def imagereward_score(device):
    scorer = ImageRewardScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        prompts = [prompt for prompt in prompts]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        "imagereward": imagereward_score,
        "pickscore": pickscore_score,
        "clipscore": clip_score,
        "hpsv2": hpsv2_score,
    }
    score_fns = {}
    for score_name, weight in score_dict.items():
        score_fns[score_name] = score_functions[score_name](device)

    def _fn(images, prompts, metadata, only_strict=True):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, rewards = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details["avg"] = total_scores
        return score_details, {}

    return _fn
