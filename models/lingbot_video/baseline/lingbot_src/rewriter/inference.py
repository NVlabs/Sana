import argparse
import contextlib
import json
import os

from rewriter_core import Rewriter, save_caption

try:
    import torch
except ImportError:
    torch = None

try:
    from peft import PeftModel
except ImportError:
    PeftModel = None

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None

BASE = os.environ.get("REWRITER_BASE_MODEL", "")     # Qwen3.6-27B VLM
ADAPTER = os.environ.get("REWRITER_ADAPTER", "")     # step2 LoRA adapter (peft format)


def _require_backend_modules(backend, modules):
    missing = [name for name, module in modules.items() if module is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ImportError(f"{backend} backend requires missing package(s): {missing_str}.")


class TransformersBackend:
    """Local base model + PeftModel adapter. step1 = base (disable_adapter), step2 = base + LoRA."""

    def __init__(self, base=BASE, adapter=ADAPTER, device="auto", max_new_tokens=6144):
        _require_backend_modules(
            "transformers",
            {
                "torch": torch,
                "transformers": AutoModelForImageTextToText,
                "peft": PeftModel,
            },
        )
        if not base or not adapter:
            raise ValueError("Set base and adapter paths (--base/--adapter or "
                             "REWRITER_BASE_MODEL/REWRITER_ADAPTER).")
        self.processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, text, image, use_lora):
        content = ([{"type": "image", "image": image}] if image is not None else []) \
            + [{"type": "text", "text": text}]
        messages = [{"role": "user", "content": content}]
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.processor(
            text=[chat], images=([image] if image is not None else None), return_tensors="pt"
        ).to(self.model.device)
        # step1 (base): disable LoRA; step2 (map): keep LoRA active
        adapter_ctx = contextlib.nullcontext() if use_lora else self.model.disable_adapter()
        with torch.no_grad(), adapter_ctx:
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]


def make_backend(name="transformers", base=BASE, adapter=ADAPTER):
    if name != "transformers":
        raise ValueError(f"unknown backend: {name!r}; only 'transformers' is supported")
    return TransformersBackend(base, adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="transformers", choices=["transformers"],
                    help="rewriter backend; only the in-process transformers backend is bundled")
    ap.add_argument("--mode", required=True, choices=["t2v", "ti2v", "t2i"])
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--first-frame", default=None, help="TI2V first frame: local path or URL")
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--output", default=None, help="save the JSON caption to this path: {caption, duration}")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--adapter", default=ADAPTER)
    args = ap.parse_args()

    rw = Rewriter(make_backend(args.backend, args.base, args.adapter))
    out = rw.rewrite(args.prompt, args.mode, args.first_frame, args.duration)
    print("\n========== DETAILED (step1) ==========\n" + out["detailed"])
    print("\n========== JSON (step2) ==========")
    print(json.dumps(out["json"], ensure_ascii=False, indent=2) if out["json"] else out["json_raw"])
    if args.output:
        save_caption(out, args.duration, args.output)
        print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
