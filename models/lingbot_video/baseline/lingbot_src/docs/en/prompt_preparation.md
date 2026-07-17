# Prompt Preparation

LingBot-Video DiT inference is designed to consume structured JSON prompts, not
raw casual natural-language prompts. The inference prompt flow is:

1. Rewrite the user's plain prompt into `prompt.json`.
2. Optionally run auto negative to create `negative.json`.

## Structured JSON Prompt

The rewriter saves:

```json
{
  "caption": {
    "...": "structured JSON caption"
  },
  "duration": 5
}
```

See `assets/cases/<mode>/example_*/prompt.json` for complete real examples
(`mode` is `t2i`, `t2v`, or `ti2v`).

For video tasks, if `prompt.json` contains `duration`, the runner derives
`num_frames` from `duration` and `fps`. You can still pass `--num_frames`
explicitly to override it.

**⚠️ Note: the structured caption carries per-action timestamps such as
`[0.0s - 5.0s]` under `prominent_elements[].actions[]`. `--duration` must match
the total span those timestamps cover (e.g. if the latest action reaches `5.0s`,
set `--duration 5`).**

## Prompt Rewriter

The prompt rewriter turns a plain user prompt into a structured JSON prompt in
two stages:

```text
user prompt (+ first frame, TI2V only)
   step1 EXPAND  (base VLM)        -> intermediate detailed caption
   step2 MAP     (base VLM + LoRA) -> structured JSON caption
```

The two stages share the same base VLM. Step 2 attaches the rewriter LoRA. TI2V
feeds the first-frame image to both stages.

**💡 Tip: Step 1 EXPAND does not rely on the rewriter LoRA, so you can swap in a stronger VLM to get a richer and more accurate intermediate detailed caption.**

Set the rewriter weights:

```bash
export REWRITER_BASE_MODEL="<path_to_rewriter_base_vlm>"
export REWRITER_ADAPTER="<path_to_rewriter_lora>"
```

The bundled rewriter ships only the `transformers` backend (single-process,
reference implementation); it is covered by `requirements.txt` and needs no extra
install.

> **💡 Production deployment**: the `transformers` backend runs one request at a
> time and has low throughput. For higher throughput, deploy the VLM as a
> standalone inference server and call it through an OpenAI-compatible API instead
> of the in-process backend. Preserve the two-stage semantics: step 1 must use the
> base VLM without the rewriter LoRA, while step 2 must use the same base VLM with
> the rewriter LoRA enabled. This can be implemented with two endpoints, or with
> one server that can select the adapter per request. See the official serving
> docs: [vLLM](https://docs.vllm.ai), [SGLang](https://docs.sglang.ai).

### T2V

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2v \
  --prompt "<plain_user_prompt>" \
  --duration 5 \
  --output prompt.json
```

### TI2V

Use the same first frame here and later in DiT inference:

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode ti2v \
  --prompt "<plain_user_prompt>" \
  --first-frame "<first_frame.png>" \
  --duration 5 \
  --output prompt.json
```

### T2I

```bash
python rewriter/inference.py \
  --backend transformers \
  --mode t2i \
  --prompt "<plain_user_prompt>" \
  --output prompt.json
```

## Auto Negative Prompt

Auto negative reads the structured caption and deletes terms from the default
negative prompt list that conflict with the intended content. It is delete-only per-sample pruning:
it removes terms from the default negative and does not add case-specific new
terms.

**💡 Tip: In most cases the built-in default negative prompt is enough; if you are not satisfied with the result, run auto negative to fine-tune it per sample.**

```bash
python rewriter/auto_negative.py \
  --backend transformers \
  --mode t2v \
  --caption prompt.json \
  --output negative.json
```

Pass the output file to DiT inference:

```bash
--negative_prompt_json negative.json
```

If auto negative is not enabled, omit `--negative_prompt_json`. The runner then
uses the built-in default negative prompt:

- `t2v` and `ti2v` use the video default negative prompt.
- `t2i` uses the image default negative prompt.
