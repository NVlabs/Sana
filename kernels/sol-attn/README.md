# Sol-Attn Hub packaging

This directory contains the Hugging Face Kernel Hub metadata, tests, and
publishing workflow inputs for Sol-Attn. The maintained implementation remains
in `techniques/sparse_backends/sol_attn`; `prepare.py` copies that exact tree
into the `torch-ext/sol_attn` layout expected by `kernel-builder`.

Prepare a local source tree with:

```bash
python kernels/sol-attn/prepare.py --output /tmp/sol-attn-kernel
kernel-builder check-config /tmp/sol-attn-kernel
```

The `Publish Sol-Attn kernel` GitHub Actions workflow builds and uploads
version 1 to `Efficient-Large-Model/sol-attn`, then tests the uploaded package
on A100, H200, and RTX PRO 6000 GPU jobs.

The repository must provide:

- an `HF_TOKEN` Actions secret with `job.write` and permission to create or
  update kernels in `Efficient-Large-Model`;
- optionally, a `SOL_ATTN_HF_JOBS_NAMESPACE` Actions variable when Jobs should
  consume quota from a namespace other than `Efficient-Large-Model`.

The workflow is manual for the first release. Trigger it from the Actions tab
after the Hub organization permissions are confirmed.
