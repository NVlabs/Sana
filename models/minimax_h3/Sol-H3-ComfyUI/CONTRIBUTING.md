# Maintaining Sol-H3 in ComfyUI

Keep node IDs and workflow inputs compatible. Put ComfyUI glue in
`sol_h3_engine/comfy_nodes.py`, process coordination in `bridge.py` and
`worker.py`, and UI changes in `web/`. The vendored runtime snapshots retain
their source receipts; explain numerical changes separately from UI changes.

Run the CPU checks from this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
python3 scripts/configure.py --help
python3 scripts/run_local.py --help
```

These checks use fake pipelines and import/path checks. Model or GPU-memory
changes also need a real run on an available GPU, with the prompt, seed,
precision, decoder, output and timing boundary recorded in `docs/validation.md`.
Keep loading and warmup separate from formal request time.

Keep host configuration, credentials, model weights, environments and raw
machine logs out of commits. Use `config.example.json` for portable examples.
The original upstream notices and source receipts belong with the code.
