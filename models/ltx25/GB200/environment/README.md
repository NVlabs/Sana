# LTX-2.5 GB200 environment

This hardware runtime defaults to the repository-local interpreter at
`<repo>/.venv/bin/python`. It does not reference another checkout or virtual
environment beside this repository.

`LTX-2/` is a vendored uv workspace containing the exact lock file and the
`ltx-core`, `ltx-pipelines`, `ltx-kernels`, and `ltx-trainer` package sources
used to create the validated Python 3.13 / PyTorch 2.11 cu130 environment. The
checked-in `ltx-kernels` tree also carries the validated SM100 native artifacts
for the local GB200 installation; `setup_env.sh` can rebuild them from source.

To rebuild the local environment, submit `setup_env.sh` to a Slurm compute
node with explicit CPU resources. The script keeps uv and temporary caches
inside `<repo>/.cache/` and creates `<repo>/.venv/`.

An already prepared Diffusers or SGLang Python environment can be used instead
without changing the configs:

```bash
python3 scripts/run.py models/ltx25/GB200/fullopt.toml \
  --set PYTHON_BIN=/absolute/path/to/python
```

The selected environment must provide the locked third-party dependencies.
The LTX runtime source and GB200 `ltx-kernels` source are loaded from this
repository through `PYTHONPATH`.
