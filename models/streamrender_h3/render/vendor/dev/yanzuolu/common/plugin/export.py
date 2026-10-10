"""Export tensor subtrees from DCP checkpoints to safetensors or torch files.

Three entry points share one extraction core:
  * ``extract_state`` / ``export_state`` -- public functions, single process,
    no process group, no model instantiation.
  * ``ExportPlugin`` -- a persistence plugin that exports on the
    ``after_checkpoint_saved`` hook in a background thread.
  * ``__main__`` CLI -- ad-hoc export from a checkpoint dir or run dir.

``export_state`` takes a ``content`` mode selecting what to write:
  * ``full`` (default) -- the extracted subtree verbatim.
  * ``peft`` -- only the PEFT/LoRA tensors, remapped to the single-adapter
    ``get_peft_model_state_dict`` layout (what ``set_peft_model_state_dict``
    expects for a hot-start).
  * ``merged`` -- LoRA deltas folded into the base weights by pure state-dict
    arithmetic (no model instantiation), yielding a plain base state dict.

Dependency direction: export depends on common.persistence, never the reverse.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import traceback
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.metadata import TensorStorageMetadata

from ..logging import get_logger

logger = get_logger()


def extract_state(checkpoint_dir: str | Path, key: str) -> dict[str, torch.Tensor]:
    """Extract the tensor subtree stored under ``key`` from a DCP checkpoint.

    Reads the DCP metadata, pre-allocates empty tensors (from the recorded
    size/dtype) for every fully-qualified name under ``key.``, then issues a
    single ``dcp.load``. The keys handed to ``dcp.load`` are the exact FQNs from
    the archive; DCP's flatten is the identity on already-dotted keys, so they
    match the metadata directly. The returned dict has the ``key.`` prefix
    stripped, yielding the module's native FQNs (e.g. ``weight``, ``bias``).

    Single process, no process group, no model instantiation required.
    """
    reader = dcp.FileSystemReader(str(checkpoint_dir))
    metadata = reader.read_metadata()
    prefix = key + "."
    flat = {
        fqn: torch.empty(md.size, dtype=md.properties.dtype)
        for fqn, md in metadata.state_dict_metadata.items()
        if fqn.startswith(prefix) and isinstance(md, TensorStorageMetadata)
    }
    if not flat:
        top_prefixes = sorted({fqn.split(".")[0] for fqn in metadata.state_dict_metadata})
        raise KeyError(f"no tensors under '{key}' in {checkpoint_dir}, available prefixes: {top_prefixes}")
    # no_dist=True is essential: without it dcp.load becomes a collective on the
    # default process group when torch.distributed is initialized, deadlocking
    # rank0-only / background-thread exports (same reason dcp_to_torch_save
    # forces no_dist). This keeps extraction a purely local file read.
    dcp.load(flat, storage_reader=reader, no_dist=True)
    return {fqn[len(prefix):]: tensor for fqn, tensor in flat.items()}


def write_safetensors(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    try:
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError("safetensors is required for the 'safetensors' format; install it with `pip install safetensors`") from exc
    save_file(tensors, str(out_path))


def write_torch(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    torch.save(tensors, out_path)


def write_dcp(tensors: dict[str, torch.Tensor], out_path: Path) -> None:
    """Write a DCP checkpoint directory. ``out_path`` is a directory, not a file.

    Format, not speed. Extraction here is single-process by construction (see
    ``extract_state``), so one writer emits one data file, and a 64-rank load
    then pulls every rank's slice out of that single file -- the same scattered
    read that makes a monolithic safetensors slow. The layout that is actually
    fast (one file per writer, each rank streaming its own) needs as many
    writers, which is what the distributed converter under a project's
    ``tools/`` is for. This writer exists so the two agree on the on-disk format
    and a DCP export is loadable by the same plugin either way.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    dcp.save(tensors, checkpoint_id=str(out_path), no_dist=True)


WRITERS: dict[str, tuple] = {
    "safetensors": (write_safetensors, ".safetensors"),
    "torch": (write_torch, ".pth"),
    "dcp": (write_dcp, ".dcp"),
}

_SUFFIX_FORMATS = {
    ".safetensors": "safetensors",
    ".pth": "torch",
    ".pt": "torch",
    ".bin": "torch",
    ".dcp": "dcp",
}

# A PeftModel nests the wrapped model under LoraModel.model, so every archived
# adapter/base FQN carries this prefix.
_PEFT_PREFIX = "base_model.model."


def export_state(
    checkpoint_dir: str | Path,
    key: str,
    out_path: str | Path,
    format: str | None = None,
    content: str = "full",
    lora_alpha: float | None = None,
    alpha_pattern: dict | None = None,
) -> Path:
    """Extract ``key`` from the checkpoint and write it via the chosen writer.

    ``format`` defaults to ``None``, inferred from ``out_path``'s suffix
    (``.safetensors`` -> safetensors; ``.pth``/``.pt``/``.bin`` -> torch). An
    unrecognized suffix (with no explicit ``format``) or an invalid ``format``
    raises ``ValueError``.

    ``content`` selects what is written:
      * ``full`` -- the extracted subtree verbatim.
      * ``peft`` -- only the LoRA/modules_to_save tensors, remapped to the
        single-``default``-adapter ``get_peft_model_state_dict`` layout.
      * ``merged`` -- LoRA deltas folded into the base weights by state-dict
        arithmetic; requires ``lora_alpha`` (``alpha_pattern`` optional).

    The write is atomic for every writer: data goes to a sibling ``.tmp`` file
    first, then ``os.replace`` swaps it into place.
    """
    out_path = Path(out_path)
    if format is None:
        format = _SUFFIX_FORMATS.get(out_path.suffix)
    if format not in WRITERS:
        raise ValueError(
            f"cannot resolve export format for '{out_path.name}' (format={format!r}); "
            f"supported formats: {sorted(WRITERS)}"
        )
    writer, _ = WRITERS[format]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors = extract_state(checkpoint_dir, key)

    if content == "peft":
        # get_peft_model_state_dict layout: the ".default" adapter segment is
        # dropped from LoRA keys (lora_A.default.weight -> lora_A.weight) and
        # the whole "modules_to_save.default." segment is dropped from
        # modules_to_save keys -- set_peft_model_state_dict looks tensors up by
        # these stripped forms and rebuilds the full parameter names itself.
        # Base weights dropped.
        adapter_out = {}
        for fqn, tensor in tensors.items():
            if ".lora_A.default." in fqn or ".lora_B.default." in fqn:
                adapter_out[fqn.replace(".default", "")] = tensor
            elif ".modules_to_save.default." in fqn:
                adapter_out[fqn.replace("modules_to_save.default.", "")] = tensor
        assert adapter_out, f"no PEFT/LoRA tensors under '{key}' in {checkpoint_dir}; the model has no adapter attached"
        tensors = adapter_out
    elif content == "merged":
        assert lora_alpha is not None, "content='merged' requires lora_alpha"
        alpha_pattern = alpha_pattern or {}
        lora_suffix = ".lora_A.default.weight"
        mpaths = [fqn[: -len(lora_suffix)] for fqn in tensors if fqn.endswith(lora_suffix)]
        assert mpaths, f"no LoRA tensors under '{key}' in {checkpoint_dir}; nothing to merge"
        merged: dict[str, torch.Tensor] = {}
        consumed = set()
        for mpath in mpaths:
            a = tensors[mpath + ".lora_A.default.weight"]
            b = tensors[mpath + ".lora_B.default.weight"]
            base_key = mpath + ".base_layer.weight"
            base_weight = tensors[base_key]
            r = a.shape[0]  # per-module rank inferred from A -> compatible with rank_pattern
            native = mpath[len(_PEFT_PREFIX):] if mpath.startswith(_PEFT_PREFIX) else mpath
            alpha = next((v for pat, v in alpha_pattern.items() if native.endswith(pat)), lora_alpha)
            delta = (alpha / r) * (b.float() @ a.float())
            merged[native + ".weight"] = (base_weight.float() + delta).to(base_weight.dtype)
            consumed.update({mpath + ".lora_A.default.weight", mpath + ".lora_B.default.weight", base_key})
        for fqn, tensor in tensors.items():
            if fqn in consumed:
                continue
            native = fqn[len(_PEFT_PREFIX):] if fqn.startswith(_PEFT_PREFIX) else fqn
            if fqn.endswith(".base_layer.bias"):
                merged[native[: -len(".base_layer.bias")] + ".bias"] = tensor
            elif ".modules_to_save.default." in fqn:
                merged[native.replace("modules_to_save.default.", "")] = tensor  # overrides the original_module copy
            elif ".original_module." in fqn:
                continue
            else:
                merged[native] = tensor
        tensors = merged
    elif content != "full":
        raise ValueError(f"unknown content {content!r}; supported: full, peft, merged")

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    writer(tensors, tmp_path)
    os.replace(tmp_path, out_path)
    return out_path


class ExportPlugin:
    """Persistence plugin: export a tensor subtree to disk after each save.

    Async semantics: the checkpointer fires ``after_checkpoint_saved`` only once
    the checkpoint is fully on disk -- in async mode that is *after*
    ``future.result()``, so ``.metadata`` already exists and the data read back
    is guaranteed to be that step's complete snapshot. The export thread reads
    only from the settled on-disk directory, so it is safe to run in the
    background. The thread is non-daemon so the process waits for it on exit.

    Config: ``key`` (required dotted prefix), ``output_dir`` (default
    ``run_dir/exports``), ``keep_latest_k`` (default -1), ``format`` (default
    ``"safetensors"``), ``content`` (``full`` | ``peft`` | ``merged``, default
    ``full``). Output files are named ``step_{step:07d}_{leaf}_{content}{ext}``
    where ``ext`` is the writer's default extension; rotation globs the same
    ``{leaf}_{content}`` pattern so different-content exports never evict each
    other::

        persistence:
          plugins:
            - module: common.plugin.export
              class_name: ExportPlugin
              key: models.backbone
              content: merged        # full | peft | merged
              keep_latest_k: 2

    For ``content: merged`` the LoRA hyperparameters are read from the model's
    own adapter config at ``config.<key>.adapter`` (``lora_alpha`` and optional
    ``alpha_pattern``); a model with no adapter configured is a hard error.
    """

    def __init__(self, plugin_config):
        self.key = plugin_config["key"]
        self.output_dir = plugin_config.get("output_dir", None)
        self.keep_latest_k = plugin_config.get("keep_latest_k", -1)
        self.format = plugin_config.get("format", "safetensors")
        if self.format not in WRITERS:
            raise ValueError(f"unknown export format {self.format!r}; supported formats: {sorted(WRITERS)}")
        self.content = plugin_config.get("content", "full")
        if self.content not in ("full", "peft", "merged"):
            raise ValueError(f"unknown content {self.content!r}; supported: full, peft, merged")
        self.ext = WRITERS[self.format][1]
        self.thread: threading.Thread | None = None

    def after_checkpoint_saved(self, config=None, checkpoint_dir=None, step=None, **kwargs):
        if int(os.environ.get("RANK", "0")) != 0:
            return
        self.join()

        lora_alpha = None
        alpha_pattern = None
        if self.content == "merged":
            node = config
            for part in self.key.split("."):
                node = node[part]
            adapter = node.get("adapter", None)
            assert adapter is not None, f"[{self.key}] content='merged' but the model has no adapter configured"
            lora_alpha = adapter["lora_alpha"]
            alpha_pattern = adapter.get("alpha_pattern", {})

        out_dir = Path(self.output_dir or Path(config.persistence.run_dir) / "exports")
        out_dir.mkdir(parents=True, exist_ok=True)
        leaf = self.key.split(".")[-1]
        out_path = out_dir / f"step_{step:07d}_{leaf}_{self.content}{self.ext}"
        self.thread = threading.Thread(
            target=self._export,
            args=(checkpoint_dir, out_path, out_dir, lora_alpha, alpha_pattern),
            daemon=False,
        )
        self.thread.start()

    def _export(self, checkpoint_dir, out_path, out_dir, lora_alpha, alpha_pattern):
        try:
            export_state(
                checkpoint_dir,
                self.key,
                out_path,
                format=self.format,
                content=self.content,
                lora_alpha=lora_alpha,
                alpha_pattern=alpha_pattern,
            )
            if self.keep_latest_k >= 0:
                leaf = self.key.split(".")[-1]
                exports = sorted(out_dir.glob(f"step_*_{leaf}_{self.content}{self.ext}"), key=lambda path: path.name)
                for path in exports[: max(0, len(exports) - self.keep_latest_k)]:
                    path.unlink()
        except Exception:
            logger.error("export failed for %s:\n%s", checkpoint_dir, traceback.format_exc())

    def join(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Export a tensor subtree from a DCP checkpoint to safetensors or torch.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint-dir", type=Path, help="DCP checkpoint directory to read directly.")
    source.add_argument("--run-dir", type=Path, help="Run directory; checkpoint = run-dir/checkpoints/{step:07d}.")
    parser.add_argument("--step", type=int, default=None, help="Step to export when using --run-dir (default: latest complete).")
    parser.add_argument("--key", required=True, help="Dotted prefix to extract, e.g. models.backbone.")
    parser.add_argument("--out", required=True, type=Path, help="Output file path.")
    parser.add_argument("--format", choices=sorted(WRITERS), default=None, help="Export format (default: infer from --out suffix).")
    parser.add_argument("--content", choices=["full", "peft", "merged"], default="full", help="What to export (default: full).")
    parser.add_argument("--lora-alpha", type=float, default=None, help="LoRA alpha (required for --content merged).")
    parser.add_argument("--alpha-pattern", type=json.loads, default=None, help="JSON dict of per-module-suffix alpha overrides (optional, --content merged).")
    args = parser.parse_args(argv)

    if args.checkpoint_dir is not None:
        checkpoint_dir = args.checkpoint_dir
    else:
        checkpoints_dir = args.run_dir / "checkpoints"
        step = args.step
        if step is None and checkpoints_dir.exists():
            steps = [
                int(path.name)
                for path in checkpoints_dir.iterdir()
                if path.is_dir() and path.name.isdigit() and (path / ".metadata").exists()
            ]
            step = max(steps) if steps else None
        if step is None:
            raise RuntimeError(f"no complete checkpoint found in {checkpoints_dir}")
        checkpoint_dir = checkpoints_dir / f"{step:07d}"

    out_path = export_state(
        checkpoint_dir,
        args.key,
        args.out,
        format=args.format,
        content=args.content,
        lora_alpha=args.lora_alpha,
        alpha_pattern=args.alpha_pattern,
    )
    print(f"exported '{args.key}' ({args.content}) from {checkpoint_dir} to {out_path}")


if __name__ == "__main__":
    main()
