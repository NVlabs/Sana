"""Engine method library: the mechanics every training engine would copy verbatim.

``BaseEngine`` is NOT an orchestration framework, and must stay that way. Hard
rules for what may live here:

* no mandatory ``__init__`` and a deliberately small ``run`` -- the heavy
  construction and the training loop are project property. The base
  ``__init__`` is an *opt-in* minimal construction (host attributes,
  unified-parallel registration, model build) and the base ``run`` is the
  val-only loop (one ``validate``, then return), so inference-only engines
  get both for free. The training engines build their own optimizers /
  dataloaders / checkpoint trees in their own ``__init__``, never call the
  base one, and override ``run`` with their train loop;
* no meta-model primitive calls -- algorithm sequencing is project property;
* a method is admitted only when a second project would copy it verbatim.

Naming: base methods are public (no underscore) -- they are the API a project
engine builds on. A project engine's own internal helpers keep the ``_`` prefix,
so at any call site ``self.optimize()`` reads as base mechanics while
``self._training_loop()`` reads as project-internal.

Host attribute protocol (fail loud -- a host that has not set these up gets the
natural AttributeError/KeyError, no defaults)::

    config           full CfgNode; launch mounted ``config.checkpointer``
    engine_config    ``config.engine``
    iter             current inner iteration (int, ga microsteps included)
    models           name -> nn.Module (``<name>_ema`` is the reserved EMA name)
    optimizers       ``build_optimizers`` result (bare leaf or {class_name: opt})
    state_tree       the checkpoint state tree
    _runtime_state   the mutable ``{"iter": ...}`` leaf inside ``state_tree``
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import torch
from tabulate import tabulate
import torch.nn as nn
from torch.distributed.tensor import DTensor

from .distributed import ops
from .distributed.unified_parallel import init_unified_parallel, use_unified_parallel
from .logging import get_logger
from .model import build_models
from .meter import AverageMeter, get_running_accumulator, get_running_average_meter
from .phase import ExecutionPhase, execution_phase

logger = get_logger()


class BaseEngine:
    """Verbatim-reusable training mechanics; see the module docstring for scope."""

    # ``iter + 1`` at the previous ``log_metrics``, for the timer/iter window.
    # Class-level because the training engines do not call ``__init__`` below.
    _last_log_iter: int = 0

    def __init__(self, config: Any, meta_model: Any) -> None:
        """Opt-in minimal construction: host attributes + model build.

        Sets ``config``/``engine_config``/``meta``/``iter``, registers the
        unified-parallel topology and builds ``models`` from ``config.models``.
        That is everything ``validate()`` needs, so inference-only engines
        inherit this and only add a ``run`` (or rely on the base ``run``, the
        val-only loop). The training engines construct optimizers, dataloaders
        and the checkpoint tree around these same hosts in their own
        ``__init__`` and do not call this -- construction order stays project
        property.
        """
        self.config = config
        # Engine block is optional on the minimal path: an inference-only
        # trial carries no engine knobs at all.
        self.engine_config = config.get("engine", {})
        self.meta = meta_model
        self.iter = 0
        # Register every project topology before model/FSDP construction.
        self.setup_unified_parallel()
        self.models = build_models(config.models)

    def run(self) -> None:
        """Default loop: one validation at step 0, then return.

        The val-only loop for inference-only engines whose meta model has no
        training loop. Training engines override this with their train loop --
        the base run must never be reached by one, because their heavier
        construction never happened.
        """
        self.validate(step=0)

    def maybe_resume(self) -> bool:
        """Resume per ``engine.resume``: ``auto`` (latest, if any) | ``never`` | <int step>.

        ``engine.resume_dir`` optionally names ANOTHER run's ``checkpoints``
        directory or a complete numeric step directory within it as a seed: it is
        consulted only when this run has no complete checkpoint of its own, so
        mid-run restarts keep resuming the run's own latest and never roll back to the seed.
        ``resumed_from_seed`` records whether that external seed was actually used.

        Loads into ``state_tree`` in place, restores ``iter`` and records the
        loaded step as ``self.resumed_step``. Returns whether a checkpoint was
        loaded -- the fresh-start path (EMA step-0 sync etc.) belongs to the
        caller.

        ``engine.resume_iter_offset`` optionally rebases ``iter`` in either
        direction. The signed offset must preserve the dataloader worker phase
        and leave both counters non-negative. With partial loading and a
        non-strict dataloader, missing rank-local state is rebuilt from ``iter``.
        """
        resume = self.engine_config.get("resume", "auto")
        resume_dir = self.engine_config.get("resume_dir", None)
        self.resumed_from_seed = False
        assert resume != "never" or resume_dir is None, "resume_dir requires resume != 'never'"
        if resume == "never":
            return False
        own_latest = self.config.checkpointer.find_latest_step()
        if resume_dir is not None and own_latest is None:
            resume_path = Path(resume_dir)
            exact_step = (
                int(resume_path.name)
                if resume_path.name.isdigit() and (resume_path / ".metadata").exists()
                else None
            )
            checkpoints_dir = resume_path.parent if exact_step is not None else resume_dir
            if exact_step is not None:
                if resume != "auto":
                    assert int(resume) == exact_step, (
                        f"resume={resume} does not match resume_dir step {exact_step}"
                    )
                step = exact_step
            else:
                step = (
                    self.config.checkpointer.find_latest_step(checkpoints_dir)
                    if resume == "auto"
                    else int(resume)
                )
            assert step is not None, f"resume_dir={resume_dir} has no complete checkpoint"
            loaded = self.config.checkpointer.load(
                self.state_tree, step=step, checkpoints_dir=checkpoints_dir
            )
            self.resumed_from_seed = True
        else:
            step = own_latest if resume == "auto" else int(resume)
            if step is None:
                return False
            loaded = self.config.checkpointer.load(self.state_tree, step=step)
        self.iter = int(loaded["runtime"]["iter"])
        self._last_log_iter = self.iter
        self.resumed_step = int(loaded["step"])
        loader = self.state_tree.get("dataloader")
        if loader is not None and self.iter != loader.consumed * self._train_up_size:
            if not (self.config.checkpointer.allow_partial_load and not loader.strict):
                raise ValueError("checkpoint iter and dataloader consumed disagree")
            consumed, remainder = divmod(self.iter, self._train_up_size)
            if remainder:
                raise ValueError("checkpoint iter is not divisible by engine.up_size.train")
            loader._reset_for_partial_resume(consumed)
            logger.warning("Reset partial dataloader state at consumed=%d", consumed)

        offset = int(self.engine_config.get("resume_iter_offset", 0))
        if offset:
            if self.resumed_step != int(self.engine_config.get("resume_iter_offset_step", -1)):
                logger.info(
                    "Skipping engine.resume_iter_offset=%d: loaded step=%d is not the offset step",
                    offset,
                    self.resumed_step,
                )
            else:
                if loader is None:
                    raise ValueError("engine.resume_iter_offset requires the dataloader in state_tree")
                consumed_offset, remainder = divmod(offset, self._train_up_size)
                if (
                    remainder
                    or consumed_offset % loader.effective_workers
                    or self.iter + offset < 0
                    or loader.consumed + consumed_offset < 0
                ):
                    raise ValueError(
                        "engine.resume_iter_offset must preserve worker phase and non-negative state"
                    )
                self.iter += offset
                self._runtime_state["iter"] = self.iter
                self._last_log_iter = self.iter
                loader.consumed += consumed_offset
                logger.warning(
                    "Applied engine.resume_iter_offset=%d at step=%d: iter=%d, loader consumed=%d",
                    offset,
                    self.resumed_step,
                    self.iter,
                    loader.consumed,
                )
        logger.info("Resumed at iter=%d (step=%d)", self.iter, self.resumed_step)
        return True

    @torch.no_grad()
    def sync_ema_from_model(self) -> None:
        """Step-0 EMA contract: on a fresh start every ``<name>_ema`` becomes an
        exact copy of its live model (the shadow build's random init does not
        match the model's -- two independent constructions). Never call this on
        a resumed run: it would overwrite the restored EMA average.
        """
        for name, model in self.models.items():
            ema = self.models.get(f"{name}_ema")
            if ema is None:
                continue
            for src, tgt in zip(model.parameters(), ema.parameters()):
                target = tgt.detach()
                source = src if src.device == target.device else src.to(target.device)
                target.copy_(source)
            logger.info("EMA[%s] initialized from the live model", name)

    # ------------------------------------------------------------ mechanics --
    def trained_models(self) -> dict[str, nn.Module]:
        """The models that own an optimizer (training targets)."""
        return {name: self.models[name] for name in self.optimizers}

    def set_gradient_sync(self, sync: bool) -> None:
        """FSDP2 gradient-sync gating; plain DDP (no method) is a no-op."""
        for model in self.trained_models().values():
            if hasattr(model, "set_requires_gradient_sync"):
                model.set_requires_gradient_sync(sync)

    def at_interval(self, start_step: int, interval: int) -> bool:
        """Whether ``iter`` sits on an optimizer-step interval boundary (ga-aligned)."""
        ga = int(self.engine_config.ga_steps)
        return (self.iter + 1) >= ga * int(start_step) and (self.iter + 1) % (ga * int(interval)) == 0

    @execution_phase(ExecutionPhase.TRAIN_BACKWARD)
    def backward(self, loss: torch.Tensor) -> None:
        """Backpropagate an already gradient-accumulation-scaled loss."""
        loss.backward()

    @execution_phase(ExecutionPhase.OPTIMIZATION)
    def optimize(self, names: list[str] | None = None) -> dict[str, float]:
        """Per trained model: grad clip -> NaN/Inf skip guard -> step/zero -> EMA.

        ``names`` restricts the pass to a subset of the trained models --
        multi-model schedules (adversarial/distillation TTUR phases) step one
        model group per call; the default (None) steps every trained model.

        Returns ``grad_norm/<name>`` scalars for logging. A non-finite norm skips
        that model's step (grads zeroed, training continues).
        """
        cfg = self.engine_config
        trained = self.trained_models()
        if names is not None:
            missing = [name for name in names if name not in trained]
            assert not missing, f"optimize names {missing} are not trained models {sorted(trained)}"
            trained = {name: trained[name] for name in names}
        norm_dict: dict[str, float] = {}
        for name, model in trained.items():
            optimizers = (
                list(self.optimizers[name].values())
                if isinstance(self.optimizers[name], dict)
                else [self.optimizers[name]]
            )

            norm = nn.utils.clip_grad_norm_(model.parameters(), float(cfg.clip_grad_norm))
            if isinstance(norm, DTensor):
                norm = norm.full_tensor()

            if not torch.isfinite(norm):
                logger.warning("grad norm %s for %s at iter %d: step skipped", norm, name, self.iter)
                for optimizer in optimizers:
                    optimizer.zero_grad()
                continue
            norm_dict[f"grad_norm/{name}"] = float(norm.item())

            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()

            self.update_ema(name, model)
        return norm_dict

    @torch.no_grad()
    def update_ema(self, name: str, model: nn.Module) -> None:
        """EMA <- decay * EMA + (1 - decay) * model, trainable params only."""
        ema = self.models.get(f"{name}_ema")
        if ema is None:
            return
        decay = float(self.engine_config.ema_decay[name])
        for src, tgt in zip(model.parameters(), ema.parameters()):
            if not src.requires_grad:
                continue
            target = tgt.detach()
            # CPU-offloaded EMA (ema.placement.fsdp.cpu_offload): bring the live
            # shard to the EMA shard's device; same-device path is unchanged.
            source = src if src.device == target.device else src.to(target.device)
            target.mul_(decay).add_(source, alpha=1.0 - decay)

    # ---------------------------------------------------- logging / storage --
    def log_metrics(self, meter: AverageMeter, step_timer: float) -> None:
        """Own the metric sync+reset boundary for one logging window.

        Callers only put scalars into their local ``meter`` (and, from deep code,
        the running meter singletons). This method syncs all three sources,
        merges them, emits metrics/stdout, then resets the window.
        """
        meter.sync()
        running_average_meter = get_running_average_meter()
        running_accumulator = get_running_accumulator()
        running_average_meter.sync()
        running_accumulator.sync()

        means = meter.avg
        means.update(running_average_meter.avg)
        means.update(running_accumulator.sum)

        ga = int(self.engine_config.ga_steps)
        optimizer_step = (self.iter + 1) // ga
        # Measured, not derived: each engine gates process_due differently, so
        # no expression over log_interval/ga/up_size holds for all of them.
        iters_since_log = max((self.iter + 1) - self._last_log_iter, 1)
        means["timer/iter"] = (time.perf_counter() - step_timer) / iters_since_log
        means["timer/step"] = means["timer/iter"] * ga
        self._last_log_iter = self.iter + 1
        if torch.cuda.is_available():
            memory_free, memory_total = torch.cuda.mem_get_info()
            means["memory/used_gb"] = (memory_total - memory_free) / (1024**3)
            means["memory/alloc_retries"] = torch.cuda.memory_stats()["num_alloc_retries"]
        get_logger().log_metrics(means, step=optimizer_step)

        # Two-column metric table, folded side by side to keep it compact.
        table_data = [[key, f"{value:.6f}"] for key, value in sorted(means.items())]
        n_cols = 2
        nrows = math.ceil(len(table_data) / n_cols)
        table_data += [["", ""]] * (nrows * n_cols - len(table_data))
        table_data = [[x for j in range(n_cols) for x in table_data[j * nrows + i]] for i in range(nrows)]
        logger.info(
            "[iter %07d | step %07d]\n%s",
            self.iter + 1, optimizer_step,
            tabulate(
                table_data,
                headers=["Metric", "Value"] * n_cols,
                tablefmt="heavy_outline",
                numalign="left",
                stralign="left",
            ),
        )
        meter.reset()
        running_average_meter.reset()
        running_accumulator.reset()

    def save(self) -> None:
        """Write state at the current optimizer step (ga-collapsed folder name)."""
        self._runtime_state["iter"] = self.iter + 1
        step = (self.iter + 1) // int(self.engine_config.ga_steps)
        self.config.checkpointer.save(self.state_tree, step=step)

    def process_due(
        self,
        meter: AverageMeter,
        step_timer: float,
        *,
        total_iters: int | None = None,
    ) -> tuple[float, bool]:
        """Evaluate the due flags and run their actions. Returns (step_timer, should_stop).

        One call replaces the log/save/val/runtime bookkeeping every training
        engine copies verbatim. Call it from the engine's one clean node -- the
        loop point where every due action is safe: logging
        (``engine.log_interval``), save/validate (``engine.save_start_step /
        save_interval / val_interval / val_start_step``, final-step save via
        ``total_iters``), and the collective runtime stop (any rank due stops
        all; ``engine.early_stop_validate`` runs one final validation before a
        runtime stop).
        """
        cfg = self.engine_config
        save_due = (
            self.at_interval(int(cfg.get("save_start_step", 0)), int(cfg.save_interval))
            or (total_iters is not None and self.iter + 1 == total_iters)
        )
        val_due = int(cfg.get("val_interval", 0)) > 0 and self.at_interval(
            int(cfg.get("val_start_step", 0)), int(cfg.get("val_interval", 0))
        )
        runtime_due = False
        if self._early_stop_seconds is not None:
            # Any rank due stops all (mirrors the DMD project's runtime cap).
            runtime_flag = torch.tensor(
                int(time.monotonic() - self._start_time >= self._early_stop_seconds),
                dtype=torch.int32,
                device="cpu",
            )
            ops.all_reduce_max(runtime_flag)
            runtime_due = bool(runtime_flag.item())
        if self.at_interval(0, int(cfg.log_interval)):
            self.log_metrics(meter, step_timer)
            step_timer = time.perf_counter()
        if save_due or runtime_due:
            self.save()
        if val_due or (runtime_due and bool(cfg.get("early_stop_validate", False))):
            self.validate(step=(self.iter + 1) // int(cfg.ga_steps))
        if runtime_due:
            self.iter += 1
            logger.info("Runtime limit reached at iter=%d; exiting training loop", self.iter)
            return step_timer, True
        return step_timer, False

    # ------------------------------------------------------ unified parallel --
    def setup_unified_parallel(self) -> None:
        """Register the unified-parallel topology before any model/FSDP construction.

        ``distributed.up_size`` registers the available UP options (the sizes an
        SP group splits a sequence across); ``engine.up_size`` picks the
        training size and optional validation size from them. With no
        ``distributed.up_size``, train size defaults to 1 (validation None) and
        UP is inert -- this is callable unconditionally. Also records
        ``sp_group_id`` and ``num_sp_groups`` for SP-aware code.
        """
        options = self.config.get("distributed", {}).get("up_size")
        selectors = self.engine_config.get("up_size")
        if options is None:
            assert selectors is None, "engine.up_size requires distributed.up_size"
            train_size = 1
            validation_size = None
        else:
            options = tuple(options)
            assert selectors is not None, "engine.up_size is required"
            train_size = selectors["train"]
            validation_size = selectors.get("validation")
            assert train_size in options, "engine.up_size.train must be registered"
            assert validation_size is None or validation_size in options, (
                "engine.up_size.validation must be registered"
            )
            init_unified_parallel(options, train_size)

        self._train_up_size = train_size
        self._validation_up_size = validation_size
        self.sp_group_id = ops.get_rank() // train_size
        self.num_sp_groups = ops.get_world_size() // train_size

    # ------------------------------------------------------------ validation --
    @torch.no_grad()
    def validate(self, step: int) -> None:
        """Run project validation at an explicit logical optimizer step.

        Validation is a meta primitive: every rank participates (SP collectives
        require the whole group -- no rank0-only generation), and sample/media
        writing goes through the framework-wide logger's media/artifact methods.
        Train/eval switching is owned by the meta model's validate (it knows
        which of its models actually train).

        Raw-vs-EMA rendering is the meta model's business, not the engine's:
        it keys off ``validation.validate_ema`` and renders ``backbone_ema`` as
        its own variant (the wan_t2v pattern). There is no engine-level EMA
        switch here.
        """
        ctx = {
            "config": self.config,
            "models": self.models,
            "step": step,
        }
        if self._validation_up_size is None:
            self.meta.validate(ctx)
        else:
            with use_unified_parallel(self._validation_up_size):
                self.meta.validate(ctx)
