# SPDX-License-Identifier: Apache-2.0
"""Deliver each streaming latent pack as one emission per training window.

Chain metas train a clip's windows in order, one window per engine
iteration, and carry model-generated history from one iteration into the
next. The engine pulls exactly one dataloader item per iteration, so a pack
is delivered as a chain of ``L`` consecutive emissions, ``L`` being the
number of streaming windows per clip: a head batch carrying the complete
pack, then ``L - 1`` light batches carrying only the chain bookkeeping the
meta needs to keep advancing its history.

Without further care every rank and worker stream would begin a chain on
the same iteration and hit the same window index thereafter. The first
chain of each stream therefore begins at a rank- and worker-dependent
phase. It still starts at window 0 but stops after ``L - phase`` windows,
and the truncated tail is simply not trained on this visit. Ranks of one
SP sync group share the phase because they must replicate one stream.

A mid-chain snapshot records the pre-pack draw state and the number of
emissions so far, so a resume regenerates the identical pack and delivers
it again as a head. By default the resumed head restarts the chain at
window 0, truncated to the windows the snapshot had not yet emitted,
because the meta's history of the interrupted chain is gone. Metas that
checkpoint their chain state set ``resume_mid_chain`` instead. Their
snapshots also record the emission count at which the chain began, and the
resumed head keeps the interrupted chain's ``chain_index`` and
``chain_length`` so that the meta continues the chain at that window. The
last emission of a chain commits the post-pack draw state.
"""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import Any

from torch.utils.data import get_worker_info

from dev.yanzuolu.common.data import WorkerResumeContext, WorkerStateEnvelope, WorkerStateLoadError
from dev.yanzuolu.projects.minimax_h3.data.streaming import StreamingLatentT2AVDataset, streaming_window_starts


class StreamingLatentChainMixin:
    """Emit a head batch with the pack, then light batches for its remaining windows.

    Hosts are streaming latent datasets that own the corpus draw, packing
    budget and worker-state schema. Every emission carries ``chain_id``,
    ``chain_index``, ``chain_length`` and ``chain_windows``. ``chain_id`` is
    keyed by the pre-pack offset so it survives a resume. A head's
    ``chain_index`` is 0 except when ``resume_mid_chain`` resumes a chain
    mid-way. ``_pack_sample`` stays inherited because other code
    instantiates training datasets as layout packers and calls it directly.
    """

    def __init__(
        self, seed: int, resume_context: WorkerResumeContext, *,
        chain_phase_stagger: bool = True, resume_mid_chain: bool = False, **kwargs: Any,
    ) -> None:
        super().__init__(seed, resume_context, **kwargs)
        self.chain_phase_stagger = bool(chain_phase_stagger)
        self.resume_mid_chain = bool(resume_mid_chain)
        self._decoded_chain_progress: dict[int, tuple[int, int | None]] = {}
        for logical_id, snapshot in resume_context.committed_states.items():
            # The base already unpickled and schema-checked this snapshot, so
            # only the chain fields themselves can be missing or malformed.
            # Only ``resume_mid_chain`` datasets record ``chain_start``.
            state = pickle.loads(snapshot)
            try:
                emitted = state["emitted"]
            except KeyError as exc:
                raise WorkerStateLoadError(f"invalid snapshot for logical worker {logical_id}") from exc
            chain_start = state.get("chain_start")
            if (
                not isinstance(emitted, int) or emitted < 0
                or (chain_start is not None and (not isinstance(chain_start, int) or not 0 <= chain_start <= emitted))
            ):
                raise WorkerStateLoadError(f"invalid snapshot for logical worker {logical_id}")
            self._decoded_chain_progress[logical_id] = (emitted, chain_start)

    def _encode_worker_state(
        self, logical_worker_id: int, offset: int, avg_seqlen: float, cnt: int, emitted: int = 0,
        chain_start: int | None = None,
    ) -> bytes:
        state = {
            "schema": self._STATE_SCHEMA,
            "version": self._STATE_VERSION,
            "logical_worker_id": logical_worker_id,
            "offset": offset,
            "avg_seqlen": avg_seqlen,
            "cnt": cnt,
            "emitted": emitted,
        }
        if chain_start is not None:
            state["chain_start"] = chain_start
        return pickle.dumps(state)

    def _chain_windows(self, samples: Sequence[dict[str, Any]]) -> int:
        windows = {
            len(streaming_window_starts(
                sample["latent_shapes"][1], **sample["streaming_config"],
                video_temporal_mapping=self.video_temporal_mapping,
            ))
            for sample in samples
        }
        if len(windows) != 1:
            raise ValueError(f"packed samples disagree on their streaming window count: {sorted(windows)}")
        return windows.pop()

    def _chain_phase(self, logical_worker_id: int, windows: int, effective_workers: int) -> int:
        if not self.chain_phase_stagger:
            return 0
        stream = self.resume_context.rank // self.sync_group_size
        return (stream + round(logical_worker_id * windows / effective_workers)) % windows

    def __iter__(self):
        # The host's _build_pack draws a pack, and its emission is expanded here.
        worker_info = get_worker_info()
        physical_worker_id = worker_info.id if worker_info else 0
        physical_worker_count = worker_info.num_workers if worker_info else 1
        effective_workers = self.resume_context.num_workers or 1
        if physical_worker_count != effective_workers:
            raise ValueError(
                "worker topology mismatch: context expects "
                f"{effective_workers}, runtime has {physical_worker_count}"
            )
        logical_worker_id = (
            physical_worker_id + self.resume_context.next_logical_worker_id
        ) % physical_worker_count
        if logical_worker_id in self._decoded_worker_states:
            pack_state = self._decoded_worker_states[logical_worker_id]
            emitted, chain_start = self._decoded_chain_progress[logical_worker_id]
            # Without ``resume_mid_chain``, or from a snapshot without a chain
            # start, the resumed chain restarts at window 0.
            if not self.resume_mid_chain or chain_start is None:
                chain_start = emitted
        else:
            pack_state = self._initial_worker_state(logical_worker_id)
            # The phase needs the window count, which the first pack supplies.
            emitted = chain_start = None
        rank = self.resume_context.rank

        while True:
            samples, next_state = self._build_pack(*pack_state)
            windows = self._chain_windows(samples)
            if emitted is None:
                emitted = chain_start = self._chain_phase(logical_worker_id, windows, effective_workers)
            chain_id = f"{rank}:{logical_worker_id}:{pack_state[0]}"
            item = {key: [sample[key] for sample in samples] for key in samples[0]}
            while emitted < windows:
                item.update(
                    chain_id=chain_id, chain_index=emitted - chain_start,
                    chain_length=windows - chain_start, chain_windows=windows,
                )
                emitted += 1
                state_after = (
                    self._encode_worker_state(
                        logical_worker_id, *pack_state, emitted=emitted,
                        chain_start=chain_start if self.resume_mid_chain else None,
                    )
                    if emitted < windows
                    else self._encode_worker_state(logical_worker_id, *next_state)
                )
                yield WorkerStateEnvelope(item, logical_worker_id, state_after)
                item = {}
            pack_state, emitted, chain_start = next_state, 0, 0


class StreamingLatentChainDataset(StreamingLatentChainMixin, StreamingLatentT2AVDataset):
    """Chain emissions over the plain T2AV streaming latent corpus."""

    _STATE_SCHEMA = "minimax_h3_streaming_latent_chain_worker"
    _STATE_VERSION = 1


EntryClass = StreamingLatentChainDataset

__all__ = ["StreamingLatentChainMixin", "StreamingLatentChainDataset", "EntryClass"]
