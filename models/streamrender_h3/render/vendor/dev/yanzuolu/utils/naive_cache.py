"""Packed key/value cache with sink and sliding-window eviction.

The constructor requires keyword-only ``sink`` and ``window_size`` values for
each batch element. Both are counted in chunks: the first ``sink`` chunks remain
pinned, the most recent ``window_size`` chunks form the sliding history, and
``window_size=None`` disables eviction. Once history exceeds
``sink + window_size``, exactly one chunk is removed per update.

``update_kvlens`` records incoming chunk lengths and maintains per-sample cache
lengths. ``update_kvcache`` removes evicted spans from packed key/value tensors
while retaining the sink and active window. ``curr_rope`` stores per-sample
relative position offsets for callers that need cache rebasing; callers using
absolute positions may leave it unchanged.

Training-time attention visibility must follow the same sink and window rules so
queries cannot attend to history that has already been removed from the cache.
"""
import itertools

import torch

from dev.yanzuolu.common.distributed.ops import get_device


class NaiveCache:
    def __init__(self, num_layers, batch_size=None, *, sink, window_size):
        self.batch_size = batch_size
        self.key_cache = {k: None for k in range(num_layers)}
        self.value_cache = {k: None for k in range(num_layers)}
        if batch_size is not None:
            self.kvlens = [0] * batch_size
            self.curr_rope = torch.tensor([0] * batch_size, device=get_device())

        self.sink = sink
        self.window_size = window_size
        # Unified history of chunk lengths for all steps
        # List[List[int]], outer: time step (chunk), inner: batch
        self.chunk_lens = []
        # Serving-only arenas keep history K/V in its final attention layout
        # and reserve a tail for the current query. Steady-state denoising
        # overwrites only that tail instead of scattering all history again.
        self._persistent_key_arenas = None
        self._persistent_value_arenas = None
        self._persistent_context_len = None
        self._persistent_query_capacity = None
        # Optional one-forward capture routing. A normal full attention can
        # tap selected post-RoPE K/V rows without entering cached attention or
        # launching a separate prefix prefill.
        self._capture_indices = None
        # self.kvlens_sink = []    # List[List[int]], inner batch, outer chunk
        # self.kvlens_window = []  # List[List[int]], inner batch, outer chunk

    def update_kvlens(self, new_sample_lens):
        assert self.batch_size is not None, "Batch size must be specified to update kvlens."

        # 1. Record the length of the incoming chunk for history
        self.chunk_lens.append(new_sample_lens)
        current_step = len(self.chunk_lens)

        # 2. Update kvlens per sample
        for b in range(self.batch_size):
            s = self.sink[b]
            w = self.window_size[b]
            new_len = new_sample_lens[b]

            # Calculate pop length if window is full
            # Condition: We have more chunks than sink + window
            pop_len = 0
            if w is not None and current_step > s + w:
                # The chunk to pop is the one that falls out of the window.
                # Index in history: current_step - w - 1 (the new one) - 1 (to get the one before window starts? No)
                # Logic:
                # Chunks: [0, 1, 2, 3], s=1, w=2. Keep 0 (sink), 2, 3 (window). Pop 1.
                # current_step=4. Pop index = 4 - 2 - 1 = 1. Correct.
                pop_idx = current_step - w - 1
                pop_len = self.chunk_lens[pop_idx][b]

            self.kvlens[b] = self.kvlens[b] + new_len - pop_len

    def update_kvcache(self, layer_idx, new_keys, new_values, new_kvlens=None):
        # new_kvlens is accepted for compatibility with positional call sites but
        # is not read. It remains optional so callers without a meaningful value
        # may omit it.
        del new_kvlens
        # Note: At this point, update_kvlens has already been called, and self.chunk_lens contains the length of the current step.
        # self.kvlens has also been updated (the pop_len has been subtracted).
        current_step = len(self.chunk_lens)

        chunks_k = []
        chunks_v = []
        input_offset = 0
        for b in range(self.batch_size):
            s = self.sink[b]
            w = self.window_size[b]

            # 1. Calculate Sink Length (s_len)
            # Sum lengths of the first 's' chunks for this batch index
            # Optimization: If s=0, s_len=0.
            s_len = 0
            if s > 0:
                # Only accumulate the actual existing chunks to prevent initial step < s
                limit = min(current_step, s)
                for i in range(limit):
                    s_len += self.chunk_lens[i][b]
            # 2. Calculate Pop Length (p_len)
            # This is the length of the chunk that is being evicted in this step
            p_len = 0
            if w is not None and current_step > s + w:
                pop_idx = current_step - w - 1
                p_len = self.chunk_lens[pop_idx][b]

            # 3. Calculate total length of the sample in the INPUT tensor (new_keys)
            # Input contains: [Retained Old Cache] + [Popped Part] + [New Token]
            # self.kvlens[b] is the target length (Retained + New).
            # So input length = self.kvlens[b] + p_len
            current_input_len = self.kvlens[b] + p_len

            # 4. Slicing Logic
            if p_len == 0:
                # No eviction, keep everything for this sample
                chunks_k.append(new_keys[input_offset : input_offset + current_input_len])
                chunks_v.append(new_values[input_offset : input_offset + current_input_len])
            else:
                # Eviction happens: Keep Sink + Skip Pop + Keep Window(including new)
                # Part 1: Sink
                if s_len > 0:
                    chunks_k.append(new_keys[input_offset : input_offset + s_len])
                    chunks_v.append(new_values[input_offset : input_offset + s_len])

                # Part 2: Window (Skip the popped part)
                # Start after sink + pop_len
                window_start = input_offset + s_len + p_len
                window_end = input_offset + current_input_len

                if window_end > window_start:
                    chunks_k.append(new_keys[window_start : window_end])
                    chunks_v.append(new_values[window_start : window_end])
            # Move offset
            input_offset += current_input_len
        # 5. Concatenate and Update
        self.key_cache[layer_idx] = (
            torch.cat(chunks_k, dim=0)
            if chunks_k else new_keys.new_empty((0, *new_keys.shape[1:]))
        )
        self.value_cache[layer_idx] = (
            torch.cat(chunks_v, dim=0)
            if chunks_v else new_values.new_empty((0, *new_values.shape[1:]))
        )

    def prepare_selected_capture(self, prefix_lens, sample_lens, device) -> None:
        """Select each packed document's leading prefix during one full forward."""
        if self.batch_size != len(sample_lens) or len(prefix_lens) != len(sample_lens):
            raise ValueError("capture lengths must match the cache batch size")
        indices = []
        offset = 0
        normalized = []
        for prefix_len, sample_len in zip(prefix_lens, sample_lens, strict=True):
            prefix_len, sample_len = int(prefix_len), int(sample_len)
            if prefix_len < 0 or prefix_len > sample_len:
                raise ValueError("capture prefix length must be inside its document")
            if prefix_len:
                indices.append(torch.arange(offset, offset + prefix_len, device=device))
            normalized.append(prefix_len)
            offset += sample_len
        self._capture_indices = (
            torch.cat(indices).to(torch.long)
            if indices else torch.empty(0, dtype=torch.long, device=device)
        )
        self.kvlens = normalized
        self.chunk_lens = [list(normalized)]

    def capture_selected_kvcache(self, layer_idx, keys, values) -> None:
        """Store selected rows while leaving full attention tensors untouched."""
        if self._capture_indices is None:
            raise RuntimeError("prepare_selected_capture must run before capture")
        self.key_cache[layer_idx] = keys.index_select(0, self._capture_indices)
        self.value_cache[layer_idx] = values.index_select(0, self._capture_indices)

    def prepare_persistent_attention_arenas(self, query_capacity: int) -> None:
        """Materialize one real prefix plus optional empty SP padding."""
        if self.batch_size is None or self.batch_size < 1:
            raise ValueError("persistent attention arenas require a concrete batch")
        if any(int(length) != 0 for length in self.kvlens[1:]):
            raise ValueError(
                "persistent prefix arena supports one real sample followed by "
                "zero-prefix alignment documents"
            )
        if query_capacity <= 0:
            raise ValueError("query_capacity must be positive")
        context_len = sum(int(length) for length in self.kvlens)
        key_arenas, value_arenas = {}, {}
        for layer_idx in range(self.num_layers):
            keys = self.key_cache[layer_idx]
            values = self.value_cache[layer_idx]
            if keys is None or values is None:
                raise ValueError(f"layer {layer_idx} has no prefetched K/V")
            if int(keys.shape[0]) != context_len or int(values.shape[0]) != context_len:
                raise ValueError("persistent arena requires contiguous prefix history")
            key_arena = keys.new_empty((context_len + query_capacity, *keys.shape[1:]))
            value_arena = values.new_empty((context_len + query_capacity, *values.shape[1:]))
            key_arena[:context_len].copy_(keys)
            value_arena[:context_len].copy_(values)
            key_arenas[layer_idx] = key_arena
            value_arenas[layer_idx] = value_arena
            self.key_cache[layer_idx] = key_arena[:context_len]
            self.value_cache[layer_idx] = value_arena[:context_len]
        self._persistent_key_arenas = key_arenas
        self._persistent_value_arenas = value_arenas
        self._persistent_context_len = context_len
        self._persistent_query_capacity = int(query_capacity)

    def merge_current_for_attention(self, layer_idx, new_keys, new_values):
        """Return a persistent ``[history,current]`` view when enabled."""
        if self._persistent_key_arenas is None:
            return None
        query_len = int(new_keys.shape[0])
        if query_len > self._persistent_query_capacity:
            raise ValueError(
                f"query length {query_len} exceeds persistent capacity "
                f"{self._persistent_query_capacity}"
            )
        start = self._persistent_context_len
        stop = start + query_len
        key_arena = self._persistent_key_arenas[layer_idx]
        value_arena = self._persistent_value_arenas[layer_idx]
        key_arena[start:stop].copy_(new_keys)
        value_arena[start:stop].copy_(new_values)
        return key_arena[:stop], value_arena[:stop]

    @property
    def num_layers(self):
        return len(self.key_cache)

    @property
    def seq_len(self):
        if self.key_cache[0] is not None:
            return self.key_cache[0].shape[0]
        else:
            return 0

    def seq_lens(self, idx):
        if self.key_cache[idx] is not None:
            return self.key_cache[idx].shape[0]
        else:
            return 0

    @staticmethod
    def merge(caches):
        """ Merge a list of NaiveCache into a single NaiveCache by concatenating along batch dimension. """
        assert len(caches) > 0
        num_layers = caches[0].num_layers
        assert all([cache.num_layers == num_layers for cache in caches]), "All caches must have the same number of layers."
        total_batch_size = sum([len(cache.kvlens) for cache in caches])
        merged_cache = NaiveCache(
            num_layers,
            total_batch_size,
            sink=list(itertools.chain.from_iterable([cache.sink for cache in caches])),
            window_size=list(itertools.chain.from_iterable([cache.window_size for cache in caches])),
        )
        for layer_idx in range(num_layers):
            merged_keys = torch.cat([cache.key_cache[layer_idx] for cache in caches], dim=0)
            merged_values = torch.cat([cache.value_cache[layer_idx] for cache in caches], dim=0)
            merged_cache.key_cache[layer_idx] = merged_keys
            merged_cache.value_cache[layer_idx] = merged_values
        merged_cache.kvlens = list(itertools.chain.from_iterable([cache.kvlens for cache in caches]))
        merged_cache.curr_rope = torch.cat([cache.curr_rope for cache in caches], dim=0)

        # Merge chunk_lens
        # chunk_lens is List[List[int]] (Time, Batch).
        # We need to concatenate the inner lists along the batch dimension for each time step.
        # Assuming all caches have the same number of steps (chunks).
        if len(caches) > 0 and hasattr(caches[0], 'chunk_lens'):
            num_steps = len(caches[0].chunk_lens)
            merged_cache.chunk_lens = []
            for i in range(num_steps):
                # Combine the batch lists for step i
                step_lens = list(itertools.chain.from_iterable([c.chunk_lens[i] for c in caches]))
                merged_cache.chunk_lens.append(step_lens)

        return merged_cache
