# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from heapq import heappop, heappush

import numpy as np

POOL_FILL = 0.95
LOOKAHEAD = 1024
END_FILL = 0.95
FIRST_AFTER_DEFERRALS = 2
SHORT_SHARE = 100
PROBE_SPAN = 6
PROBE_SEED = 0

NOT_POOLED, WAITING, PLACED = 0, 1, 2


def pack_steps(order, lengths, ranks: int, chunk_budget: int, min_gap: int) -> list:
    """Return the training steps of an epoch: per step one row of windows for every rank, each row holding at most ``chunk_budget`` chunks (a longer window takes a row alone), cut from the queue ``order`` of windows of ``lengths`` chunks. ``PackedWindowSampler`` describes the rule."""
    order = [int(window) for window in order]
    chunks = [int(length) for length in np.asarray(lengths, dtype=np.int64).tolist()]
    if not order:
        return []
    count = len(order)
    position = [count] * len(chunks)
    for index, window in enumerate(order):
        position[window] = index
    shortest = min(chunks[window] for window in order)
    longest = max(chunks[window] for window in order)
    state = bytearray(len(chunks))
    unplaced_by_length = [deque() for _ in range(longest + 1)]
    for window in order:
        unplaced_by_length[chunks[window]].append(window)

    def earliest(length, limit):
        queue = unplaced_by_length[length]
        while queue and state[queue[0]] == PLACED:
            queue.popleft()
        return queue[0] if queue and position[queue[0]] < limit else None

    pool_chunks = POOL_FILL * ranks * chunk_budget
    end_load = END_FILL * chunk_budget
    cursor, waiting, steps = 0, [], []
    deferrals = bytearray(len(chunks))
    while True:
        carried, fresh, pooled, kept = [], [], 0, []
        for window in waiting:
            if state[window] != WAITING:
                continue
            if pooled < pool_chunks:
                (carried if deferrals[window] >= FIRST_AFTER_DEFERRALS else fresh).append(window)
                pooled += chunks[window]
            else:
                kept.append(window)
        while pooled < pool_chunks and cursor < count:
            window = order[cursor]
            cursor += 1
            if state[window] == NOT_POOLED:
                state[window] = WAITING
                fresh.append(window)
                pooled += chunks[window]
        if not carried and not fresh:
            break
        loads = [0] * ranks
        rows = [[] for _ in range(ranks)]
        lightest = [(0, rank) for rank in range(ranks)]

        def least_loaded():
            while lightest[0][0] != loads[lightest[0][1]]:
                heappop(lightest)
            return lightest[0]

        def place(window, rank):
            rows[rank].append(window)
            loads[rank] += chunks[window]
            state[window] = PLACED
            heappush(lightest, (loads[rank], rank))

        deferred = []
        for window in carried:
            load, rank = least_loaded()
            if load and load + chunks[window] > chunk_budget:
                deferred.append(window)
            else:
                place(window, rank)
        fresh.sort(key=lambda window: (-chunks[window], position[window]))
        for window in fresh:
            load, rank = least_loaded()
            gap = chunk_budget - load - chunks[window]
            if 0 < gap < min_gap:
                exact_load = chunk_budget - chunks[window]
                rank = next((other for other in range(ranks) if loads[other] == exact_load), None)
                gap = None if rank is None else 0
            if gap is None or (gap < 0 and load):
                deferred.append(window)
            else:
                place(window, rank)
        limit = cursor + LOOKAHEAD
        for rank in sorted(range(ranks), key=lambda other: (loads[other], other)):
            while True:
                gap = chunk_budget - loads[rank]
                if gap < shortest:
                    break
                window = earliest(gap, limit) if gap <= longest else None
                for length in range(min(gap - min_gap, longest), shortest - 1, -1):
                    if window is not None:
                        break
                    window = earliest(length, limit)
                for length in range(min(gap, longest), max(gap - min_gap, shortest - 1), -1):
                    if window is not None:
                        break
                    window = earliest(length, limit)
                if window is None and not rows[rank]:
                    fronts = [earliest(length, count) for length in range(shortest, longest + 1)]
                    window = min(
                        (front for front in fronts if front is not None), key=position.__getitem__, default=None
                    )
                if window is None:
                    break
                place(window, rank)
        while cursor < count and state[order[cursor]] == PLACED:
            cursor += 1
        if cursor >= count and min(loads) < end_load:
            break
        steps.append(rows)
        for window in deferred:
            if state[window] == WAITING and deferrals[window] < 255:
                deferrals[window] += 1
        waiting = sorted((window for window in deferred + kept if state[window] == WAITING), key=position.__getitem__)
    return steps


def choose_min_gap(lengths, ranks: int, chunk_budget: int) -> int:
    """Return the smallest gap a placement may leave on a rank: a small gap needs a window of exactly its length, and short windows are rare. Each candidate, from the shortest length with at most 1% of the windows below it, packs one probe epoch of a fixed random order, and the one with the fullest rows wins."""
    lengths = np.asarray(lengths, dtype=np.int64)
    first = int(np.sort(lengths)[len(lengths) // SHORT_SHARE])
    order = np.random.default_rng(PROBE_SEED).permutation(len(lengths))
    best, best_placed, best_capacity = first, 0, 1
    for candidate in range(first, first + PROBE_SPAN):
        steps = pack_steps(order, lengths, ranks, chunk_budget, candidate)
        placed = sum(int(lengths[row].sum()) for step in steps for row in step)
        capacity = len(steps) * ranks * chunk_budget
        if capacity and placed * best_capacity > best_placed * capacity:
            best, best_placed, best_capacity = candidate, placed, capacity
    return best
