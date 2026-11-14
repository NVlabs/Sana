"""Give the slower GPUs of a node fewer sequence rows, so the step stops being `max over ranks`.

The eight H100s in a node do not all compute a denoise step in the same time. Per-rank Kineto of one
g8 step (job 21827378, node pool0-01964) measured compute — everything that is not NCCL — at
min 611.0 / mean 623.4 / max 645.4 ms, a 5.6% spread, with the attention kernels alone spanning
365.6-384.4 ms and the GEMMs 255.0-267.5 on identical shapes. Because every layer ends in two
collectives, the step costs the *slowest* rank and the fast ranks bank the difference as NCCL wait
(exposed NCCL 60.7-99.3 ms, anticorrelated with compute). Flattening the distribution is worth
max-to-mean, ~3.4% of the step at g8.

What can be moved: the Ulysses row shard. Attention runs on the whole sequence for this rank's head
group, so it is the same work on every rank no matter how the rows are split; everything else in a
block — the five GEMMs, the norms, the SwiGLU, qk-norm/RoPE — is per row and splits arbitrarily, and
`ulysses_custom`'s all-to-all already carries explicit per-peer split sizes because 38247 rows never
divided evenly anyway. Row i's output depends only on row i and the weights, so handing a rank a
different number of rows computes the same numbers; only each GEMM's M changes, which is a
kernel-selection (rounding-order) difference, not an algorithmic one.

A rank's compute time is `T_r = (A_work + c * rows_r) / f_r`, with `A_work` the attention cost and
`c` the per-row cost identical on every rank and `f_r` the clock it runs at. Writing
`R0 = A_work / c` (the attention cost expressed in rows) and noting that the *measured* attention
time is `attn_r = A_work / f_r`,

    T_r  =  attn_r * (R0 + rows_r) / R0

— so timing the attention core is enough to know every rank's whole step, whatever its row count.
`R0` comes from `beta`, the row-proportional share of a rank's work at the even split: Kineto puts
that at (gemm + other) / (attn + gemm + other) = 295/666 = 0.443 at g8 and 596/1337 = 0.446 at g4 —
it barely moves with the GPU count, because both halves scale as 1/world (`H3_BALANCE_BETA`).

**Why the speed is measured in situ, and why the loop is closed.** Two earlier shapes of this failed,
and both failures are the reason for the current one:

  * *Benchmark at startup* (job 21827378): timing the same kernels on every rank at placement time
    saw a 1.3% spread where Kineto sees 5.6% in the real step, with a different ranking every run.
    A third of a second of kernels runs on boost clocks, where the GPUs all look alike.
  * *One open-loop correction* (job 21832646): in situ probing gets the speeds right — the per-rank
    attention time is stable to ~0.5% over 42 forwards and ranks the GPUs the same way Kineto does —
    but applying the full computed shift in one go moved the compute maximum from one rank to
    another and widened the spread (8.1% -> 9.4%), with no change in step time. The node is
    **power-capped**: under load every GPU sits at 691 W with throttle reason 0x4 (SW power cap) at
    57-63 C, and the clock each one picks there is its own silicon's (median 1515-1620 MHz across the
    eight, a 6.9% spread that is exactly the compute spread). In that regime a rank that waits in
    NCCL banks power credit and runs its next kernels faster, so handing rows to a "fast" rank makes
    it slower than the model says.

So the probe is in situ and the controller is closed-loop and damped: at a handful of fixed forwards
it re-reads the attention times *under the current split*, predicts every rank's time with the
identity above, and takes an `ALPHA` fraction of the step towards the mean. Repeated, that finds the
fixed point whatever the power coupling does, which no single correction can.

**Deadlock guard.** Two ranks that disagree about the split sizes hang forever in the next
all-to-all. So (a) the decision to re-balance is taken on a forward counter, never on event
readiness, and (b) the probe times are quantised to int64 nanoseconds before the all-gather and the
split is a pure function of that integer list — a float collective can hand two ranks bit-different
sums and therefore different roundings.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist

BETA = float(os.environ.get("H3_BALANCE_BETA", "0.445"))     # row-proportional share of a rank's work
ALPHA = float(os.environ.get("H3_BALANCE_ALPHA", "0.4"))     # damping: fraction of the step the controller takes
MAXDEV = float(os.environ.get("H3_BALANCE_MAXDEV", "0.25"))  # cap |rows_r - S/world| at this fraction
MINSPREAD = float(os.environ.get("H3_BALANCE_MINSPREAD", "0.01"))   # below this, keep the even split
# Forwards after which to re-derive the split, counting from the first forward of the process. The
# first one is skipped (cold caches, autotuning); the rest refresh it as the node heats up.
UPDATE_AT = tuple(int(x) for x in os.environ.get("H3_BALANCE_AT", "4,8,12,16,20,24,28,32").split(",") if x)
WINDOW = int(os.environ.get("H3_BALANCE_WINDOW", "2"))       # forwards averaged into one update


def even_split(seq: int, world: int) -> list:
    return [seq // world + (1 if r < seq % world else 0) for r in range(world)]


def _integerise(want: list, seq: int) -> list:
    """Floor, then give the remaining rows to the largest remainders (ties by rank index), so every
    rank derives the same list from the same floats."""
    counts = [int(w) for w in want]
    order = sorted(range(len(want)), key=lambda r: (-(want[r] - counts[r]), r))
    for r in order[:seq - sum(counts)]:
        counts[r] += 1
    while sum(counts) > seq:                                 # cannot happen; never ship a bad split
        counts[max(range(len(counts)), key=lambda r: counts[r])] -= 1
    return counts


def est_time(rows: list, attn_ns: list, seq: int, world: int) -> list:
    """Each rank's compute time for the step, in the probe's units.

    A rank's attention cost is `A_work / f_r` and its row cost `c * rows_r / f_r`, so with
    `R0 = A_work / c` (the attention cost expressed in rows) the measured attention time scales the
    whole thing: `T_r = attn_r * (R0 + rows_r) / R0`. `R0` follows from BETA at the even split.
    Checked against Kineto (job 21832646, g8, balanced profile): this ranks the eight ranks' measured
    compute in the right order, max and min included.
    """
    r0 = (seq / float(world)) * (1.0 - BETA) / BETA
    return [a * (r0 + r) for a, r in zip(attn_ns, rows)]


def split(seq: int, world: int, rows: list, attn_ns: list) -> list:
    """One damped step of the controller: rows that make every rank's estimated time the mean.

    Open loop does not work. The node is power-capped — every GPU sits at 691 W with throttle reason
    0x4 and a clock its own silicon picks (1515-1620 MHz across the eight, job 21832646) — so a rank
    that waits in NCCL banks power credit and runs its next kernels faster. Taking rows off the
    slowest rank therefore makes the ranks that receive them slower than the model predicts, and a
    single full-size correction just moves the maximum to another rank (measured: compute max 642.5
    -> 645.0, spread 8.1% -> 9.4%). A damped step, re-measured and repeated, is the only form of this
    that can find the fixed point, because it reads the speeds *as they are under the current split*.
    """
    if ALPHA <= 0.0 or min(attn_ns) <= 0:
        return even_split(seq, world)
    t = est_time(rows, attn_ns, seq, world)
    if (max(t) - min(t)) / min(t) < MINSPREAD:
        return list(rows)

    r0 = (seq / float(world)) * (1.0 - BETA) / BETA
    target = sum(t) / world
    want = [r + ALPHA * ((target / a) - r0 - r) for r, a in zip(rows, attn_ns)]
    shift = (sum(want) - seq) / float(world)                 # back to exactly `seq` rows
    base = seq / float(world)
    lo, hi = base * (1.0 - MAXDEV), base * (1.0 + MAXDEV)
    return _integerise([min(hi, max(lo, w - shift)) for w in want], seq)


class Balancer:
    """Times the attention core on every rank and re-sizes the row shards from what it sees.

    `measure_only` (H3_BALANCE_PROBE=1) keeps the even split and just reports, which is how the
    question "is the per-rank spread a stable property of the GPU or run-to-run noise?" gets
    answered without changing any numbers.
    """

    def __init__(self, world: int, rank: int, group=None, measure_only: bool = False):
        self.world, self.rank, self.group = world, rank, group
        self.measure_only = measure_only
        self.forwards = 0
        self.sizes: dict = {}            # seq -> current row counts
        self.pairs: list = []            # event pairs of the forward being recorded
        self.recent: list = []           # per-forward total attention ms on this rank
        self.history: list = []          # (forward, [per-rank ms]) for the report
        self._a = None

    # --- probe -------------------------------------------------------------------------------
    def begin(self):
        self._a = torch.cuda.Event(enable_timing=True)
        self._a.record()

    def end(self):
        b = torch.cuda.Event(enable_timing=True)
        b.record()
        self.pairs.append((self._a, b))

    # --- per-forward bookkeeping -------------------------------------------------------------
    def open_forward(self):
        self.pairs = []

    def close_forward(self):
        n = self.forwards
        self.forwards += 1
        # Both tests are on the forward counter alone, so every rank reads its events and enters the
        # collective at exactly the same forward. Forward 0 is outside every window on purpose:
        # cold caches and Triton/cuBLAS first-call work make it unrepresentative.
        if any(u - WINDOW < n <= u for u in UPDATE_AT):
            torch.cuda.synchronize()      # the only sync this adds, and only inside a window
            self.recent.append(sum(a.elapsed_time(b) for a, b in self.pairs))
        self.pairs = []
        if n in UPDATE_AT and self.recent:
            self._update(n)
            self.recent = []

    def _update(self, n: int):
        mine = sum(self.recent) / len(self.recent)
        buf = torch.empty(self.world, dtype=torch.int64, device="cuda")
        dist.all_gather_into_tensor(
            buf, torch.tensor([round(mine * 1e6)], dtype=torch.int64, device="cuda"), group=self.group)
        ns = buf.tolist()
        self.history.append((n, [round(x / 1e6, 2) for x in ns]))
        before = dict(self.sizes)
        if not self.measure_only:
            self.sizes = {seq: split(seq, self.world, rows, ns) for seq, rows in self.sizes.items()}
        if self.rank == 0:
            print(f"[h3-h100] {self.report(n, ns, before)}", flush=True)

    # --- what forward_mp asks for ------------------------------------------------------------
    def rows(self, seq: int) -> list:
        sizes = self.sizes.get(seq)
        if sizes is None:
            sizes = self.sizes[seq] = even_split(seq, self.world)
        return sizes

    def report(self, n: int, ns: list, before: dict) -> str:
        ms = [x / 1e6 for x in ns]
        seq = next(iter(self.sizes), 0)
        rows_before = before.get(seq, [])
        rows = self.sizes.get(seq, [])
        even = seq // self.world if seq else 0
        dev = [round(100.0 * (c - even) / even, 1) for c in rows] if even else []
        # est_T is what the controller is actually trying to flatten: the attention probe scaled by
        # the row count the rank was carrying while it was probed.
        t = est_time(rows_before, ns, seq, self.world) if rows_before else ms
        tm = sum(t) / len(t)
        return (f"balance@{n}: attn_ms {[round(x, 1) for x in ms]} "
                f"(spread {100.0 * (max(ms) - min(ms)) / min(ms):.1f}%); "
                f"est_T rel {[round(x / tm, 4) for x in t]} "
                f"(spread {100.0 * (max(t) - min(t)) / min(t):.1f}%); "
                f"rows {rows} (even {even}, dev {dev}%)"
                + (" [probe only]" if self.measure_only else ""))


def make(world: int, rank: int, group=None):
    """A Balancer if H3_BALANCE=1 (balance) or H3_BALANCE_PROBE=1 (measure and report), else None."""
    if world < 2:
        return None
    on = os.environ.get("H3_BALANCE", "0") == "1"
    probe = os.environ.get("H3_BALANCE_PROBE", "0") == "1"
    if not (on or probe):
        return None
    return Balancer(world, rank, group, measure_only=not on)
