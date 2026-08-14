"""Placeholders for the FP4 / captured-graph paths, which are not enabled here.

They raise rather than no-op: if the cache ever reaches these, the run is doing
something the configuration did not ask for, and that must surface as an error
rather than as a quietly different measurement.
"""


class _Raiser:
    def __init__(self, what):
        self._what = what

    def __getattr__(self, name):
        raise RuntimeError(
            f"{self._what}.{name} reached, but that path is disabled in this port"
        )


class BlockGraphStub(_Raiser):
    STATS = {"captured": 0, "replays": 0, "replays_bf16": 0, "bypass": 0, "fail": None}

    def __init__(self):
        super().__init__("block_graph")


class NVFP4Stub(_Raiser):
    GUARD = {"step": -1, "n_bf16": 0, "n_fp4": 0}

    def __init__(self):
        super().__init__("nvfp4")

    def step_is_dense(self):
        return True
