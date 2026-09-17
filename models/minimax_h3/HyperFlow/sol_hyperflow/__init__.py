"""HyperFlow integration with the shared Sol-H3 inference kernels."""

__all__ = ["HyperFlowInference"]


def __getattr__(name):
    if name == "HyperFlowInference":
        from .engine import HyperFlowInference

        return HyperFlowInference
    raise AttributeError(name)
