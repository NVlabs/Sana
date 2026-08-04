"""SM120 kernel recipe."""

from .mainloop import SolAttnForwardSm120


def make_kernel(*, debug_route_trace: bool = False):
    return SolAttnForwardSm120(debug_route_trace=debug_route_trace)


__all__ = ["make_kernel"]
