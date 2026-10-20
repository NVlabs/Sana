"""ComfyUI entry; model frameworks stay in the selected runtime process."""
from .sol_h3_engine.comfy_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

try:
    # Studio routes need ComfyUI's server; CPU tests import this package without it.
    from .sol_h3_engine.comfy_routes import register as _register_routes
    _register_routes()
except ImportError:
    pass
except Exception as _error:
    import logging
    logging.warning("[Sol-H3] Studio routes not registered: %s", _error)

WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
