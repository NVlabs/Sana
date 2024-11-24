import importlib.util
import importlib_metadata
from packaging import version
import logging

logger = logging.getLogger(__name__)

_xformers_available = importlib.util.find_spec("xformers") is not None
try:
    if _xformers_available:
        _xformers_version = importlib_metadata.version("xformers")
        _torch_version = importlib_metadata.version("torch")
        if version.Version(_torch_version) < version.Version("1.12"):
            raise ValueError("xformers is installed but requires PyTorch >= 1.12")
        logger.debug(f"Successfully imported xformers version {_xformers_version}")
except importlib_metadata.PackageNotFoundError:
    _xformers_available = False


def is_xformers_available():
    return _xformers_available