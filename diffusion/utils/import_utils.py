import importlib.util
import logging
import warnings

import importlib_metadata
from packaging import version

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

_triton_modules_available = importlib.util.find_spec("triton") is not None
try:
    if _triton_modules_available:
        _triton_version = importlib_metadata.version("triton")
        if version.Version(_triton_version) < version.Version("3.0.0"):
            raise ValueError("triton is installed but requires Triton >= 3.0.0")
        logger.debug(f"Successfully imported triton version {_triton_version}")
except ImportError:
    _triton_modules_available = False
    warnings.warn("TritonLiteMLA and TritonMBConvPreGLU with `triton` is not available on your platform.")


def is_xformers_available():
    return _xformers_available


def is_triton_module_available():
    return _triton_modules_available


import inspect
import warnings
from typing import Any, Dict, Optional, Union

from packaging import version


def deprecate(*args, take_from: Optional[Union[Dict, Any]] = None, standard_warn=True, stacklevel=2):
    from .. import __version__

    deprecated_kwargs = take_from
    values = ()
    if not isinstance(args[0], tuple):
        args = (args,)

    for attribute, version_name, message in args:
        if version.parse(version.parse(__version__).base_version) >= version.parse(version_name):
            raise ValueError(
                f"The deprecation tuple {(attribute, version_name, message)} should be removed since sana's"
                f" version {__version__} is >= {version_name}"
            )

        warning = None
        if isinstance(deprecated_kwargs, dict) and attribute in deprecated_kwargs:
            values += (deprecated_kwargs.pop(attribute),)
            warning = f"The `{attribute}` argument is deprecated and will be removed in version {version_name}."
        elif hasattr(deprecated_kwargs, attribute):
            values += (getattr(deprecated_kwargs, attribute),)
            warning = f"The `{attribute}` attribute is deprecated and will be removed in version {version_name}."
        elif deprecated_kwargs is None:
            warning = f"`{attribute}` is deprecated and will be removed in version {version_name}."

        if warning is not None:
            warning = warning + " " if standard_warn else ""
            warnings.warn(warning + message, FutureWarning, stacklevel=stacklevel)

    if isinstance(deprecated_kwargs, dict) and len(deprecated_kwargs) > 0:
        call_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = call_frame.filename
        line_number = call_frame.lineno
        function = call_frame.function
        key, value = next(iter(deprecated_kwargs.items()))
        raise TypeError(f"{function} in {filename} line {line_number-1} got an unexpected keyword argument `{key}`")

    if len(values) == 0:
        return
    elif len(values) == 1:
        return values[0]
    return values
