try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .io_hooks import napari_get_reader
from . import _key_bindings

__all__ = ["napari_get_reader"]
del _key_bindings
