try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from . import _key_bindings


del _key_bindings
