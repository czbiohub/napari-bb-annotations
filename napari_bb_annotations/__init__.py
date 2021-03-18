try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .bb_annotations import main


__all__ = ["main"]
