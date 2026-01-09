from importlib.metadata import version

from . import ds, pl, pp, tl

__all__ = ["ds", "pl", "pp", "tl"]

__version__ = version("pygreta")
