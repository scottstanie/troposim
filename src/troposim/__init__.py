# -*- coding: utf-8 -*-
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("troposim")
except PackageNotFoundError:
    __version__ = "unknown"

# Most used modules should get auto-import
from . import turbulence
