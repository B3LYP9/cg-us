"""cg-us: umbrella-sampling wrapper around CHAPERONg for protein-peptide complexes."""

__version__ = "0.19.2"

from .config import Protocol
from .manifest import Entry, read_manifest

__all__ = ["Protocol", "Entry", "read_manifest", "__version__"]
