from importlib import metadata as _meta

try:
    __version__: str = _meta.version(__name__)
except _meta.PackageNotFoundError:
    __version__ = "0.0.0+dev"


def version() -> str:
    """Return the ChemSTEP version string."""
    return __version__
