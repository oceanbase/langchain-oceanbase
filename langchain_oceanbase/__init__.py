from importlib import metadata

from langchain_oceanbase.vectorstores import OceanbaseVectorStore

try:
    from langchain_oceanbase.checkpoint.saver import OceanBaseSaver
except ImportError:
    OceanBaseSaver = None  # type: ignore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "OceanbaseVectorStore",
    "OceanBaseSaver",
    "__version__",
]
