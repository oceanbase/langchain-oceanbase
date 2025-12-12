from importlib import metadata

try:
    from langchain_oceanbase.ai_functions import OceanBaseAIFunctions
except ImportError:
    OceanBaseAIFunctions = None  # type: ignore

try:
    from langchain_oceanbase.chat_message_histories import OceanBaseChatMessageHistory
except ImportError:
    OceanBaseChatMessageHistory = None  # type: ignore

try:
    from langchain_oceanbase.embedding_utils import DefaultEmbeddingFunction
except ImportError:
    DefaultEmbeddingFunction = None  # type: ignore

try:
    from langchain_oceanbase.vectorstores import OceanbaseVectorStore
except ImportError:
    OceanbaseVectorStore = None  # type: ignore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "OceanbaseVectorStore",
    "OceanBaseChatMessageHistory",
    "OceanBaseAIFunctions",
    "DefaultEmbeddingFunction",
    "__version__",
]
