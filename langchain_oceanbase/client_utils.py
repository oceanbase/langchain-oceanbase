"""Client utility functions for detecting and validating client types."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Try to import pyobvector for detection (optional, may not be installed)
try:
    import pyobvector

    PYOBVECTOR_AVAILABLE = True
except ImportError:
    pyobvector = None  # type: ignore
    PYOBVECTOR_AVAILABLE = False


def is_pyobvector_client(client: Any) -> bool:
    """
    Check if the client is from pyobvector library.

    Args:
        client: Client object to check

    Returns:
        bool: True if client is from pyobvector, False otherwise
    """
    if not PYOBVECTOR_AVAILABLE:
        return False

    # Check by module name
    client_module = getattr(client, "__module__", None)
    if client_module and "pyobvector" in client_module:
        return True

    # Check by class name
    client_class = getattr(client, "__class__", None)
    if client_class:
        class_name = getattr(client_class, "__name__", "")
        class_module = getattr(client_class, "__module__", "")
        if "pyobvector" in class_module or "obvector" in class_name.lower():
            return True

    # Check by type comparison if pyobvector is available
    try:
        if pyobvector and isinstance(client, (pyobvector.Client, pyobvector.OBVector)):
            return True
    except (AttributeError, TypeError):
        pass

    return False


def check_and_warn_pyobvector_deprecation(client: Any) -> None:
    """
    Check if client is from pyobvector and warn about deprecation.

    This function checks if the provided client is from pyobvector library
    and issues a deprecation warning if it is. Users should migrate to
    pyseekdb.Client instead.

    Args:
        client: Client object to check

    Example:
        >>> from langchain_oceanbase.client_utils import check_and_warn_pyobvector_deprecation
        >>> check_and_warn_pyobvector_deprecation(client)
    """
    if is_pyobvector_client(client):
        logger.warning(
            "⚠️  Deprecation Warning: pyobvector client is detected. "
            "The current version of langchain-oceanbase has migrated to pyseekdb.Client. "
            "Please update your code to use pyseekdb.Client instead. "
            "Support for pyobvector will be removed in a future version.\n"
            "Migration guide:\n"
            "  Old: from pyobvector import OBVector\n"
            "  New: from pyseekdb import Client\n"
            "For more information, see: https://github.com/langchain-ai/langchain"
        )


def check_pyobvector_in_kwargs(**kwargs: Any) -> Any:
    """
    Check if kwargs contains a pyobvector client and warn about deprecation.

    This function checks if 'client' or 'obvector' is provided in kwargs
    and if it's from pyobvector, issues a deprecation warning.

    Args:
        **kwargs: Keyword arguments that may contain 'client' or 'obvector'

    Returns:
        The client from kwargs if found, None otherwise

    Example:
        >>> client = check_pyobvector_in_kwargs(client=some_client)
    """
    # Check for 'client' or 'obvector' in kwargs
    client = kwargs.get("client") or kwargs.get("obvector")

    if client is not None:
        check_and_warn_pyobvector_deprecation(client)
        return client

    return None
