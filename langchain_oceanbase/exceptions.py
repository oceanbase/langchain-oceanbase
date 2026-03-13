"""Custom exceptions for langchain-oceanbase.

This module provides a hierarchy of custom exceptions that include
helpful error messages with troubleshooting hints.
"""

# Troubleshooting documentation URL
TROUBLESHOOTING_URL = "https://github.com/oceanbase/langchain-oceanbase#troubleshooting"


class OceanBaseError(Exception):
    """Base exception for all langchain-oceanbase errors.

    All custom exceptions in this package inherit from this class,
    making it easy to catch any langchain-oceanbase specific error.

    Example:
        >>> try:
        ...     # some operation
        ... except OceanBaseError as e:
        ...     print(f"OceanBase error: {e}")
    """

    pass


class OceanBaseConnectionError(OceanBaseError):
    """Raised when connection to OceanBase fails.

    Common causes:
        - OceanBase server is not running
        - Wrong host or port
        - Network connectivity issues
        - Authentication failure

    Example:
        >>> raise OceanBaseConnectionError(
        ...     "Failed to connect to OceanBase at localhost:2881. "
        ...     "Check that OceanBase is running and accessible."
        ... )
    """

    def __init__(
        self,
        message: str = "Failed to connect to OceanBase.",
        host: str | None = None,
        port: str | int | None = None,
    ):
        """Initialize the connection error.

        Args:
            message: The error message.
            host: The host that was being connected to.
            port: The port that was being connected to.
        """
        if host and port:
            full_message = (
                f"{message} "
                f"Host: {host}, Port: {port}. "
                f"Check that OceanBase is running: docker ps | grep oceanbase. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        else:
            full_message = f"{message} See: {TROUBLESHOOTING_URL}"
        super().__init__(full_message)
        self.host = host
        self.port = port


class OceanBaseVectorDimensionError(OceanBaseError):
    """Raised when vector dimensions don't match.

    Common causes:
        - Embedding model changed after table creation
        - Wrong embedding_dim parameter specified
        - Mixing embeddings from different models

    Example:
        >>> raise OceanBaseVectorDimensionError(
        ...     expected_dim=384,
        ...     actual_dim=768,
        ... )
    """

    def __init__(
        self,
        message: str = "Vector dimension mismatch.",
        expected_dim: int | None = None,
        actual_dim: int | None = None,
    ):
        """Initialize the dimension error.

        Args:
            message: The error message.
            expected_dim: The expected vector dimension.
            actual_dim: The actual vector dimension received.
        """
        if expected_dim and actual_dim:
            full_message = (
                f"{message} "
                f"Expected dimension: {expected_dim}, got: {actual_dim}. "
                f"Check your embedding model's output dimension. "
                f"If the model changed, recreate the table with drop_old=True. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        else:
            full_message = (
                f"{message} "
                f"Check your embedding model's output dimension. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        super().__init__(full_message)
        self.expected_dim = expected_dim
        self.actual_dim = actual_dim


class OceanBaseIndexError(OceanBaseError):
    """Raised when index creation or operation fails.

    Common causes:
        - Insufficient memory for index creation
        - OceanBase version doesn't support the index type
        - Invalid index parameters
        - Index already exists

    Example:
        >>> raise OceanBaseIndexError(
        ...     "Failed to create HNSW index.",
        ...     index_type="HNSW",
        ... )
    """

    def __init__(
        self,
        message: str = "Index operation failed.",
        index_type: str | None = None,
    ):
        """Initialize the index error.

        Args:
            message: The error message.
            index_type: The type of index that failed.
        """
        hints = [
            "Check available memory.",
            "Verify OceanBase version supports this index type.",
            "Try a different index type (FLAT for small datasets).",
        ]
        if index_type:
            full_message = (
                f"{message} "
                f"Index type: {index_type}. "
                f"{' '.join(hints)} "
                f"See: {TROUBLESHOOTING_URL}"
            )
        else:
            full_message = f"{message} {' '.join(hints)} See: {TROUBLESHOOTING_URL}"
        super().__init__(full_message)
        self.index_type = index_type


class OceanBaseVersionError(OceanBaseError):
    """Raised when OceanBase version doesn't support a feature.

    Common causes:
        - AI Functions require OceanBase 4.4.1 or later
        - Certain index types require specific versions
        - New features not available in older versions

    Example:
        >>> raise OceanBaseVersionError(
        ...     "AI Functions",
        ...     required_version="4.4.1",
        ...     current_version="4.3.0",
        ... )
    """

    def __init__(
        self,
        feature: str = "This feature",
        required_version: str | None = None,
        current_version: str | None = None,
    ):
        """Initialize the version error.

        Args:
            feature: The feature that requires a newer version.
            required_version: The minimum required version.
            current_version: The current OceanBase version.
        """
        if required_version and current_version:
            message = (
                f"{feature} requires OceanBase {required_version} or later. "
                f"Current version: {current_version}. "
                f"Please upgrade OceanBase. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        elif required_version:
            message = (
                f"{feature} requires OceanBase {required_version} or later. "
                f"Please upgrade OceanBase. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        else:
            message = (
                f"{feature} is not supported in your OceanBase version. "
                f"Please check version requirements. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        super().__init__(message)
        self.feature = feature
        self.required_version = required_version
        self.current_version = current_version


class OceanBaseConfigurationError(OceanBaseError):
    """Raised when configuration is invalid or missing.

    Common causes:
        - Missing required parameters
        - Invalid parameter values
        - Incompatible configuration options

    Example:
        >>> raise OceanBaseConfigurationError(
        ...     "embedding_dim must be specified when embedding_function is None."
        ... )
    """

    def __init__(
        self,
        message: str = "Invalid configuration.",
        parameter: str | None = None,
        valid_values: list | None = None,
    ):
        """Initialize the configuration error.

        Args:
            message: The error message.
            parameter: The parameter that has an invalid value.
            valid_values: List of valid values for the parameter.
        """
        if parameter and valid_values:
            full_message = (
                f"{message} "
                f"Parameter: {parameter}. "
                f"Valid values: {valid_values}. "
                f"See: {TROUBLESHOOTING_URL}"
            )
        elif parameter:
            full_message = (
                f"{message} Parameter: {parameter}. See: {TROUBLESHOOTING_URL}"
            )
        else:
            full_message = f"{message} See: {TROUBLESHOOTING_URL}"
        super().__init__(full_message)
        self.parameter = parameter
        self.valid_values = valid_values
