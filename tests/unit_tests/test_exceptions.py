"""Unit tests for custom exceptions.

These tests verify that the custom exception classes work correctly
and provide helpful error messages with troubleshooting hints.
"""

import pytest

from langchain_oceanbase.exceptions import (
    TROUBLESHOOTING_URL,
    OceanBaseConfigurationError,
    OceanBaseConnectionError,
    OceanBaseError,
    OceanBaseIndexError,
    OceanBaseVectorDimensionError,
    OceanBaseVersionError,
)


class TestOceanBaseError:
    """Tests for the base OceanBaseError exception."""

    def test_base_exception_inheritance(self):
        """Test that OceanBaseError inherits from Exception."""
        assert issubclass(OceanBaseError, Exception)

    def test_base_exception_message(self):
        """Test that OceanBaseError can be raised with a message."""
        with pytest.raises(OceanBaseError) as exc_info:
            raise OceanBaseError("Test error message")

        assert "Test error message" in str(exc_info.value)

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from OceanBaseError."""
        assert issubclass(OceanBaseConnectionError, OceanBaseError)
        assert issubclass(OceanBaseVectorDimensionError, OceanBaseError)
        assert issubclass(OceanBaseIndexError, OceanBaseError)
        assert issubclass(OceanBaseVersionError, OceanBaseError)
        assert issubclass(OceanBaseConfigurationError, OceanBaseError)


class TestOceanBaseConnectionError:
    """Tests for OceanBaseConnectionError."""

    def test_default_message(self):
        """Test default error message."""
        error = OceanBaseConnectionError()
        assert "Failed to connect" in str(error)
        assert TROUBLESHOOTING_URL in str(error)

    def test_message_with_host_and_port(self):
        """Test error message includes host and port when provided."""
        error = OceanBaseConnectionError(
            "Connection failed",
            host="localhost",
            port="2881"
        )
        error_str = str(error)

        assert "localhost" in error_str
        assert "2881" in error_str
        assert "docker ps" in error_str  # Troubleshooting hint
        assert TROUBLESHOOTING_URL in error_str

    def test_attributes_stored(self):
        """Test that host and port are stored as attributes."""
        error = OceanBaseConnectionError(
            "Test",
            host="myhost",
            port="3306"
        )
        assert error.host == "myhost"
        assert error.port == "3306"

    def test_can_be_caught_as_base_exception(self):
        """Test that OceanBaseConnectionError can be caught as OceanBaseError."""
        with pytest.raises(OceanBaseError):
            raise OceanBaseConnectionError("Test")


class TestOceanBaseVectorDimensionError:
    """Tests for OceanBaseVectorDimensionError."""

    def test_default_message(self):
        """Test default error message."""
        error = OceanBaseVectorDimensionError()
        assert "dimension" in str(error).lower()
        assert TROUBLESHOOTING_URL in str(error)

    def test_message_with_dimensions(self):
        """Test error message includes expected and actual dimensions."""
        error = OceanBaseVectorDimensionError(
            "Dimension mismatch",
            expected_dim=384,
            actual_dim=768
        )
        error_str = str(error)

        assert "384" in error_str
        assert "768" in error_str
        assert "drop_old=True" in error_str  # Troubleshooting hint
        assert TROUBLESHOOTING_URL in error_str

    def test_attributes_stored(self):
        """Test that dimensions are stored as attributes."""
        error = OceanBaseVectorDimensionError(
            "Test",
            expected_dim=384,
            actual_dim=768
        )
        assert error.expected_dim == 384
        assert error.actual_dim == 768


class TestOceanBaseIndexError:
    """Tests for OceanBaseIndexError."""

    def test_default_message(self):
        """Test default error message."""
        error = OceanBaseIndexError()
        assert "index" in str(error).lower()
        assert TROUBLESHOOTING_URL in str(error)

    def test_message_with_index_type(self):
        """Test error message includes index type when provided."""
        error = OceanBaseIndexError(
            "Failed to create index",
            index_type="HNSW"
        )
        error_str = str(error)

        assert "HNSW" in error_str
        assert "memory" in error_str.lower()  # Troubleshooting hint
        assert "FLAT" in error_str  # Alternative suggestion
        assert TROUBLESHOOTING_URL in error_str

    def test_attribute_stored(self):
        """Test that index_type is stored as attribute."""
        error = OceanBaseIndexError("Test", index_type="IVF_FLAT")
        assert error.index_type == "IVF_FLAT"


class TestOceanBaseVersionError:
    """Tests for OceanBaseVersionError."""

    def test_default_message(self):
        """Test default error message."""
        error = OceanBaseVersionError()
        assert "not supported" in str(error).lower()
        assert TROUBLESHOOTING_URL in str(error)

    def test_message_with_versions(self):
        """Test error message includes version information."""
        error = OceanBaseVersionError(
            feature="AI Functions",
            required_version="4.4.1",
            current_version="4.3.0"
        )
        error_str = str(error)

        assert "AI Functions" in error_str
        assert "4.4.1" in error_str
        assert "4.3.0" in error_str
        assert "upgrade" in error_str.lower()
        assert TROUBLESHOOTING_URL in error_str

    def test_message_without_current_version(self):
        """Test error message when only required version is provided."""
        error = OceanBaseVersionError(
            feature="Vector Index",
            required_version="4.3.0"
        )
        error_str = str(error)

        assert "Vector Index" in error_str
        assert "4.3.0" in error_str
        assert TROUBLESHOOTING_URL in error_str

    def test_attributes_stored(self):
        """Test that version info is stored as attributes."""
        error = OceanBaseVersionError(
            feature="Test Feature",
            required_version="1.0.0",
            current_version="0.9.0"
        )
        assert error.feature == "Test Feature"
        assert error.required_version == "1.0.0"
        assert error.current_version == "0.9.0"


class TestOceanBaseConfigurationError:
    """Tests for OceanBaseConfigurationError."""

    def test_default_message(self):
        """Test default error message."""
        error = OceanBaseConfigurationError()
        assert "configuration" in str(error).lower()
        assert TROUBLESHOOTING_URL in str(error)

    def test_message_with_parameter(self):
        """Test error message includes parameter name."""
        error = OceanBaseConfigurationError(
            "Invalid value",
            parameter="vidx_metric_type"
        )
        error_str = str(error)

        assert "vidx_metric_type" in error_str
        assert TROUBLESHOOTING_URL in error_str

    def test_message_with_valid_values(self):
        """Test error message includes valid values when provided."""
        error = OceanBaseConfigurationError(
            "Invalid metric type",
            parameter="vidx_metric_type",
            valid_values=["l2", "cosine", "inner_product"]
        )
        error_str = str(error)

        assert "vidx_metric_type" in error_str
        assert "l2" in error_str
        assert "cosine" in error_str
        assert "inner_product" in error_str
        assert TROUBLESHOOTING_URL in error_str

    def test_attributes_stored(self):
        """Test that parameter and valid_values are stored as attributes."""
        valid_values = ["a", "b", "c"]
        error = OceanBaseConfigurationError(
            "Test",
            parameter="test_param",
            valid_values=valid_values
        )
        assert error.parameter == "test_param"
        assert error.valid_values == valid_values


class TestExceptionCatching:
    """Tests for exception catching patterns."""

    def test_catch_all_oceanbase_errors(self):
        """Test that all custom exceptions can be caught with OceanBaseError."""
        exceptions = [
            OceanBaseConnectionError("test"),
            OceanBaseVectorDimensionError("test"),
            OceanBaseIndexError("test"),
            OceanBaseVersionError("test"),
            OceanBaseConfigurationError("test"),
        ]

        for exc in exceptions:
            with pytest.raises(OceanBaseError):
                raise exc

    def test_catch_specific_exception(self):
        """Test that specific exceptions can be caught individually."""
        with pytest.raises(OceanBaseConnectionError):
            raise OceanBaseConnectionError("test")

        with pytest.raises(OceanBaseVersionError):
            raise OceanBaseVersionError("test")

    def test_exception_chaining(self):
        """Test that exceptions can be chained with 'from'."""
        original_error = ValueError("Original error")

        with pytest.raises(OceanBaseConfigurationError) as exc_info:
            try:
                raise original_error
            except ValueError as e:
                raise OceanBaseConfigurationError("Wrapped error") from e

        assert exc_info.value.__cause__ is original_error


class TestTroubleshootingURL:
    """Tests for troubleshooting URL presence."""

    def test_url_is_github_link(self):
        """Test that the troubleshooting URL points to GitHub."""
        assert "github.com" in TROUBLESHOOTING_URL
        assert "troubleshooting" in TROUBLESHOOTING_URL.lower()

    def test_all_exceptions_include_url(self):
        """Test that all exception messages include the troubleshooting URL."""
        exceptions = [
            OceanBaseConnectionError(),
            OceanBaseVectorDimensionError(),
            OceanBaseIndexError(),
            OceanBaseVersionError(),
            OceanBaseConfigurationError(),
        ]

        for exc in exceptions:
            assert TROUBLESHOOTING_URL in str(exc), \
                f"{type(exc).__name__} should include troubleshooting URL"
