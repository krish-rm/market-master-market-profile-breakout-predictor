"""Pytest configuration and shared fixtures."""

import warnings
import pytest

# Filter scikit-learn version compatibility warnings
# These occur when loading models trained with scikit-learn 1.3.2 in 1.4.2+
# The models still work correctly, this is just a version mismatch warning
# Filter by message pattern since the warning type may vary
warnings.filterwarnings(
    "ignore",
    message=".*Trying to unpickle estimator.*from version.*",
    module="sklearn.base"
)
warnings.filterwarnings(
    "ignore",
    message=".*InconsistentVersionWarning.*",
    module="sklearn.base"
)

# Filter pandas FutureWarning about 'H' frequency (we've already fixed it, but keep filter for safety)
warnings.filterwarnings(
    "ignore",
    message=".*'H' is deprecated.*",
    category=FutureWarning,
    module="pandas"
)

# Filter Pydantic deprecation warnings (we've fixed them, but keep filter for any remaining)
warnings.filterwarnings(
    "ignore",
    message=".*Support for class-based `config` is deprecated.*",
    category=DeprecationWarning,
    module="pydantic"
)

warnings.filterwarnings(
    "ignore",
    message=".*'schema_extra' has been renamed to 'json_schema_extra'.*",
    category=UserWarning,
    module="pydantic"
)

