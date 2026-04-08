"""
numpy2 - Advanced NumPy for Web Applications
============================================

A powerful library that bridges NumPy and web frameworks, solving critical pain points:
- JSON serialization of NumPy types (int64, float64, etc.)
- Automatic type conversion for web APIs
- FastAPI/Flask/Django integration
- Zero-configuration setup
- Production-ready performance

Build by: Mahesh Makvana
GitHub: https://github.com/maheshmakvana/numpy2

Example:
    >>> import numpy2 as np2
    >>> arr = np2.array([1, 2, 3])
    >>> json_safe = np2.to_json(arr)
    >>> fastapi_response = np2.serialize(arr)  # Ready for FastAPI JSONResponse
"""

__version__ = "1.0.0"
__author__ = "Mahesh Makvana"
__email__ = "mahesh@example.com"
__license__ = "MIT"

from .core import (
    array,
    ndarray,
    to_json,
    from_json,
    serialize,
    deserialize,
    JSONEncoder,
    JSONDecoder,
)

from .converters import (
    numpy_to_python,
    pandas_to_json,
    python_to_numpy,
)

from .integrations import (
    FastAPIResponse,
    FlaskResponse,
    DjangoResponse,
    setup_json_encoder,
)

__all__ = [
    "array",
    "ndarray",
    "to_json",
    "from_json",
    "serialize",
    "deserialize",
    "JSONEncoder",
    "JSONDecoder",
    "numpy_to_python",
    "pandas_to_json",
    "python_to_numpy",
    "FastAPIResponse",
    "FlaskResponse",
    "DjangoResponse",
    "setup_json_encoder",
]
