"""
numpy2 - Drop-in NumPy replacement with Web Superpowers
========================================================

numpy2 is a 100% compatible drop-in replacement for NumPy.
Replace every ``import numpy as np`` with ``import numpy2 as np``
and everything works identically — PLUS you get built-in JSON
serialization, web framework integration, and type-safe conversions.

Built by: Mahesh Makvana
GitHub: https://github.com/maheshmakvana/numpy2

Drop-in usage:
    >>> import numpy2 as np          # replaces: import numpy as np
    >>> arr = np.array([1, 2, 3])    # all numpy functions work
    >>> arr.mean()                    # all numpy methods work
    >>> np.linalg.inv(matrix)        # all submodules work
    >>> json_str = np.to_json(arr)   # NEW: instant JSON serialization

Zero migration cost:
    Just change your import line. Nothing else breaks.
"""

__version__ = "2.0.0"
__author__ = "Mahesh Makvana"
__email__ = "mahesh.makvana@example.com"
__license__ = "MIT"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Re-export the ENTIRE NumPy public API
#     `from numpy import *` pulls in everything listed in numpy.__all__
#     so any code doing `import numpy2 as np` gets full NumPy behaviour.
# ─────────────────────────────────────────────────────────────────────────────
from numpy import *  # noqa: F401, F403

import numpy as _np  # internal reference used below

# Expose numpy itself as an attribute (e.g. numpy2.numpy.linalg)
numpy = _np

# ── Submodules ────────────────────────────────────────────────────────────────
# These must be imported explicitly; `from numpy import *` does NOT include them.
from numpy import linalg     # noqa: F401
from numpy import fft        # noqa: F401
from numpy import random     # noqa: F401
from numpy import ma         # noqa: F401
from numpy import polynomial # noqa: F401
from numpy import testing    # noqa: F401
from numpy import lib        # noqa: F401
from numpy import char       # noqa: F401

# numpy.strings exists in NumPy 2.x; fall back silently on older versions
try:
    from numpy import strings  # noqa: F401
except ImportError:
    pass

# numpy.exceptions exists in NumPy 1.25+; fall back silently
try:
    from numpy import exceptions  # noqa: F401
except ImportError:
    pass

# ── Extra top-level names that numpy exposes but may not be in __all__ ────────
from numpy import (  # noqa: F401
    # Index / grid helpers
    c_, r_, s_, ix_, ogrid, mgrid, index_exp, ndindex,
    # Functional
    frompyfunc,
    # Type aliases
    int_, intp, intc,
    bool_, object_, str_, bytes_,
    # nditer
    nditer, ndenumerate,
    # Error handling
    seterr, geterr, errstate, seterrcall, geterrcall,
    # IO helpers
    savetxt,
    # Polynomial (legacy numpy.poly* API)
    poly, poly1d, polyadd, polysub, polymul, polydiv,
    polyder, polyint, polyval, polyfit,
)

# ─────────────────────────────────────────────────────────────────────────────
# 2.  numpy2 Web Extras — new features on top of NumPy
#     These additions make numpy2 superior to bare NumPy for web development.
# ─────────────────────────────────────────────────────────────────────────────
from .core import (           # noqa: F401
    to_json,
    from_json,
    serialize,
    deserialize,
    JSONEncoder,
    JSONDecoder,
    # Re-import our enhanced ndarray wrapper AFTER the star import so it
    # takes precedence over numpy.ndarray for numpy2 users who call
    # np2.ndarray([1,2,3]) and expect the web-enabled wrapper object.
    ndarray as ndarray,  # noqa: PLC0414
)

from .converters import (     # noqa: F401
    numpy_to_python,
    pandas_to_json,
    python_to_numpy,
    infer_dtype,
    safe_cast,
    batch_convert,
)

from .integrations import (   # noqa: F401
    FastAPIResponse,
    FlaskResponse,
    DjangoResponse,
    setup_json_encoder,
    create_response_handler,
)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  __all__ — NumPy's full list + numpy2 extras
# ─────────────────────────────────────────────────────────────────────────────
_numpy2_extras = [
    # Web / serialization
    "to_json", "from_json", "serialize", "deserialize",
    "JSONEncoder", "JSONDecoder",
    # Type converters
    "numpy_to_python", "pandas_to_json", "python_to_numpy",
    "infer_dtype", "safe_cast", "batch_convert",
    # Framework integrations
    "FastAPIResponse", "FlaskResponse", "DjangoResponse",
    "setup_json_encoder", "create_response_handler",
    # Submodules
    "linalg", "fft", "random", "ma", "polynomial",
    "testing", "lib", "char", "numpy",
]

__all__ = list(getattr(_np, "__all__", [])) + _numpy2_extras
