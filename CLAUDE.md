# numpy2 — Claude Code Context

## What this package does

Drop-in NumPy enhancement for web apps. Solves `TypeError: Object of type int64 is not JSON serializable` and related NumPy-in-web-API pain points.

**Install:** `pip install numpy2`  
**Usage:** `import numpy2 as np2`

## Architecture

**numpy2 is 100% pure Python — NumPy is NOT required.**
NumPy is used as an optional accelerator only if installed.

## Source files (only read/edit these)

| File | Responsibility |
|------|---------------|
| `numpy2/dtypes.py` | Pure-Python dtype system — `dtype`, all dtype singletons, `_normalise`, `_infer_dtype_from_data` |
| `numpy2/array.py` | Pure-Python `ndarray` class + all array creation functions (`zeros`, `ones`, `arange`, `linspace`, `concatenate`, `where`, `isnan`, etc.) |
| `numpy2/math_ops.py` | Pure-Python ufuncs and math — `sin`, `cos`, `sqrt`, `sum`, `mean`, `std`, `cov`, `histogram`, etc. |
| `numpy2/linalg.py` | Pure-Python linear algebra — `det`, `inv`, `solve`, `svd`, `eig`, `qr`, `norm`, etc. |
| `numpy2/fft.py` | Pure-Python FFT (Cooley-Tukey) — `fft`, `ifft`, `rfft`, `fft2`, `fftfreq`, etc. |
| `numpy2/random.py` | Pure-Python RNG — `rand`, `randn`, `randint`, `normal`, `seed`, `Generator`, etc. |
| `numpy2/core.py` | JSON serialization — `JSONEncoder`, `JSONDecoder`, `to_json`, `from_json`, `serialize`, `deserialize` |
| `numpy2/converters.py` | Type conversion helpers — `numpy_to_python`, `pandas_to_json`, `infer_dtype`, `safe_cast`, `batch_convert` |
| `numpy2/integrations.py` | Web framework helpers — `FastAPIResponse`, `FlaskResponse`, `DjangoResponse`, `setup_json_encoder` |
| `numpy2/__init__.py` | Public API — imports everything from all modules above |
| `tests/test_core.py` | Full test suite (57 tests, 0 numpy required) |
| `setup.py` | Package config (name=numpy2, version=2.0.0, no required deps) |
| `README.md` | PyPI description |

## Ignore completely

`venv/`, `.git/`, `dist/`, `__pycache__/`, `*.egg-info/`

## Dependencies

- **Required:** none (pure Python 3.8+ stdlib only)
- **Optional:** `numpy` (accelerator), `pandas` (interop), `fastapi`, `flask`, `django`
- Install: `pip install numpy2` — no extras needed
