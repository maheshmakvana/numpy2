"""
numpy2.advanced — Advanced array operations and utilities.

New in 2.1.0:
- ArrayCache: LRU cache for expensive array computations
- ArrayPipeline: Chainable transformation pipeline
- compress_array / decompress_array: zlib-based array compression
- ArrayValidator: Schema-based array validation
- sliding_window_view: Efficient sliding window without copies
- batch_apply: Apply a function across batches of rows
- to_structured: Convert ndarray + field names to structured-dict list
- ProfiledArray: Transparent profiling wrapper
"""
from __future__ import annotations

import json
import math
import time
import zlib
import base64
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

from .array import ndarray, asarray, zeros


# ---------------------------------------------------------------------------
# LRU Array Cache
# ---------------------------------------------------------------------------

class _CacheEntry:
    __slots__ = ("value", "hits", "created_at", "size_bytes")

    def __init__(self, value, size_bytes: int = 0):
        self.value = value
        self.hits = 0
        self.created_at = time.time()
        self.size_bytes = size_bytes


class ArrayCache:
    """
    Thread-safe LRU cache for memoising expensive array computations.

    Parameters
    ----------
    maxsize : int
        Maximum number of entries to keep (default 128).
    ttl : float | None
        Time-to-live in seconds; None means no expiry.

    Example
    -------
    >>> cache = ArrayCache(maxsize=64)
    >>> @cache.memoize
    ... def expensive(x):
    ...     return x * 2
    >>> expensive(np2.array([1, 2, 3]))
    """

    def __init__(self, maxsize: int = 128, ttl: Optional[float] = None) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    def _make_key(self, *args, **kwargs) -> str:
        parts = []
        for a in args:
            if isinstance(a, ndarray):
                parts.append(f"arr:{a.tolist()!r}:{a.dtype}")
            else:
                parts.append(repr(a))
        for k, v in sorted(kwargs.items()):
            parts.append(f"{k}={v!r}")
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            entry = self._store[key]
            if self._ttl is not None and (time.time() - entry.created_at) > self._ttl:
                del self._store[key]
                self._misses += 1
                return None
            self._store.move_to_end(key)
            entry.hits += 1
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any) -> None:
        size = 0
        if isinstance(value, ndarray):
            size = len(value._data) * 8
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
                self._store[key] = _CacheEntry(value, size)
                return
            if len(self._store) >= self._maxsize:
                self._store.popitem(last=False)
            self._store[key] = _CacheEntry(value, size)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    @property
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._store),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total else 0.0,
            }

    def memoize(self, func: Callable) -> Callable:
        """Decorator: cache results of *func* based on its arguments."""
        def wrapper(*args, **kwargs):
            key = self._make_key(*args, **kwargs)
            cached = self.get(key)
            if cached is not None:
                return cached
            result = func(*args, **kwargs)
            self.set(key, result)
            return result
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper


# ---------------------------------------------------------------------------
# Array Pipeline
# ---------------------------------------------------------------------------

class ArrayPipeline:
    """
    Chainable, lazy transformation pipeline for ndarrays.

    Example
    -------
    >>> pipe = ArrayPipeline(np2.array([1, 2, 3, 4, 5]))
    >>> result = pipe.filter(lambda x: x > 2).map(lambda x: x ** 2).run()
    """

    def __init__(self, arr: ndarray) -> None:
        self._source = asarray(arr)
        self._steps: List[Tuple[str, Callable]] = []

    def map(self, func: Callable) -> "ArrayPipeline":
        """Apply element-wise function."""
        self._steps.append(("map", func))
        return self

    def filter(self, predicate: Callable) -> "ArrayPipeline":
        """Keep elements where predicate returns True."""
        self._steps.append(("filter", predicate))
        return self

    def normalize(self) -> "ArrayPipeline":
        """Min-max normalize to [0, 1]."""
        def _norm(arr):
            mn = min(arr._data)
            mx = max(arr._data)
            rng = mx - mn
            if rng == 0:
                return zeros(len(arr._data), dtype='float64')
            return ndarray([(v - mn) / rng for v in arr._data], dtype='float64')
        self._steps.append(("transform", _norm))
        return self

    def clip(self, lo: float, hi: float) -> "ArrayPipeline":
        """Clip values to [lo, hi]."""
        self._steps.append(("map", lambda x: max(lo, min(hi, x))))
        return self

    def round(self, decimals: int = 0) -> "ArrayPipeline":
        """Round values to *decimals* decimal places."""
        self._steps.append(("map", lambda x: round(x, decimals)))
        return self

    def transform(self, func: Callable) -> "ArrayPipeline":
        """Apply a function to the entire array."""
        self._steps.append(("transform", func))
        return self

    def run(self) -> ndarray:
        """Execute all pipeline steps and return final ndarray."""
        arr = self._source
        for step_type, func in self._steps:
            if step_type == "map":
                arr = ndarray([func(v) for v in arr._data], dtype=arr.dtype)
            elif step_type == "filter":
                arr = ndarray([v for v in arr._data if func(v)], dtype=arr.dtype)
            elif step_type == "transform":
                arr = func(arr)
        return arr


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------

def compress_array(arr: ndarray, level: int = 6) -> bytes:
    """
    Compress an ndarray to bytes using zlib.

    Returns a self-describing byte payload that includes shape, dtype, and data.

    Example
    -------
    >>> blob = np2.compress_array(np2.array([1.0, 2.0, 3.0]))
    >>> arr = np2.decompress_array(blob)
    """
    arr = asarray(arr)
    payload = {
        "shape": list(arr.shape),
        "dtype": arr.dtype.name,
        "data": arr.tolist(),
        "version": 1,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode()
    compressed = zlib.compress(raw, level=level)
    return compressed


def decompress_array(blob: bytes) -> ndarray:
    """
    Decompress bytes produced by compress_array back to an ndarray.
    """
    raw = zlib.decompress(blob)
    payload = json.loads(raw.decode())
    return ndarray(payload["data"], dtype=payload.get("dtype"))


def compress_to_b64(arr: ndarray, level: int = 6) -> str:
    """Compress array and encode as a base64 string (safe for JSON transport)."""
    return base64.b64encode(compress_array(arr, level)).decode()


def decompress_from_b64(b64_str: str) -> ndarray:
    """Decompress a base64-encoded compressed array."""
    return decompress_array(base64.b64decode(b64_str.encode()))


# ---------------------------------------------------------------------------
# Array Validator
# ---------------------------------------------------------------------------

class ArrayValidationError(Exception):
    pass


class ArrayValidator:
    """
    Validate ndarrays against a set of declarative constraints.

    Example
    -------
    >>> v = ArrayValidator(dtype='float64', min_val=0.0, max_val=1.0, ndim=1)
    >>> v.validate(np2.array([0.2, 0.5, 0.8]))  # passes silently
    >>> v.validate(np2.array([-1.0]))  # raises ArrayValidationError
    """

    def __init__(
        self,
        dtype: Optional[str] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        shape: Optional[Tuple[int, ...]] = None,
        ndim: Optional[int] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
    ) -> None:
        self._dtype = dtype
        self._min_val = min_val
        self._max_val = max_val
        self._shape = shape
        self._ndim = ndim
        self._min_size = min_size
        self._max_size = max_size
        self._allow_nan = allow_nan
        self._allow_inf = allow_inf

    def validate(self, arr: ndarray) -> None:
        """Raise ArrayValidationError if any constraint is violated."""
        errors = self.check(arr)
        if errors:
            raise ArrayValidationError("; ".join(errors))

    def check(self, arr: ndarray) -> List[str]:
        """Return list of violated constraint descriptions (empty = valid)."""
        arr = asarray(arr)
        errors: List[str] = []

        if self._dtype and arr.dtype.name != self._dtype:
            errors.append(f"dtype must be '{self._dtype}', got '{arr.dtype.name}'")

        if self._ndim is not None and arr.ndim != self._ndim:
            errors.append(f"ndim must be {self._ndim}, got {arr.ndim}")

        if self._shape is not None and tuple(arr.shape) != tuple(self._shape):
            errors.append(f"shape must be {self._shape}, got {tuple(arr.shape)}")

        if self._min_size is not None and arr.size < self._min_size:
            errors.append(f"size must be >= {self._min_size}, got {arr.size}")

        if self._max_size is not None and arr.size > self._max_size:
            errors.append(f"size must be <= {self._max_size}, got {arr.size}")

        for v in arr._data:
            if isinstance(v, float):
                if not self._allow_nan and math.isnan(v):
                    errors.append("array contains NaN (allow_nan=False)")
                    break
                if not self._allow_inf and math.isinf(v):
                    errors.append("array contains Inf (allow_inf=False)")
                    break

        if self._min_val is not None:
            for v in arr._data:
                if isinstance(v, (int, float)) and not math.isnan(v) and v < self._min_val:
                    errors.append(f"all values must be >= {self._min_val}")
                    break

        if self._max_val is not None:
            for v in arr._data:
                if isinstance(v, (int, float)) and not math.isnan(v) and v > self._max_val:
                    errors.append(f"all values must be <= {self._max_val}")
                    break

        return errors


# ---------------------------------------------------------------------------
# Sliding window view
# ---------------------------------------------------------------------------

def sliding_window_view(arr: ndarray, window_size: int, step: int = 1) -> ndarray:
    """
    Return a 2-D array of sliding windows over a 1-D input array.

    Parameters
    ----------
    arr : ndarray (1-D)
        Input array.
    window_size : int
        Number of elements in each window.
    step : int
        Number of elements to advance per step (default 1).

    Example
    -------
    >>> np2.sliding_window_view(np2.array([1,2,3,4,5]), window_size=3)
    [[1,2,3],[2,3,4],[3,4,5]]
    """
    arr = asarray(arr)
    if arr.ndim != 1:
        raise ValueError("sliding_window_view requires a 1-D array")
    n = arr.size
    if window_size > n:
        raise ValueError(f"window_size ({window_size}) exceeds array size ({n})")
    data = arr._data
    windows = []
    for start in range(0, n - window_size + 1, step):
        windows.append(list(data[start: start + window_size]))
    return ndarray(windows, dtype=arr.dtype)


# ---------------------------------------------------------------------------
# Batch apply
# ---------------------------------------------------------------------------

def batch_apply(
    arr: ndarray,
    func: Callable,
    batch_size: int = 256,
    axis: int = 0,
) -> ndarray:
    """
    Apply *func* to consecutive batches of rows along *axis*.

    Useful for processing large arrays without loading everything into memory.

    Example
    -------
    >>> arr = np2.arange(1000).reshape(100, 10)
    >>> result = np2.batch_apply(arr, lambda b: b * 2, batch_size=25)
    """
    arr = asarray(arr)
    if arr.ndim < 2:
        # 1-D: split into chunks
        n = arr.size
        results = []
        for start in range(0, n, batch_size):
            chunk = ndarray(arr._data[start: start + batch_size], dtype=arr.dtype)
            results.extend(asarray(func(chunk))._data)
        return ndarray(results, dtype=arr.dtype)

    rows = arr.shape[0]
    result_rows = []
    for start in range(0, rows, batch_size):
        batch_data = arr._data[start * arr.shape[1]: (start + batch_size) * arr.shape[1]]
        batch_shape = (min(batch_size, rows - start),) + arr.shape[1:]
        batch = ndarray.__new__(ndarray)
        batch._data = list(batch_data)
        batch._shape = list(batch_shape)
        batch._dtype = arr._dtype
        out = asarray(func(batch))
        result_rows.extend(out._data)
    result = ndarray.__new__(ndarray)
    result._data = result_rows
    result._shape = [rows] + list(arr.shape[1:])
    result._dtype = arr._dtype
    return result


# ---------------------------------------------------------------------------
# to_structured
# ---------------------------------------------------------------------------

def to_structured(arr: ndarray, field_names: List[str]) -> List[Dict[str, Any]]:
    """
    Convert a 2-D ndarray to a list of dicts with named fields.

    Example
    -------
    >>> arr = np2.array([[1, 2], [3, 4]])
    >>> np2.to_structured(arr, ['x', 'y'])
    [{'x': 1, 'y': 2}, {'x': 3, 'y': 4}]
    """
    arr = asarray(arr)
    if arr.ndim != 2:
        raise ValueError("to_structured requires a 2-D array")
    n_cols = arr.shape[1]
    if len(field_names) != n_cols:
        raise ValueError(
            f"field_names length ({len(field_names)}) must match number of columns ({n_cols})"
        )
    rows = []
    data = arr._data
    for r in range(arr.shape[0]):
        row = {field_names[c]: data[r * n_cols + c] for c in range(n_cols)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# ProfiledArray
# ---------------------------------------------------------------------------

class ProfiledArray:
    """
    Transparent profiling wrapper around ndarray.

    Records every operation (method call) with timing.

    Example
    -------
    >>> pa = ProfiledArray(np2.array([1, 2, 3]))
    >>> _ = pa.mean()
    >>> pa.report()
    """

    def __init__(self, arr: ndarray) -> None:
        self._arr = asarray(arr)
        self._log: List[Dict[str, Any]] = []

    def __getattr__(self, name: str):
        attr = getattr(self._arr, name)
        if callable(attr):
            def _timed(*args, **kwargs):
                t0 = time.perf_counter()
                result = attr(*args, **kwargs)
                elapsed = (time.perf_counter() - t0) * 1000
                self._log.append({"op": name, "ms": round(elapsed, 4)})
                return result
            return _timed
        return attr

    def report(self) -> List[Dict[str, Any]]:
        """Return a list of {op, ms} dicts recorded so far."""
        total = sum(e["ms"] for e in self._log)
        print(f"ProfiledArray — {len(self._log)} ops, total {total:.3f} ms")
        for e in self._log:
            print(f"  {e['op']:30s} {e['ms']:.4f} ms")
        return self._log

    def clear_profile(self) -> None:
        """Clear recorded profiling data."""
        self._log.clear()


# ---------------------------------------------------------------------------
# Chunk generator (streaming)
# ---------------------------------------------------------------------------

def array_chunks(arr: ndarray, chunk_size: int) -> Generator[ndarray, None, None]:
    """
    Yield successive chunks of a 1-D ndarray as separate ndarrays.

    Useful for streaming large arrays without loading all into memory.

    Example
    -------
    >>> for chunk in np2.array_chunks(big_array, 100):
    ...     process(chunk)
    """
    arr = asarray(arr)
    n = arr.size
    data = arr._data
    for start in range(0, n, chunk_size):
        yield ndarray(data[start: start + chunk_size], dtype=arr.dtype)


# ---------------------------------------------------------------------------
# describe()  (summary statistics)
# ---------------------------------------------------------------------------

def describe(arr: ndarray) -> Dict[str, float]:
    """
    Return a dict of descriptive statistics for a 1-D ndarray.

    Keys: count, mean, std, min, p25, p50, p75, max

    Example
    -------
    >>> np2.describe(np2.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    {'count': 5, 'mean': 3.0, 'std': 1.414..., 'min': 1.0, ...}
    """
    arr = asarray(arr)
    data = [v for v in arr._data if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(data)
    if n == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"),
                "min": float("nan"), "p25": float("nan"), "p50": float("nan"),
                "p75": float("nan"), "max": float("nan")}
    data_sorted = sorted(data)
    mean = sum(data_sorted) / n
    variance = sum((x - mean) ** 2 for x in data_sorted) / n
    std = math.sqrt(variance)

    def _percentile(sorted_data, p):
        idx = (len(sorted_data) - 1) * p / 100
        lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
        return sorted_data[lo] + (sorted_data[hi] - sorted_data[lo]) * (idx - lo)

    return {
        "count": n,
        "mean": mean,
        "std": std,
        "min": data_sorted[0],
        "p25": _percentile(data_sorted, 25),
        "p50": _percentile(data_sorted, 50),
        "p75": _percentile(data_sorted, 75),
        "max": data_sorted[-1],
    }
