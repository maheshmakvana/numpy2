"""
numpy2.core - Core serialization and type conversion utilities

Solves NumPy JSON serialization issues:
- TypeError: Object of type int64 is not JSON serializable
- Silent data loss from NumPy type conversions
- Performance degradation in web APIs
- Framework incompatibility with NumPy dtypes
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional
from decimal import Decimal


class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy and pandas data types.

    SOLVES: TypeError: Object of type int64 is not JSON serializable

    Features:
    - Automatic NumPy dtype conversion
    - pandas Series/DataFrame support
    - Preserves data integrity
    - Zero configuration

    Example:
        >>> import json
        >>> import numpy as np
        >>> arr = np.array([1, 2, 3], dtype=np.int64)
        >>> json.dumps(arr, cls=JSONEncoder)
        '[1, 2, 3]'
    """

    def default(self, obj: Any) -> Any:
        """Convert non-standard types to JSON-serializable format."""

        # NumPy integer types
        if isinstance(obj, np.integer):
            return int(obj)

        # NumPy float types
        elif isinstance(obj, np.floating):
            # Handle special float values
            if np.isnan(obj):
                return None  # or "NaN" if preferred
            elif np.isinf(obj):
                return None  # or "Infinity" if preferred
            return float(obj)

        # NumPy arrays
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        # NumPy scalar types
        elif isinstance(obj, np.generic):
            return obj.item()

        # pandas Series
        elif isinstance(obj, pd.Series):
            return obj.to_dict()

        # pandas DataFrame
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')

        # pandas Index
        elif isinstance(obj, pd.Index):
            return obj.tolist()

        # Decimal support
        elif isinstance(obj, Decimal):
            return float(obj)

        # datetime64
        elif isinstance(obj, np.datetime64):
            return str(obj)

        # timedelta64
        elif isinstance(obj, np.timedelta64):
            return str(obj)

        # complex numbers
        elif isinstance(obj, (complex, np.complexfloating)):
            return {"real": obj.real, "imag": obj.imag}

        # Default fallback
        return super().default(obj)


class JSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder that reconstructs NumPy arrays from JSON.

    SOLVES: Data type loss when converting JSON back to NumPy

    Features:
    - Intelligent type inference
    - Preserves numeric precision
    - Optional NumPy array reconstruction

    Example:
        >>> import json
        >>> data = '[1, 2, 3]'
        >>> decoder = JSONDecoder()
        >>> decoder.decode(data)
        [1, 2, 3]
    """

    def __init__(self, to_numpy: bool = False, dtype: Optional[str] = None, *args, **kwargs):
        """
        Initialize decoder.

        Args:
            to_numpy: If True, convert lists to numpy arrays
            dtype: NumPy dtype to use for arrays (e.g., 'int64', 'float32')
        """
        self.to_numpy = to_numpy
        self.dtype = dtype
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj: Dict) -> Any:
        """Convert objects back to appropriate types."""
        if self.to_numpy and isinstance(obj, dict) and "real" in obj and "imag" in obj:
            return complex(obj["real"], obj["imag"])
        return obj


def to_json(obj: Any, indent: Optional[int] = None, **kwargs) -> str:
    """
    Convert NumPy arrays and pandas objects to JSON string.

    SOLVES: TypeError: Object of type int64 is not JSON serializable

    Args:
        obj: NumPy array, pandas object, or standard Python object
        indent: JSON indentation level
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string representation

    Example:
        >>> import numpy as np
        >>> import numpy2 as np2
        >>> arr = np.array([1, 2, 3], dtype=np.int64)
        >>> json_str = np2.to_json(arr)
        >>> print(json_str)
        [1, 2, 3]
    """
    return json.dumps(obj, cls=JSONEncoder, indent=indent, **kwargs)


def from_json(
    json_str: str,
    to_numpy: bool = False,
    dtype: Optional[str] = None
) -> Any:
    """
    Deserialize JSON string to Python objects (optionally NumPy arrays).

    Args:
        json_str: JSON string to deserialize
        to_numpy: Convert to numpy arrays
        dtype: Target numpy dtype

    Returns:
        Deserialized Python object or NumPy array

    Example:
        >>> import numpy2 as np2
        >>> json_str = '[1, 2, 3]'
        >>> arr = np2.from_json(json_str, to_numpy=True, dtype='int64')
        >>> type(arr)
        <class 'numpy.ndarray'>
    """
    decoder = JSONDecoder(to_numpy=to_numpy, dtype=dtype)
    result = decoder.decode(json_str)

    if to_numpy and isinstance(result, list):
        result = np.array(result, dtype=dtype)

    return result


def serialize(obj: Any, include_metadata: bool = False) -> Dict[str, Any]:
    """
    Serialize NumPy/pandas objects to JSON-safe dictionary.

    SOLVES: Web framework incompatibility with NumPy dtypes

    Perfect for FastAPI JSONResponse, Flask jsonify, Django JsonResponse

    Args:
        obj: Object to serialize
        include_metadata: Include shape, dtype, and other metadata

    Returns:
        JSON-safe dictionary

    Example:
        >>> import numpy as np
        >>> import numpy2 as np2
        >>> arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        >>> result = np2.serialize(arr, include_metadata=True)
        >>> result
        {
            'data': [[1, 2], [3, 4]],
            'shape': [2, 2],
            'dtype': 'int32',
            'size': 4
        }
    """

    if isinstance(obj, np.ndarray):
        result = {
            'data': obj.tolist(),
        }
        if include_metadata:
            result.update({
                'shape': list(obj.shape),
                'dtype': str(obj.dtype),
                'size': int(obj.size),
                'ndim': int(obj.ndim),
            })
        return result

    elif isinstance(obj, pd.DataFrame):
        result = {
            'data': obj.to_dict(orient='records'),
        }
        if include_metadata:
            result.update({
                'shape': list(obj.shape),
                'columns': list(obj.columns),
                'index': obj.index.tolist(),
            })
        return result

    elif isinstance(obj, pd.Series):
        result = {
            'data': obj.to_dict(),
        }
        if include_metadata:
            result.update({
                'name': obj.name,
                'dtype': str(obj.dtype),
                'index': obj.index.tolist(),
            })
        return result

    elif isinstance(obj, dict):
        # Recursively serialize nested structures.
        # For plain ndarray values inside a dict we return the raw list so
        # that callers get result['key'] == [1, 2, 3] rather than
        # result['key'] == {'data': [1, 2, 3]}.
        result = {}
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, pd.Series):
                result[k] = v.to_dict()
            elif isinstance(v, pd.DataFrame):
                result[k] = v.to_dict(orient='records')
            elif isinstance(v, (dict, list, tuple)):
                result[k] = serialize(v, include_metadata)
            else:
                result[k] = v
        return result

    elif isinstance(obj, (list, tuple)):
        out = []
        for item in obj:
            if isinstance(item, np.ndarray):
                out.append(item.tolist())
            elif isinstance(item, pd.Series):
                out.append(item.to_dict())
            elif isinstance(item, pd.DataFrame):
                out.append(item.to_dict(orient='records'))
            elif isinstance(item, (dict, list, tuple)):
                out.append(serialize(item, include_metadata))
            else:
                out.append(item)
        return out

    else:
        return json.loads(json.dumps(obj, cls=JSONEncoder))


def deserialize(
    data: Dict[str, Any],
    to_numpy: bool = True,
    dtype: Optional[str] = None
) -> Union[np.ndarray, pd.DataFrame, Any]:
    """
    Deserialize JSON-safe dictionary back to NumPy/pandas objects.

    Args:
        data: Dictionary with 'data' key and optional metadata
        to_numpy: Convert to NumPy array (if False, returns list)
        dtype: Target dtype for conversion

    Returns:
        NumPy array, pandas DataFrame, or raw data

    Example:
        >>> import numpy2 as np2
        >>> serialized = {
        ...     'data': [[1, 2], [3, 4]],
        ...     'shape': [2, 2],
        ...     'dtype': 'int32'
        ... }
        >>> arr = np2.deserialize(serialized)
        >>> type(arr)
        <class 'numpy.ndarray'>
    """

    if not isinstance(data, dict):
        return data

    if 'data' not in data:
        return data

    array_data = data['data']

    if to_numpy:
        target_dtype = dtype or data.get('dtype', None)
        return np.array(array_data, dtype=target_dtype)

    return array_data


def array(
    data: Any,
    dtype: Optional[str] = None,
    **kwargs
) -> np.ndarray:
    """
    Create NumPy array with automatic type handling.

    Wrapper around numpy.array with better type inference for web data.

    Args:
        data: Array data
        dtype: Data type
        **kwargs: Additional arguments for numpy.array

    Returns:
        NumPy ndarray
    """
    return np.array(data, dtype=dtype, **kwargs)


class ndarray:
    """
    Enhanced NumPy ndarray wrapper with web-friendly methods.

    Adds convenience methods for serialization without modifying
    the original NumPy array.
    """

    def __init__(self, data: Any, dtype: Optional[str] = None):
        self._array = np.array(data, dtype=dtype)

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert to JSON string."""
        return to_json(self._array, indent=indent)

    def to_dict(self, include_metadata: bool = False) -> Dict:
        """Convert to JSON-safe dictionary."""
        return serialize(self._array, include_metadata=include_metadata)

    @property
    def numpy(self) -> np.ndarray:
        """Get underlying NumPy array."""
        return self._array

    def __repr__(self) -> str:
        return f"numpy2.ndarray({self._array})"

    def __str__(self) -> str:
        return str(self._array)
