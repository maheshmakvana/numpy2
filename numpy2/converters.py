"""
numpy2.converters - Data type conversion utilities

Handles conversions between NumPy, pandas, Python types, and JSON
with automatic type inference and data integrity preservation.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union, Optional


def numpy_to_python(obj: Any) -> Any:
    """
    Convert NumPy types to native Python types.

    SOLVES: Silent data loss from NumPy type conversions

    Args:
        obj: NumPy object to convert

    Returns:
        Native Python type

    Example:
        >>> import numpy as np
        >>> import numpy2 as np2
        >>> val = np.int64(42)
        >>> result = np2.numpy_to_python(val)
        >>> type(result)
        <class 'int'>
    """

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.datetime64):
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return type(obj)(numpy_to_python(item) for item in obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    else:
        return obj


def pandas_to_json(
    df: pd.DataFrame,
    orient: str = 'records',
    include_index: bool = False
) -> Dict[str, Any]:
    """
    Convert pandas DataFrame to JSON-safe dictionary.

    SOLVES: JSON serialization of DataFrames with NumPy columns

    Args:
        df: pandas DataFrame
        orient: DataFrame orientation ('records', 'list', 'dict', 'split', 'tight', 'index', 'columns', 'values')
        include_index: Include index in output

    Returns:
        JSON-safe dictionary

    Example:
        >>> import pandas as pd
        >>> import numpy2 as np2
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3.5, 4.5]})
        >>> result = np2.pandas_to_json(df)
        >>> type(result)
        <class 'dict'>
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(df)}")

    # Convert all NumPy types to Python types
    df_converted = df.copy()

    for col in df_converted.columns:
        if df_converted[col].dtype == 'object':
            continue
        if hasattr(df_converted[col].dtype, 'name'):
            if 'int' in str(df_converted[col].dtype) or 'float' in str(df_converted[col].dtype):
                df_converted[col] = df_converted[col].astype(str).apply(
                    lambda x: int(x) if '.' not in x else float(x)
                )

    result = df_converted.to_dict(orient=orient)

    if include_index and orient == 'records':
        result = [
            {**row, '__index__': idx}
            for idx, row in enumerate(result)
        ]

    return result


def python_to_numpy(
    data: Any,
    dtype: Optional[str] = None
) -> np.ndarray:
    """
    Convert Python types to NumPy array.

    Args:
        data: Python data (list, tuple, dict, etc.)
        dtype: Target NumPy dtype

    Returns:
        NumPy ndarray

    Example:
        >>> import numpy2 as np2
        >>> data = [1, 2, 3]
        >>> arr = np2.python_to_numpy(data, dtype='float32')
        >>> arr.dtype
        dtype('float32')
    """

    return np.array(data, dtype=dtype)


def infer_dtype(data: Any) -> str:
    """
    Intelligently infer appropriate NumPy dtype from data.

    SOLVES: Type inference problems in web APIs

    Args:
        data: Data to analyze

    Returns:
        Inferred NumPy dtype string

    Example:
        >>> import numpy2 as np2
        >>> dtype = np2.infer_dtype([1, 2, 3])
        >>> dtype
        'int64'
    """

    if isinstance(data, (list, tuple)):
        if not data:
            return 'float64'

        # Check first element type
        first = data[0]

        if isinstance(first, bool):
            return 'bool'
        elif isinstance(first, int):
            # Check range to choose appropriate int type
            values = [x for x in data if isinstance(x, int)]
            if all(-128 <= v < 128 for v in values):
                return 'int8'
            elif all(-32768 <= v < 32768 for v in values):
                return 'int16'
            elif all(-2147483648 <= v < 2147483648 for v in values):
                return 'int32'
            else:
                return 'int64'
        elif isinstance(first, float):
            return 'float64'
        elif isinstance(first, str):
            return 'object'
        elif isinstance(first, (list, tuple)):
            return infer_dtype(first)

    elif isinstance(data, dict):
        return 'object'

    elif isinstance(data, np.ndarray):
        return str(data.dtype)

    return 'object'


def safe_cast(
    value: Any,
    target_dtype: str,
    raise_on_error: bool = False
) -> Any:
    """
    Safely cast value to target dtype with error handling.

    SOLVES: Type conversion errors breaking web APIs

    Args:
        value: Value to cast
        target_dtype: Target NumPy dtype
        raise_on_error: Raise exception on failure (default: return original)

    Returns:
        Casted value or original value on failure

    Example:
        >>> import numpy2 as np2
        >>> result = np2.safe_cast("123", 'int32')
        >>> result
        123
    """

    try:
        if target_dtype in ['int8', 'int16', 'int32', 'int64']:
            return int(value)
        elif target_dtype in ['float16', 'float32', 'float64']:
            return float(value)
        elif target_dtype == 'bool':
            return bool(value)
        else:
            return np.array([value], dtype=target_dtype)[0]
    except (ValueError, TypeError) as e:
        if raise_on_error:
            raise
        return value


def batch_convert(
    data: List[Dict[str, Any]],
    dtype_map: Optional[Dict[str, str]] = None
) -> List[Dict[str, Any]]:
    """
    Convert batch of records with consistent type handling.

    SOLVES: Bulk data conversion issues in APIs

    Args:
        data: List of dictionaries to convert
        dtype_map: Mapping of field names to target dtypes

    Returns:
        Converted data

    Example:
        >>> import numpy2 as np2
        >>> data = [{'id': 1, 'value': 3.14}]
        >>> converted = np2.batch_convert(data, {'id': 'int32', 'value': 'float32'})
    """

    if dtype_map is None:
        dtype_map = {}

    converted = []
    for record in data:
        new_record = {}
        for key, value in record.items():
            target_dtype = dtype_map.get(key)
            if target_dtype:
                new_record[key] = safe_cast(value, target_dtype)
            else:
                new_record[key] = numpy_to_python(value)
        converted.append(new_record)

    return converted
