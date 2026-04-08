"""Tests for numpy2.core module."""

import pytest
import json
import numpy as np
import pandas as pd
import numpy2 as np2


class TestJSONEncoder:
    """Test custom JSON encoder."""

    def test_int64_serialization(self):
        """Test int64 serialization."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert result == "[1, 2, 3]"

    def test_float64_serialization(self):
        """Test float64 serialization."""
        arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "[1.5, 2.5, 3.5]" in result

    def test_ndarray_serialization(self):
        """Test ndarray serialization."""
        arr = np.array([[1, 2], [3, 4]])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "[[1, 2], [3, 4]]" in result

    def test_dataframe_serialization(self):
        """Test DataFrame serialization."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = json.dumps(df, cls=np2.JSONEncoder)
        assert '"A"' in result and '"B"' in result

    def test_series_serialization(self):
        """Test Series serialization."""
        s = pd.Series([1, 2, 3])
        result = json.dumps(s, cls=np2.JSONEncoder)
        assert isinstance(result, str)

    def test_nan_handling(self):
        """Test NaN handling."""
        arr = np.array([1.0, np.nan, 3.0])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "null" in result or "nan" in result.lower()

    def test_inf_handling(self):
        """Test infinity handling."""
        arr = np.array([1.0, np.inf, 3.0])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert isinstance(result, str)


class TestToJson:
    """Test to_json function."""

    def test_to_json_int_array(self):
        """Test to_json with int array."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = np2.to_json(arr)
        assert result == "[1, 2, 3]"

    def test_to_json_float_array(self):
        """Test to_json with float array."""
        arr = np.array([1.5, 2.5], dtype=np.float32)
        result = np2.to_json(arr)
        assert "1.5" in result and "2.5" in result

    def test_to_json_with_indent(self):
        """Test to_json with indentation."""
        arr = np.array([1, 2, 3])
        result = np2.to_json(arr, indent=2)
        assert "\n" in result

    def test_to_json_dataframe(self):
        """Test to_json with DataFrame."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = np2.to_json(df)
        assert isinstance(result, str)


class TestFromJson:
    """Test from_json function."""

    def test_from_json_basic(self):
        """Test from_json basic deserialization."""
        json_str = "[1, 2, 3]"
        result = np2.from_json(json_str)
        assert result == [1, 2, 3]

    def test_from_json_to_numpy(self):
        """Test from_json to NumPy array."""
        json_str = "[1, 2, 3]"
        result = np2.from_json(json_str, to_numpy=True)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_from_json_with_dtype(self):
        """Test from_json with dtype."""
        json_str = "[1, 2, 3]"
        result = np2.from_json(json_str, to_numpy=True, dtype='int32')
        assert result.dtype == np.int32


class TestSerialize:
    """Test serialize function."""

    def test_serialize_ndarray(self):
        """Test serialize with ndarray."""
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = np2.serialize(arr)
        assert isinstance(result, dict)
        assert 'data' in result
        assert result['data'] == [1, 2, 3]

    def test_serialize_with_metadata(self):
        """Test serialize with metadata."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        result = np2.serialize(arr, include_metadata=True)
        assert result['shape'] == [2, 2]
        assert result['dtype'] == 'int32'
        assert result['size'] == 4

    def test_serialize_dataframe(self):
        """Test serialize with DataFrame."""
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = np2.serialize(df)
        assert isinstance(result, dict)
        assert 'data' in result

    def test_serialize_nested_dict(self):
        """Test serialize with nested structures."""
        data = {
            'arr': np.array([1, 2, 3]),
            'val': 42
        }
        result = np2.serialize(data)
        assert isinstance(result, dict)
        assert result['arr'] == [1, 2, 3]


class TestDeserialize:
    """Test deserialize function."""

    def test_deserialize_to_numpy(self):
        """Test deserialize to numpy array."""
        data = {'data': [1, 2, 3]}
        result = np2.deserialize(data, to_numpy=True)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, np.array([1, 2, 3]))

    def test_deserialize_with_dtype(self):
        """Test deserialize with dtype."""
        data = {'data': [1, 2, 3], 'dtype': 'int32'}
        result = np2.deserialize(data)
        assert result.dtype == np.int32

    def test_deserialize_to_list(self):
        """Test deserialize to list."""
        data = {'data': [1, 2, 3]}
        result = np2.deserialize(data, to_numpy=False)
        assert isinstance(result, list)


class TestArrayFunction:
    """Test array wrapper function."""

    def test_array_creation(self):
        """Test array creation."""
        arr = np2.array([1, 2, 3])
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1, 2, 3]))

    def test_array_with_dtype(self):
        """Test array with dtype."""
        arr = np2.array([1, 2, 3], dtype='int32')
        assert arr.dtype == np.int32


class TestNdarrayWrapper:
    """Test ndarray wrapper class."""

    def test_ndarray_creation(self):
        """Test ndarray wrapper creation."""
        arr = np2.ndarray([1, 2, 3])
        assert isinstance(arr.numpy, np.ndarray)

    def test_ndarray_to_json(self):
        """Test ndarray to_json method."""
        arr = np2.ndarray([1, 2, 3])
        json_str = arr.to_json()
        assert "[1, 2, 3]" in json_str

    def test_ndarray_to_dict(self):
        """Test ndarray to_dict method."""
        arr = np2.ndarray([1, 2, 3])
        result = arr.to_dict()
        assert result['data'] == [1, 2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
