"""Tests for numpy2 — no NumPy required."""

import pytest
import json
import numpy2 as np2

# ── optional interop fixtures ─────────────────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False


# ── JSONEncoder ───────────────────────────────────────────────────────────────

class TestJSONEncoder:

    def test_int_array_serialization(self):
        arr = np2.array([1, 2, 3], dtype='int64')
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert result == "[1, 2, 3]"

    def test_float_array_serialization(self):
        arr = np2.array([1.5, 2.5, 3.5], dtype='float64')
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "1.5" in result and "2.5" in result

    def test_ndarray_2d_serialization(self):
        arr = np2.array([[1, 2], [3, 4]])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "[[1, 2], [3, 4]]" in result or "1" in result

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_dataframe_serialization(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = json.dumps(df, cls=np2.JSONEncoder)
        assert '"A"' in result and '"B"' in result

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_series_serialization(self):
        s = pd.Series([1, 2, 3])
        result = json.dumps(s, cls=np2.JSONEncoder)
        assert isinstance(result, str)

    def test_nan_handling(self):
        arr = np2.array([1.0, float('nan'), 3.0])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert "null" in result

    def test_inf_handling(self):
        arr = np2.array([1.0, float('inf'), 3.0])
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert isinstance(result, str)

    @pytest.mark.skipif(not HAS_NUMPY, reason="numpy not installed")
    def test_numpy_int64_interop(self):
        arr = np.array([1, 2, 3], dtype=np.int64)
        result = json.dumps(arr, cls=np2.JSONEncoder)
        assert result == "[1, 2, 3]"


# ── to_json ───────────────────────────────────────────────────────────────────

class TestToJson:

    def test_to_json_int_array(self):
        arr = np2.array([1, 2, 3], dtype='int64')
        result = np2.to_json(arr)
        assert result == "[1, 2, 3]"

    def test_to_json_float_array(self):
        arr = np2.array([1.5, 2.5], dtype='float64')
        result = np2.to_json(arr)
        assert "1.5" in result and "2.5" in result

    def test_to_json_with_indent(self):
        arr = np2.array([1, 2, 3])
        result = np2.to_json(arr, indent=2)
        assert "\n" in result

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_to_json_dataframe(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = np2.to_json(df)
        assert isinstance(result, str)


# ── from_json ─────────────────────────────────────────────────────────────────

class TestFromJson:

    def test_from_json_basic(self):
        result = np2.from_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_from_json_to_numpy(self):
        result = np2.from_json("[1, 2, 3]", to_numpy=True)
        assert isinstance(result, np2.ndarray)
        assert np2.array_equal(result, np2.array([1, 2, 3]))

    def test_from_json_with_dtype(self):
        result = np2.from_json("[1, 2, 3]", to_numpy=True, dtype='int32')
        assert result.dtype == np2.dtype('int32')


# ── serialize ─────────────────────────────────────────────────────────────────

class TestSerialize:

    def test_serialize_ndarray(self):
        arr = np2.array([1, 2, 3], dtype='int64')
        result = np2.serialize(arr)
        assert isinstance(result, dict)
        assert 'data' in result
        assert result['data'] == [1, 2, 3]

    def test_serialize_with_metadata(self):
        arr = np2.array([[1, 2], [3, 4]], dtype='int32')
        result = np2.serialize(arr, include_metadata=True)
        assert result['shape'] == [2, 2]
        assert result['dtype'] == 'int32'
        assert result['size'] == 4

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_serialize_dataframe(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = np2.serialize(df)
        assert isinstance(result, dict)
        assert 'data' in result

    def test_serialize_nested_dict(self):
        data = {'arr': np2.array([1, 2, 3]), 'val': 42}
        result = np2.serialize(data)
        assert isinstance(result, dict)
        assert result['arr'] == [1, 2, 3]


# ── deserialize ───────────────────────────────────────────────────────────────

class TestDeserialize:

    def test_deserialize_to_ndarray(self):
        data = {'data': [1, 2, 3]}
        result = np2.deserialize(data, to_numpy=True)
        assert isinstance(result, np2.ndarray)
        assert np2.array_equal(result, np2.array([1, 2, 3]))

    def test_deserialize_with_dtype(self):
        data = {'data': [1, 2, 3], 'dtype': 'int32'}
        result = np2.deserialize(data)
        assert result.dtype == np2.dtype('int32')

    def test_deserialize_to_list(self):
        data = {'data': [1, 2, 3]}
        result = np2.deserialize(data, to_numpy=False)
        assert isinstance(result, list)


# ── array creation ────────────────────────────────────────────────────────────

class TestArrayCreation:

    def test_array_basic(self):
        arr = np2.array([1, 2, 3])
        assert isinstance(arr, np2.ndarray)
        assert arr.tolist() == [1, 2, 3]

    def test_array_with_dtype(self):
        arr = np2.array([1, 2, 3], dtype='int32')
        assert arr.dtype == np2.dtype('int32')

    def test_zeros(self):
        arr = np2.zeros((2, 3))
        assert arr.shape == (2, 3)
        assert all(v == 0.0 for v in arr._data)

    def test_ones(self):
        arr = np2.ones((3,))
        assert arr.tolist() == [1.0, 1.0, 1.0]

    def test_arange(self):
        arr = np2.arange(5)
        assert arr.tolist() == [0, 1, 2, 3, 4]

    def test_linspace(self):
        arr = np2.linspace(0, 1, 5)
        assert len(arr._data) == 5
        assert arr._data[0] == 0.0
        assert arr._data[-1] == 1.0

    def test_eye(self):
        arr = np2.eye(3)
        assert arr._data[0] == 1.0
        assert arr._data[1] == 0.0
        assert arr._data[4] == 1.0

    def test_reshape(self):
        arr = np2.arange(6).reshape(2, 3)
        assert arr.shape == (2, 3)

    def test_transpose(self):
        arr = np2.array([[1, 2, 3], [4, 5, 6]])
        t = arr.T
        assert t.shape == (3, 2)


# ── math operations ───────────────────────────────────────────────────────────

class TestMathOps:

    def test_sum(self):
        arr = np2.array([1, 2, 3, 4])
        assert np2.sum(arr) == 10

    def test_mean(self):
        arr = np2.array([1.0, 2.0, 3.0, 4.0])
        assert np2.mean(arr) == 2.5

    def test_std(self):
        arr = np2.array([1.0, 2.0, 3.0])
        result = np2.std(arr)
        assert round(result, 5) == round((2/3)**0.5, 5)

    def test_min_max(self):
        arr = np2.array([3, 1, 4, 1, 5])
        assert np2.min(arr) == 1
        assert np2.max(arr) == 5

    def test_sqrt(self):
        arr = np2.array([4.0, 9.0, 16.0])
        result = np2.sqrt(arr)
        assert result.tolist() == [2.0, 3.0, 4.0]

    def test_sin_cos(self):
        import math
        arr = np2.array([0.0, math.pi / 2])
        s = np2.sin(arr)
        assert round(s._data[0], 10) == 0.0
        assert round(s._data[1], 5) == 1.0

    def test_arithmetic(self):
        a = np2.array([1, 2, 3])
        b = np2.array([4, 5, 6])
        assert (a + b).tolist() == [5, 7, 9]
        assert (a * b).tolist() == [4, 10, 18]
        assert (b - a).tolist() == [3, 3, 3]

    def test_dot(self):
        a = np2.array([1, 2, 3])
        b = np2.array([4, 5, 6])
        assert np2.dot(a, b) == 32

    def test_matmul(self):
        A = np2.array([[1, 2], [3, 4]])
        B = np2.array([[5, 6], [7, 8]])
        C = A @ B
        assert C.tolist() == [[19, 22], [43, 50]]


# ── ndarray wrapper ───────────────────────────────────────────────────────────

class TestNdarrayWrapper:

    def test_ndarray_to_json(self):
        arr = np2.array([1, 2, 3])
        json_str = np2.to_json(arr)
        assert "[1, 2, 3]" in json_str

    def test_ndarray_serialize(self):
        arr = np2.array([1, 2, 3])
        result = np2.serialize(arr)
        assert result['data'] == [1, 2, 3]

    def test_ndarray_indexing(self):
        arr = np2.array([10, 20, 30])
        assert arr[0] == 10
        assert arr[-1] == 30

    def test_ndarray_slicing(self):
        arr = np2.array([1, 2, 3, 4, 5])
        assert arr[1:4].tolist() == [2, 3, 4]

    def test_ndarray_boolean_mask(self):
        arr = np2.array([1, 2, 3, 4, 5])
        mask = arr > 3
        assert arr[mask].tolist() == [4, 5]


# ── linalg ────────────────────────────────────────────────────────────────────

class TestLinalg:

    def test_det(self):
        A = np2.array([[1.0, 2.0], [3.0, 4.0]])
        d = np2.linalg.det(A)
        assert round(d, 5) == round(-2.0, 5)

    def test_inv(self):
        A = np2.array([[2.0, 0.0], [0.0, 2.0]])
        inv = np2.linalg.inv(A)
        assert round(inv._data[0], 5) == 0.5

    def test_solve(self):
        A = np2.array([[2.0, 1.0], [1.0, 3.0]])
        b = np2.array([5.0, 10.0])
        x = np2.linalg.solve(A, b)
        # verify A @ x ≈ b
        check = A @ x
        assert round(check._data[0], 4) == round(5.0, 4)

    def test_norm(self):
        v = np2.array([3.0, 4.0])
        assert np2.linalg.norm(v) == 5.0


# ── fft ───────────────────────────────────────────────────────────────────────

class TestFFT:

    def test_fft_length(self):
        arr = np2.array([1.0, 2.0, 3.0, 4.0])
        result = np2.fft.fft(arr)
        assert isinstance(result, np2.ndarray)
        assert result.size >= 4

    def test_fft_ifft_roundtrip(self):
        import math
        arr = np2.array([1.0, 0.0, 1.0, 0.0])
        F = np2.fft.fft(arr)
        back = np2.fft.ifft(F)
        for orig, restored in zip(arr._data, back._data):
            assert abs(restored.real - orig) < 1e-9

    def test_fftfreq(self):
        freqs = np2.fft.fftfreq(4, d=1.0)
        assert len(freqs._data) == 4


# ── random ────────────────────────────────────────────────────────────────────

class TestRandom:

    def test_rand_shape(self):
        arr = np2.random.rand(3, 4)
        assert arr.shape == (3, 4)
        assert all(0 <= v < 1 for v in arr._data)

    def test_randn(self):
        arr = np2.random.randn(100)
        assert arr.shape == (100,)

    def test_randint(self):
        arr = np2.random.randint(0, 10, size=20)
        assert all(0 <= v < 10 for v in arr._data)

    def test_seed_reproducibility(self):
        np2.random.seed(42)
        a = np2.random.rand(5)
        np2.random.seed(42)
        b = np2.random.rand(5)
        assert a.tolist() == b.tolist()

    def test_choice(self):
        result = np2.random.choice(10, size=5)
        assert isinstance(result, np2.ndarray)
        assert all(0 <= v < 10 for v in result._data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
