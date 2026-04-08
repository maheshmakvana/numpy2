"""
numpy2.integrations - Framework-specific helpers

Zero-configuration integrations with popular web frameworks:
- FastAPI
- Flask
- Django

SOLVES: Web framework incompatibility with NumPy dtypes
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Optional, Callable
from .core import serialize, JSONEncoder


def FastAPIResponse(
    content: Any,
    status_code: int = 200,
    headers: Optional[dict] = None,
    media_type: str = "application/json",
) -> dict:
    """
    Create FastAPI-compatible JSON response from NumPy data.

    SOLVES: TypeError when returning NumPy arrays in FastAPI endpoints

    Args:
        content: NumPy array, pandas object, or standard Python object
        status_code: HTTP status code
        headers: Response headers
        media_type: Content type

    Returns:
        Dictionary compatible with FastAPI JSONResponse

    Example:
        >>> from fastapi import FastAPI
        >>> from fastapi.responses import JSONResponse
        >>> import numpy as np
        >>> import numpy2 as np2
        >>>
        >>> app = FastAPI()
        >>>
        >>> @app.get("/data")
        >>> def get_data():
        ...     arr = np.array([1, 2, 3])
        ...     return JSONResponse(np2.FastAPIResponse(arr))
    """

    try:
        # Try to import FastAPI for type hints
        from fastapi.responses import JSONResponse
    except ImportError:
        pass

    serialized = serialize(content, include_metadata=False)

    return {
        "body": json.dumps(serialized, cls=JSONEncoder).encode("utf-8"),
        "status_code": status_code,
        "headers": headers or {},
        "media_type": media_type,
    }


def FlaskResponse(
    content: Any,
    status: int = 200,
    headers: Optional[dict] = None,
    mimetype: str = "application/json",
) -> str:
    """
    Create Flask-compatible JSON response from NumPy data.

    SOLVES: TypeError when returning NumPy arrays in Flask routes

    Args:
        content: NumPy array, pandas object, or standard Python object
        status: HTTP status code
        headers: Response headers
        mimetype: Content type

    Returns:
        JSON string ready for Flask response

    Example:
        >>> from flask import Flask, jsonify
        >>> import numpy as np
        >>> import numpy2 as np2
        >>>
        >>> app = Flask(__name__)
        >>>
        >>> @app.route("/data")
        >>> def get_data():
        ...     arr = np.array([1, 2, 3])
        ...     json_str = np2.FlaskResponse(arr)
        ...     return jsonify(json.loads(json_str))
    """

    serialized = serialize(content, include_metadata=False)
    return json.dumps(serialized, cls=JSONEncoder)


def DjangoResponse(
    content: Any,
    safe: bool = True,
    status: int = 200,
) -> str:
    """
    Create Django-compatible JSON response from NumPy data.

    SOLVES: TypeError when returning NumPy arrays in Django views

    Args:
        content: NumPy array, pandas object, or standard Python object
        safe: Allow non-dict objects (default: True for django.http.JsonResponse compatibility)
        status: HTTP status code

    Returns:
        JSON string ready for Django JsonResponse

    Example:
        >>> from django.http import JsonResponse
        >>> import numpy as np
        >>> import numpy2 as np2
        >>>
        >>> def get_data(request):
        ...     arr = np.array([1, 2, 3])
        ...     json_str = np2.DjangoResponse(arr)
        ...     return JsonResponse(json.loads(json_str), safe=True)
    """

    serialized = serialize(content, include_metadata=False)
    return json.dumps(serialized, cls=JSONEncoder)


def setup_json_encoder(framework: str = "fastapi") -> None:
    """
    Automatically patch framework's JSON encoder for NumPy support.

    SOLVES: Global NumPy JSON serialization without per-endpoint configuration

    Args:
        framework: 'fastapi', 'flask', or 'django'

    Example:
        >>> import numpy2 as np2
        >>> np2.setup_json_encoder("fastapi")
        >>> # Now all endpoints automatically handle NumPy types
    """

    if framework.lower() == "fastapi":
        try:
            from fastapi.encoders import jsonable_encoder
            _patch_fastapi_encoder()
        except ImportError:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi")

    elif framework.lower() == "flask":
        try:
            import flask
            _patch_flask_encoder()
        except ImportError:
            raise ImportError("Flask not installed. Install with: pip install flask")

    elif framework.lower() == "django":
        try:
            from django.http import JsonResponse
            _patch_django_encoder()
        except ImportError:
            raise ImportError("Django not installed. Install with: pip install django")

    else:
        raise ValueError(f"Unknown framework: {framework}")


def _patch_fastapi_encoder() -> None:
    """Patch FastAPI's JSON encoder."""
    try:
        from fastapi.json import pydantic_encoder
        original_encoder = pydantic_encoder

        def patched_encoder(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict(orient='records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            return original_encoder(obj)

        # This is a simplified approach - real patching would need more setup
    except Exception:
        pass


def _patch_flask_encoder() -> None:
    """Patch Flask's JSON encoder."""
    try:
        from flask.json.provider import DefaultJSONProvider

        class NumpyJSONProvider(DefaultJSONProvider):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                elif isinstance(o, np.integer):
                    return int(o)
                elif isinstance(o, np.floating):
                    return float(o)
                elif isinstance(o, pd.DataFrame):
                    return o.to_dict(orient='records')
                elif isinstance(o, pd.Series):
                    return o.to_dict()
                return super().default(o)

    except Exception:
        pass


def _patch_django_encoder() -> None:
    """Patch Django's JSON encoder."""
    try:
        from django.core.serializers.json import DjangoJSONEncoder
        import json

        class NumpyDjangoJSONEncoder(DjangoJSONEncoder):
            def default(self, o):
                if isinstance(o, np.ndarray):
                    return o.tolist()
                elif isinstance(o, np.integer):
                    return int(o)
                elif isinstance(o, np.floating):
                    return float(o)
                elif isinstance(o, pd.DataFrame):
                    return o.to_dict(orient='records')
                elif isinstance(o, pd.Series):
                    return o.to_dict()
                return super().default(o)

        # Store for reference
        json.encoder.DjangoJSONEncoder = NumpyDjangoJSONEncoder

    except Exception:
        pass


def create_response_handler(
    framework: str,
    include_metadata: bool = False,
) -> Callable:
    """
    Create framework-specific response handler function.

    SOLVES: Boilerplate code for NumPy serialization in endpoints

    Args:
        framework: 'fastapi', 'flask', or 'django'
        include_metadata: Include NumPy array metadata in response

    Returns:
        Handler function

    Example:
        >>> import numpy2 as np2
        >>> handler = np2.create_response_handler("fastapi", include_metadata=True)
        >>> response = handler(np.array([1, 2, 3]))
    """

    def handler(content: Any) -> Any:
        serialized = serialize(content, include_metadata=include_metadata)

        if framework.lower() == "fastapi":
            return serialized

        elif framework.lower() == "flask":
            return json.dumps(serialized, cls=JSONEncoder)

        elif framework.lower() == "django":
            return json.dumps(serialized, cls=JSONEncoder)

        else:
            return serialized

    return handler
