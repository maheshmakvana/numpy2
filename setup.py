"""
numpy2 - Advanced NumPy for Web Applications

Setup configuration for PyPI deployment.
Built by: Mahesh Makvana
GitHub: https://github.com/maheshmakvana/numpy2
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="numpy2",
    version="2.1.0",
    author="Mahesh Makvana",
    author_email="mahesh.makvana@example.com",
    description="Pure-Python NumPy drop-in: full NumPy API + JSON serialization, array compression, pipeline transforms, schema validation, zero dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/maheshmakvana/numpy2",
    project_urls={
        "Bug Tracker": "https://github.com/maheshmakvana/numpy2/issues",
        "Documentation": "https://github.com/maheshmakvana/numpy2/wiki",
        "Source Code": "https://github.com/maheshmakvana/numpy2",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Internet :: WWW/HTTP",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
    ],
    python_requires=">=3.8",
    install_requires=[],   # zero required dependencies — pure Python
    extras_require={
        "numpy": ["numpy>=1.20.0"],    # optional accelerator
        "pandas": ["pandas>=1.3.0"],   # optional pandas interop
        "fastapi": ["fastapi>=0.95.0", "starlette>=0.26.0"],
        "flask": ["Flask>=2.0.0"],
        "django": ["Django>=3.2"],
        "full": ["numpy>=1.20.0", "pandas>=1.3.0"],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.990",
        ],
    },
    keywords=[
        "numpy",
        "numpy2",
        "numpy drop-in",
        "numpy replacement",
        "pure python numpy",
        "json serialization",
        "numpy json",
        "fastapi numpy",
        "flask numpy",
        "django numpy",
        "web api numpy",
        "array compression",
        "array cache",
        "array pipeline",
        "array validation",
        "sliding window",
        "batch processing numpy",
        "streaming array",
        "pandas",
        "data-science",
        "type-conversion",
        "rest-api",
        "ndarray",
        "linear algebra",
        "fft",
        "random",
        "numerical computing",
        "scientific computing",
        "int64 json",
        "numpy serializable",
        "object of type int64 is not json serializable",
        "numpy web framework",
        "numpy fastapi",
    ],
    zip_safe=False,
)
