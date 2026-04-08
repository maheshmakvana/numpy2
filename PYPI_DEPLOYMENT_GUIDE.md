# PyPI Deployment Guide for numpy2

## Prerequisites

Before deploying to PyPI, ensure you have:

1. **PyPI Account**: https://pypi.org/account/register/
2. **API Token**: Create one at https://pypi.org/manage/account/tokens/
3. **Test PyPI Account (Optional)**: https://test.pypi.org/account/register/

## Step-by-Step Deployment

### 1. Create PyPI Configuration File

Create `~/.pypirc` (Unix/Mac) or `%APPDATA%\pip\pip.ini` (Windows):

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = __token__
password = pypi_YOUR_API_TOKEN_HERE

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi_YOUR_TEST_TOKEN_HERE
```

### 2. Install Required Tools

```bash
source venv/Scripts/activate
pip install twine setuptools wheel
```

### 3. Build Distribution Files

```bash
python setup.py sdist bdist_wheel
```

This creates:
- `dist/numpy2-1.0.0.tar.gz` (source distribution)
- `dist/numpy2-1.0.0-py3-none-any.whl` (wheel)

### 4. Check Distribution Quality (Optional but Recommended)

```bash
twine check dist/*
```

### 5. Test Upload (Optional, Recommended First Time)

```bash
twine upload --repository testpypi dist/*
```

Then test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ numpy2
```

### 6. Production Upload to PyPI

```bash
twine upload dist/*
```

You'll be prompted for your credentials (or use the API token from `.pypirc`)

### 7. Verify Deployment

Visit: https://pypi.org/project/numpy2/

Install from PyPI:
```bash
pip install numpy2
```

## File Structure for Deployment

```
numpy2/
├── numpy2/                  # Main package
├── tests/                   # Tests
├── setup.py                 # Package configuration
├── pyproject.toml          # Modern Python packaging
├── README.md               # Long description for PyPI
├── LICENSE                 # License file
├── MANIFEST.in            # Files to include in distribution
├── CONTRIBUTING.md        # Contribution guidelines
├── dist/                  # Distribution files (created by build)
│   ├── numpy2-1.0.0.tar.gz
│   └── numpy2-1.0.0-py3-none-any.whl
└── .gitignore
```

## Metadata Best Practices

Our setup includes:

✅ **Project Metadata**
- Clear description
- Author: Mahesh Makvana
- Email: mahesh.makvana@example.com
- License: MIT

✅ **SEO Keywords**
- numpy, json, serialization
- fastapi, flask, django
- web, api, pandas, data-science

✅ **Project URLs**
- GitHub repository
- Issue tracker
- Documentation links

✅ **Classifiers**
- Development status
- Intended audience
- Programming language versions
- Topic categorization

## After Deployment

### Update GitHub

1. Create repository on GitHub: https://github.com/new
2. Clone locally: `git clone https://github.com/maheshmakvana/numpy2.git`
3. Push from local repo:
   ```bash
   git remote add origin https://github.com/maheshmakvana/numpy2.git
   git branch -M main
   git push -u origin main
   ```

### Update Documentation

1. Set up GitHub Pages (optional)
2. Create Wiki with:
   - API Reference
   - Framework Integration Guides
   - Examples
   - Troubleshooting

### Release Management

For future releases:

```bash
# Update version in setup.py and numpy2/__init__.py
# Commit changes
git add -A
git commit -m "Release v1.0.1"
git tag v1.0.1

# Build new distribution
python setup.py sdist bdist_wheel

# Upload
twine upload dist/numpy2-1.0.1*
```

## Troubleshooting

### "twine: command not found"
```bash
source venv/Scripts/activate
pip install twine
```

### "Invalid distribution"
Run: `twine check dist/*`

### "Unauthorized (HTTP 401)"
- Check API token in `.pypirc`
- Use `__token__` as username
- Use full token as password

### "Package already registered"
- Update version number in setup.py
- Build and upload new version

## Security

🔒 **Best Practices:**

1. Use API tokens instead of passwords
2. Keep `.pypirc` file private (add to `.gitignore`)
3. Never commit secrets to Git
4. Rotate tokens regularly
5. Use protected branches for releases

## Quick Reference Commands

```bash
# Activate virtual environment
source venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Build distributions
python setup.py sdist bdist_wheel

# Check distribution quality
twine check dist/*

# Upload to TestPyPI (first time)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*

# Clean up build artifacts
rm -rf build/ dist/ *.egg-info/
```

---

**Ready to deploy? Follow the steps above and numpy2 will be available on PyPI!**

Built by: Mahesh Makvana  
GitHub: https://github.com/maheshmakvana/numpy2
