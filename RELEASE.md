# Creating a Release

This document explains how to create a new release of lobpy that will be automatically published to PyPI.

## Versioning

lobpy uses [Semantic Versioning](https://semver.org/):
- **MAJOR** version: incompatible API changes
- **MINOR** version: backwards-compatible functionality additions
- **PATCH** version: backwards-compatible bug fixes

Examples:
- `v1.0.0` - First stable release
- `v1.1.0` - New features added
- `v1.1.1` - Bug fix
- `v2.0.0` - Breaking changes

## Steps to Create a Release

### 1. Update the Version

Update the version in `lobpy/__init__.py`:
```python
__version__ = "1.0.0"
```

Also update the version in `pyproject.toml`:
```toml
version = "1.0.0"
```

And in `setup.py`:
```python
version='1.0.0',
```

### 2. Commit the Changes

```bash
git add lobpy/__init__.py pyproject.toml setup.py
git commit -m "Bump version to 1.0.0"
```

### 3. Create and Push the Tag

```bash
# Create an annotated tag
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push the tag to GitHub
git push origin v1.0.0
```

### 4. Create the GitHub Release

1. Go to https://github.com/mattermat/lob.py/releases
2. Click "Draft a new release"
3. Choose the tag you just pushed (e.g., `v1.0.0`)
4. Add release notes describing the changes
5. Click "Publish release"

### 5. Automatic PyPI Publication

Once the release is published on GitHub, the GitHub Actions workflow will:
1. Build the package (source distribution and wheel)
2. Automatically publish to PyPI

No manual intervention required!

## Testing Before Release

Before creating a production release, you can test locally:

```bash
# Build the package
python -m build

# Test installing the wheel
pip install dist/lobpy-*-py3-none-any.whl

# Test the package
python -c "import lobpy; print(lobpy.__version__)"
python -c "from lobpy import LoB; l = LoB(); print(l.name)"
```

## Checklist Before Release

- [ ] Version updated in `lobpy/__init__.py`
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `setup.py`
- [ ] All tests pass
- [ ] README.md is up to date
- [ ] CHANGELOG.md is updated (if you have one)
- [ ] Tag follows semantic versioning
- [ ] Release notes are prepared

## PyPI Setup (First Time Only)

Before publishing, you need to set up trusted publishing on PyPI:

1. Go to https://pypi.org/manage/account/publishing/
2. Add a new publisher:
   - **PyPI project name**: `lobpy`
   - **Owner**: `mattermat` (your GitHub username)
   - **Repository name**: `lob.py`
   - **Workflow name**: `publish.yml`
   - **Environment name**: leave empty

This enables OIDC/Trusted Publishing - no API tokens needed!

## PyPI Page

After publication, your package will be available at:
https://pypi.org/project/lobpy/
