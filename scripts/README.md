# Signal-Kit Setup Scripts

This directory contains utility scripts for setting up and working with signal-kit.rs.

## Scripts Overview

| Script | Purpose |
|--------|---------|
| `setup_python_env.sh` | Create venv, install maturin, pytest, numpy |
| `build_and_test_release.sh` | Build optimized wheels and run full test suite |

---

## setup_python_env.sh

Automated setup script for Python development environment with signal-kit Python bindings.

### Features

- ✅ Creates Python virtual environment
- ✅ Installs maturin for building Python bindings
- ✅ Installs pytest for running tests
- ✅ Installs numpy for array operations
- ✅ Optionally builds signal-kit with maturin
- ✅ Provides helpful next steps

### Prerequisites

- Python 3.8+ installed and in PATH
- Rust toolchain installed (for building maturin)
- Git (to clone the repository)

### Usage

**Basic usage (creates `.venv` directory):**
```bash
./scripts/setup_python_env.sh
```

**Custom venv location:**
```bash
./scripts/setup_python_env.sh ~/my_signal_kit_env
```

The script will:
1. Check for Python installation
2. Create a virtual environment
3. Upgrade pip, setuptools, and wheel
4. Install maturin
5. Install pytest and pytest-cov
6. Install numpy
7. Optionally build signal-kit immediately
8. Print next steps

### What Gets Installed

| Package | Purpose |
|---------|---------|
| maturin | Build and package Rust Python bindings |
| pytest | Python testing framework |
| pytest-cov | Code coverage for pytest |
| numpy | Numerical computing library |

### Example Session

```bash
# Run the setup script
$ ./scripts/setup_python_env.sh

# Output shows progress...
# At the end, activate the environment:
$ source .venv/bin/activate

# Build signal-kit (if not done during setup)
(.venv) $ maturin develop

# Run tests
(.venv) $ pytest tests/python/

# Run specific test
(.venv) $ pytest tests/python/test_carrier.py::TestCarrierGeneration

# Use signal-kit in Python
(.venv) $ python3
>>> import signal_kit
>>> carrier = signal_kit.Carrier(...)
>>> iq = carrier.generate(1000)

# When done, deactivate
(.venv) $ deactivate
```

### Troubleshooting

**"Python3 is not installed"**
- Install Python 3.8 or higher from https://www.python.org/

**"maturin develop" fails with linking errors**
- Ensure Rust toolchain is installed: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- On macOS with Apple Silicon, ensure you have Xcode Command Line Tools: `xcode-select --install`

**Virtual environment activation doesn't work**
- The script uses bash-compatible activation
- If using zsh or other shells, source it differently: `. .venv/bin/activate`

**pytest can't find signal_kit module**
- Make sure `maturin develop` was run successfully
- Check that the virtual environment is activated (prompt should show `(.venv)`)

### Manual Alternative

If you prefer to set up manually:

```bash
# Create venv
python3 -m venv .venv

# Activate
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install maturin pytest pytest-cov numpy

# Build signal-kit
maturin develop

# Run tests
pytest tests/python/
```

---

## build_and_test_release.sh

Build optimized release wheels and run comprehensive tests on the resulting package.

This script is perfect for:
- Creating production wheels for distribution
- Testing package functionality end-to-end
- Validating the build pipeline
- CI/CD automation
- Pre-release verification

### Features

- ✅ Validates virtual environment is activated
- ✅ Checks for required tools (maturin, pytest)
- ✅ Cleans previous build artifacts
- ✅ Builds optimized release wheels (`maturin build --release`)
- ✅ Lists created wheel files with sizes
- ✅ Installs the freshly-built wheel
- ✅ Verifies signal-kit can be imported
- ✅ Runs full pytest test suite
- ✅ Optional: generates test coverage reports
- ✅ Provides detailed status output

### Usage

**Basic usage:**
```bash
./scripts/build_and_test_release.sh
```

**Custom venv:**
```bash
./scripts/build_and_test_release.sh ~/my_signal_kit_env
```

**For CI/CD (skip venv check):**
```bash
SKIP_VENV_CHECK=true ./scripts/build_and_test_release.sh
```

**Get help:**
```bash
./scripts/build_and_test_release.sh --help
```

### Process Flow

The script performs these steps in order:

1. **Environment Checks**
   - Validates virtual environment is active
   - Checks maturin and pytest are installed

2. **Build Preparation**
   - Cleans previous build artifacts (`dist/`, `build/`)
   - Removes old wheels to prevent confusion

3. **Wheel Building**
   - Calls `maturin build --release` for optimized compilation
   - Creates wheels in `dist/` directory

4. **Installation**
   - Lists all created wheels
   - Installs the most recent wheel
   - Uses `--force-reinstall` to ensure clean installation

5. **Verification**
   - Tests that signal-kit can be imported
   - Displays installed version

6. **Testing**
   - Runs full pytest test suite
   - Runs tests by category (optional)
   - Optional: generates coverage report

### Output Example

```
========================================
Signal-kit Release Build & Test
========================================

[INFO] Virtual environment is active: /path/to/.venv
[INFO] maturin 0.15.1
[INFO] pytest 7.4.3

▶ Cleaning previous builds
[INFO] Removing dist/
[INFO] Clean complete

[INFO] Building optimized wheels with maturin...
[INFO] This may take a minute or two on first build

Building wheels...
    ...compilation output...
[INFO] Wheel build completed successfully

▶ Build Artifacts
Wheels created in dist/:
-rw-r--r--  3.2M signal_kit-0.1.0-cp38-cp38-macosx_11_0_arm64.whl
...

▶ Installing Wheel
[INFO] Installing wheel: signal_kit-0.1.0-cp38-cp38-macosx_11_0_arm64.whl
[INFO] Wheel installed successfully

▶ Verifying signal-kit import
signal-kit version: 0.1.0
[INFO] signal-kit successfully imported

========================================
Running Test Suite
========================================

[INFO] Running pytest on tests/python/...
test_carrier.py::TestCarrierCreation::test_create_qpsk_carrier PASSED
test_carrier.py::TestCarrierGeneration::test_generate_returns_numpy_array PASSED
...
======================== 25 passed in 1.23s =========================

[INFO] All tests passed!

Generate test coverage report? (y/N)
```

### Build Artifacts

After running the script, wheels are created in the `dist/` directory:

```bash
# List wheel files
ls -lh dist/

# Output:
# -rw-r--r--  3.2M signal_kit-0.1.0-cp38-cp38-macosx_11_0_arm64.whl
# -rw-r--r--  3.2M signal_kit-0.1.0-cp39-cp39-macosx_11_0_arm64.whl
# -rw-r--r--  3.2M signal_kit-0.1.0-cp310-cp310-macosx_11_0_arm64.whl
# ... (one for each Python version)
```

### Using Generated Wheels

After building, you can:

**Install in another environment:**
```bash
pip install dist/signal_kit-0.1.0-cp310-cp310-macosx_11_0_arm64.whl
```

**Upload to PyPI:**
```bash
pip install twine
twine upload dist/*.whl
```

**Share with others:**
```bash
# Users can install directly from the wheel
pip install path/to/signal_kit-0.1.0-*.whl
```

### Troubleshooting

**"Virtual environment is not activated"**
- Activate your venv: `source .venv/bin/activate`
- Or skip check: `SKIP_VENV_CHECK=true ./scripts/build_and_test_release.sh`

**"maturin is not installed"**
- Install: `pip install maturin`

**Wheel build fails**
- Ensure Rust toolchain is installed: `rustup update`
- On macOS: `xcode-select --install`

**Wheel installation fails**
- Check Python version compatibility
- Try cleaning: `pip cache purge`

**Tests fail after wheel installation**
- Rebuild: `rm -rf dist/ build/ *.so`
- Then run script again

### Performance Notes

- Release builds use optimizations (`--release` flag)
- First build takes longer (compiling Rust → Python)
- Subsequent builds cache dependencies (faster)
- Wheel files are typically 3-4MB (depends on platform)

### Integration with CI/CD

The script is CI/CD-friendly:

```bash
#!/bin/bash
# Example GitHub Actions workflow
- name: Build and Test
  env:
    SKIP_VENV_CHECK: 'true'
  run: |
    ./scripts/build_and_test_release.sh
```

---

## Other Scripts

(Add more scripts as needed)
