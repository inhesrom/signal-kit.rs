# Signal-Kit Setup Scripts

This directory contains utility scripts for setting up and working with signal-kit.rs.

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

## Other Scripts

(Add more scripts as needed)
