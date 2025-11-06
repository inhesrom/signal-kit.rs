#!/bin/bash

# Setup Python virtual environment with maturin and pytest
# This script creates a Python virtual environment and installs the necessary
# dependencies for working with signal-kit Python bindings.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
VENV_DIR="${1:-.venv}"
PYTHON_VERSION="3.8"

# Functions
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
    echo ""
}

# Check if Python is installed
check_python() {
    print_info "Checking for Python installation..."

    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        echo "Please install Python 3.8 or higher from https://www.python.org/"
        exit 1
    fi

    PYTHON_VERSION_INSTALLED=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_info "Found Python $PYTHON_VERSION_INSTALLED"
}

# Create virtual environment
create_venv() {
    print_header "Creating Python Virtual Environment"

    if [ -d "$VENV_DIR" ]; then
        print_warn "Virtual environment already exists at $VENV_DIR"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing virtual environment..."
            rm -rf "$VENV_DIR"
            print_info "Creating new virtual environment..."
            python3 -m venv "$VENV_DIR"
        else
            print_info "Using existing virtual environment"
        fi
    else
        print_info "Creating new virtual environment at $VENV_DIR..."
        python3 -m venv "$VENV_DIR"
    fi
}

# Activate virtual environment
activate_venv() {
    print_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    print_info "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_header "Upgrading pip, setuptools, and wheel"
    python3 -m pip install --upgrade pip setuptools wheel
    print_info "pip, setuptools, and wheel upgraded"
}

# Install maturin
install_maturin() {
    print_header "Installing maturin"
    pip install maturin
    print_info "maturin installed successfully"

    # Show maturin version
    maturin --version
}

# Install pytest
install_pytest() {
    print_header "Installing pytest"
    pip install pytest pytest-cov
    print_info "pytest and pytest-cov installed successfully"

    # Show pytest version
    pytest --version
}

# Install numpy (optional but useful)
install_numpy() {
    print_header "Installing numpy"
    pip install numpy
    print_info "numpy installed successfully"

    python3 -c "import numpy; print(f'numpy {numpy.__version__}')"
}

# Build signal-kit with maturin
build_signal_kit() {
    print_header "Building signal-kit with maturin"

    read -p "Do you want to build signal-kit now? (Y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        print_info "Building signal-kit..."
        maturin develop
        print_info "signal-kit built and installed successfully"
    else
        print_warn "Skipping signal-kit build"
        echo "You can build later by running: maturin develop"
    fi
}

# Print next steps
print_next_steps() {
    print_header "Setup Complete!"

    echo "Next steps:"
    echo ""
    echo "1. Activate the virtual environment:"
    echo "   source $VENV_DIR/bin/activate"
    echo ""
    echo "2. Build and install signal-kit (if not done above):"
    echo "   maturin develop"
    echo ""
    echo "3. Run Python tests:"
    echo "   pytest tests/python/"
    echo ""
    echo "4. Run specific test:"
    echo "   pytest tests/python/test_carrier.py::TestCarrierGeneration::test_generate_returns_numpy_array"
    echo ""
    echo "5. Deactivate virtual environment when done:"
    echo "   deactivate"
    echo ""
}

# Main execution
main() {
    echo "========================================"
    echo "signal-kit Python Environment Setup"
    echo "========================================"
    echo ""
    echo "Virtual environment: $VENV_DIR"
    echo ""

    check_python
    create_venv
    activate_venv
    upgrade_pip
    install_maturin
    install_pytest
    install_numpy
    build_signal_kit
    print_next_steps
}

# Run main function
main
