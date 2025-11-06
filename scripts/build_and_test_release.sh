#!/bin/bash

# Build release wheels and test the resulting package
# This script creates optimized Python wheels using maturin and tests them

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
BUILD_DIR="dist"
VENV_DIR="${1:-.venv}"
SKIP_VENV_CHECK="${SKIP_VENV_CHECK:-false}"

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
    echo -e "${BLUE}========================================"
    echo "$1"
    echo "========================================${NC}"
    echo ""
}

print_section() {
    echo ""
    echo -e "${BLUE}▶ $1${NC}"
    echo ""
}

# Check if virtual environment is activated
check_venv_activated() {
    if [ "$SKIP_VENV_CHECK" = "true" ]; then
        print_warn "Skipping virtual environment check"
        return 0
    fi

    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Virtual environment is not activated!"
        echo ""
        echo "Please activate your virtual environment first:"
        echo "  source $VENV_DIR/bin/activate"
        echo ""
        echo "Or skip this check with:"
        echo "  SKIP_VENV_CHECK=true ./scripts/build_and_test_release.sh"
        exit 1
    fi

    print_info "Virtual environment is active: $VIRTUAL_ENV"
}

# Check if maturin is installed
check_maturin() {
    print_section "Checking for maturin"

    if ! command -v maturin &> /dev/null; then
        print_error "maturin is not installed"
        echo ""
        echo "Install maturin with:"
        echo "  pip install maturin"
        exit 1
    fi

    MATURIN_VERSION=$(maturin --version)
    print_info "$MATURIN_VERSION"
}

# Check if pytest is installed
check_pytest() {
    print_section "Checking for pytest"

    if ! command -v pytest &> /dev/null; then
        print_error "pytest is not installed"
        echo ""
        echo "Install pytest with:"
        echo "  pip install pytest"
        exit 1
    fi

    PYTEST_VERSION=$(pytest --version)
    print_info "$PYTEST_VERSION"
}

# Clean build artifacts
clean_build() {
    print_section "Cleaning previous builds"

    if [ -d "$BUILD_DIR" ]; then
        print_info "Removing $BUILD_DIR/"
        rm -rf "$BUILD_DIR"
    fi

    if [ -d "build" ]; then
        print_info "Removing build/"
        rm -rf "build"
    fi

    print_info "Clean complete"
}

# Build release wheels
build_wheels() {
    print_header "Building Release Wheels"

    print_info "Building optimized wheels with maturin..."
    print_info "This may take a minute or two on first build"
    echo ""

    if ! maturin build --release; then
        print_error "Wheel build failed"
        exit 1
    fi

    print_info "Wheel build completed successfully"
}

# List build artifacts
list_artifacts() {
    print_section "Build Artifacts"

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "No build artifacts found"
        return 1
    fi

    echo "Wheels created in $BUILD_DIR/:"
    echo ""
    ls -lh "$BUILD_DIR"/*.whl 2>/dev/null || print_warn "No .whl files found"
    echo ""
}

# Install the wheel
install_wheel() {
    print_header "Installing Wheel"

    if [ ! -d "$BUILD_DIR" ]; then
        print_error "Build directory not found"
        exit 1
    fi

    # Find the most recent wheel
    WHEEL=$(ls -t "$BUILD_DIR"/*.whl 2>/dev/null | head -n 1)

    if [ -z "$WHEEL" ]; then
        print_error "No wheel found in $BUILD_DIR"
        exit 1
    fi

    print_info "Installing wheel: $(basename $WHEEL)"
    echo ""

    if ! pip install "$WHEEL" --force-reinstall --no-deps; then
        print_error "Wheel installation failed"
        exit 1
    fi

    print_info "Wheel installed successfully"
    echo ""
}

# Verify signal-kit can be imported
verify_import() {
    print_section "Verifying signal-kit import"

    if python3 -c "import signal_kit; print(f'signal-kit version: {signal_kit.__version__}')" 2>&1; then
        print_info "signal-kit successfully imported"
    else
        print_error "Failed to import signal-kit"
        exit 1
    fi
}

# Run the test suite
run_tests() {
    print_header "Running Test Suite"

    if [ ! -d "tests/python" ]; then
        print_warn "No Python tests found at tests/python/"
        return 0
    fi

    print_info "Running pytest on tests/python/..."
    echo ""

    if pytest tests/python/ -v; then
        print_info "All tests passed!"
        return 0
    else
        print_error "Some tests failed"
        return 1
    fi
}

# Run specific test categories
run_test_categories() {
    print_header "Running Test Categories"

    # Test creation
    print_section "Testing Carrier Creation"
    pytest tests/python/test_carrier.py::TestCarrierCreation -v

    # Test generation
    print_section "Testing Signal Generation"
    pytest tests/python/test_carrier.py::TestCarrierGeneration -v

    # Test combination
    print_section "Testing Signal Combination"
    pytest tests/python/test_carrier.py::TestCarrierCombination -v

    # Test properties
    print_section "Testing Signal Properties"
    pytest tests/python/test_carrier.py::TestCarrierProperties -v
}

# Generate test report
generate_test_report() {
    print_section "Running tests with coverage report"

    if command -v pytest &> /dev/null && python3 -c "import pytest_cov" 2>/dev/null; then
        print_info "Generating coverage report..."
        pytest tests/python/ --cov=signal_kit --cov-report=term-missing --cov-report=html
        print_info "Coverage report generated in htmlcov/index.html"
    else
        print_warn "pytest-cov not installed, skipping coverage report"
    fi
}

# Print summary
print_summary() {
    print_header "Build & Test Summary"

    echo "✅ Build completed successfully"
    echo "✅ Wheel installed"
    echo "✅ Tests passed"
    echo ""
    echo "Signal-kit is ready to use!"
    echo ""
    echo "Next steps:"
    echo "  - Import signal-kit in Python: python3 -c \"import signal_kit\""
    echo "  - Read the documentation: cat README.md"
    echo "  - Check out the examples: ls tests/python/"
    echo ""
}

# Print usage information
print_usage() {
    cat << 'EOF'
Usage: ./scripts/build_and_test_release.sh [VENV_DIR]

Build optimized release wheels and run the test suite.

Arguments:
  VENV_DIR        Path to virtual environment (default: .venv)

Environment Variables:
  SKIP_VENV_CHECK If set to 'true', skips virtual environment activation check

Examples:
  # Standard usage (with .venv)
  ./scripts/build_and_test_release.sh

  # Use custom venv
  ./scripts/build_and_test_release.sh ~/my_signal_kit_env

  # Skip venv check (for CI/CD)
  SKIP_VENV_CHECK=true ./scripts/build_and_test_release.sh

Features:
  ✓ Checks for required tools (maturin, pytest)
  ✓ Cleans previous builds
  ✓ Builds optimized wheels
  ✓ Lists build artifacts
  ✓ Installs the wheel
  ✓ Verifies import
  ✓ Runs full test suite
  ✓ Optional coverage report

Requirements:
  - Virtual environment activated (or SKIP_VENV_CHECK=true)
  - maturin installed (pip install maturin)
  - pytest installed (pip install pytest)
  - Rust toolchain (for compilation)

EOF
}

# Main execution
main() {
    print_header "Signal-kit Release Build & Test"

    # Check if help requested
    if [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        print_usage
        exit 0
    fi

    # Run checks and build
    check_venv_activated
    check_maturin
    check_pytest
    clean_build
    build_wheels
    list_artifacts
    install_wheel
    verify_import
    run_tests

    # Optional: run detailed test report
    read -p "Generate test coverage report? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        generate_test_report
    fi

    print_summary
}

# Run main function
main "$@"
