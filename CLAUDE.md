# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SignalKit.rs is a Rust library for digital signal processing, focused on communication systems. It implements components for generating, modulating, filtering, and processing digital signals commonly used in wireless communications.

## Build and Test Commands

### Rust (Core Library)

```bash
# Build the project (Rust-only, no Python bindings)
cargo build

# Build with optimizations
cargo build --release

# Run all Rust tests
cargo test

# Run tests for a specific module
cargo test symbol_mapper
cargo test rrc_filter
cargo test random_bit_generator

# Run a specific test
cargo test test_qpsk_mapper
```

### Python Bindings

```bash
# Build Python bindings (requires maturin)
# Install maturin first: pip install maturin

# Install in development environment
maturin develop

# Build wheel for distribution
maturin build

# Run Python tests (after maturin develop)
pytest tests/python/

# Run specific Python test
pytest tests/python/test_carrier.py::TestCarrierGeneration::test_generate_returns_numpy_array
```

**Important**: By default, the Cargo build does NOT include Python support. This ensures the core Rust library can be used without any Python dependencies.

**For Python bindings, you MUST use `maturin`**, which properly configures Python development headers and linking:

```bash
# Install maturin
pip install maturin

# For development: install in current Python environment
maturin develop

# For distribution: build a wheel
maturin build --release
```

Do NOT use `cargo build --features python` directly, as it lacks proper Python environment configuration. The `python` feature flag is used automatically by maturin during the build process.

## Architecture

### Core Components

The codebase is structured as a collection of independent signal processing modules:

1. **BitGenerator** (`src/random_bit_generator.rs`):
   - Generates random bits for testing and simulation
   - Maintains an internal buffer to efficiently extract 1-64 bits at a time
   - Supports seeded generation for reproducible tests and entropy-based generation
   - Convenience methods: `next_bit()`, `next_2_bits()`, `next_3_bits()`

2. **MapDemap** (`src/symbol_mapper.rs`):
   - Symbol mapper/demapper using generics over Float types
   - Currently implements QPSK (Quadrature Phase Shift Keying) modulation
   - Maps 2-bit values to IQ constellation points
   - Demodulates symbols back to bits using nearest-neighbor search
   - Designed to be extensible for 8-PSK and 16-APSK (infrastructure in place)

3. **RRCFilter** (`src/rrc_filter.rs`):
   - Root-Raised Cosine pulse shaping filter
   - Generic implementation works with any Float type
   - Uses generic return types with trait bounds (`V: Default + Extend<T>`)
   - This allows building filters into `Vec<T>` or any compatible collection type
   - Parameters: number of taps, sample rate, symbol rate, and roll-off factor (beta)
   - Handles special cases: t=0 and t at discontinuity points

### Design Patterns

- **Generic Float Types**: All DSP components use `num_traits::Float` to work with both `f32` and `f64`
- **Type-Driven Collections**: The `RRCFilter::build_filter()` uses generic return types with trait bounds, enabling flexible output container types
- **Complex Number Support**: Uses `num_complex::Complex` for IQ samples and constellation points
- **Testing**: Each module includes unit tests to verify correctness

### Current Workflow

The `main.rs` demonstrates a typical signal generation pipeline:
1. Generate random bits using `BitGenerator`
2. Map bit pairs to QPSK symbols using `MapDemap`
3. Collect IQ samples (currently generates 80M symbols)

Note: The RRC filter is implemented but not yet integrated into the main pipeline.

## Dependencies

- `rustfft`: FFT operations (though not currently used in visible code)
- `num-complex`: Complex number arithmetic for IQ samples
- `num-traits`: Generic numeric traits for Float types
- `rand`: Random number generation
- `wide`: SIMD operations (not currently used in visible code)
- `bimap`: Bidirectional maps (not currently used in visible code)

### Optional Dependencies (Python Bindings)

When the `python` feature is enabled:
- `pyo3`: Python bindings for Rust
- `numpy`: NumPy array interface

## Python Bindings Architecture

### Design Principles

1. **No Breaking Changes**: The Rust API remains completely unchanged. Python support is purely additive via the `python` feature flag.
2. **Zero-Copy Data Transfer**: Uses rust-numpy for efficient data sharing between Python and Rust without copying.
3. **Conditional Compilation**: All Python-specific code is behind `#[cfg(feature = "python")]` guards.

### Structure

- **src/python_bindings.rs**: All PyO3 wrapper code
  - `PythonCarrier` class wraps the Rust `Carrier` struct
  - Exposes `generate()` method that returns NumPy arrays
  - Handles string-to-enum conversion for modulation types

- **python/signal_kit/**: Python package wrapper
  - Re-exports types from compiled `_signal_kit` module
  - Provides convenience functions (e.g., `combine_carriers()`)
  - Type hints via `py.typed` marker (PEP 561)

- **pyproject.toml**: Maturin configuration for Python packaging

### Python API

```python
import signal_kit
import numpy as np

# Create a carrier
carrier = signal_kit.Carrier(
    modulation="QPSK",
    bandwidth=0.1,
    center_freq=0.1,
    snr_db=10.0,
    rolloff=0.35,
    sample_rate_hz=1e6,
    seed=42
)

# Generate IQ samples (returns numpy array)
iq = carrier.generate(1000)  # Returns np.ndarray[np.complex128]

# Combine multiple carriers
carrier2 = signal_kit.Carrier(...)
iq2 = carrier2.generate(1000)
combined = iq + iq2  # NumPy addition
```

### Data Transfer

- **Input to Rust**: Python NumPy arrays are passed as read-only or read-write views
- **Output from Rust**: Generated samples are transferred to Python with ownership, no copying of the actual data
- **Supported Types**: `complex64` and `complex128` map to Rust's `Complex<f32>` and `Complex<f64>`

## Edition Note

This project uses Rust edition 2024, which requires a recent Rust toolchain.
