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

### Project Structure

The codebase is organized into logical modules:

```
src/
├── lib.rs              # Main library entry point with public re-exports
├── mod_type.rs         # Modulation type definitions
├── symbol_maps.rs      # Symbol mapping tables (QPSK, 8PSK, QAM, etc.)
├── complex_vec.rs      # Complex number vector operations
├── fft.rs              # FFT/IFFT operations
├── vector_ops.rs       # Vector operations (to_db, to_linear, add, etc.)
├── vector_simd.rs      # SIMD-optimized vector operations
├── plot.rs             # Plotting utilities (test-only)
├── python_bindings.rs  # Python/NumPy bindings (optional)
│
├── filter/             # Signal filtering
│   ├── mod.rs
│   ├── rrc_filter.rs   # Root-Raised Cosine filter
│   └── fft_interpolator.rs  # FFT-based resampling
│
├── generate/           # Signal generation
│   ├── mod.rs
│   ├── awgn.rs         # Additive White Gaussian Noise
│   ├── carrier.rs      # High-level carrier generator
│   ├── channel.rs      # Multi-carrier channel simulation
│   ├── cw.rs           # Continuous Wave (CW) generation
│   ├── fsk_carrier.rs  # FSK modulated carriers
│   ├── psk_carrier.rs  # PSK modulated carriers
│   ├── ofdm_carrier.rs # OFDM carriers (placeholder)
│   ├── impairment.rs   # Channel impairments
│   └── random_bit_generator.rs  # Random bit generation
│
└── spectrum/           # Spectrum analysis
    ├── mod.rs
    ├── welch.rs        # Welch PSD estimation
    └── window.rs       # Window functions (Hann, Hamming, etc.)
```

### Module Organization

The library uses idiomatic Rust module organization:

- **Core types and utilities** are at the top level (`src/*.rs`)
- **Domain-specific functionality** is organized into submodules:
  - `filter::` - All filtering and resampling operations
  - `generate::` - Signal and noise generation
  - `spectrum::` - Spectral analysis tools

### Public API

The main public API is exposed through re-exports in `src/lib.rs`:

```rust
// Import commonly used items
use signal_kit::{Carrier, Channel, ComplexVec, ModType};

// Access submodule items
use signal_kit::generate::{AWGN, BitGenerator, PskCarrier};
use signal_kit::filter::{RRCFilter, fft_interpolator};
use signal_kit::spectrum::{welch, WindowType};
```

### Core Components

1. **BitGenerator** (`src/generate/random_bit_generator.rs`):
   - Generates random bits for testing and simulation
   - Maintains an internal buffer to efficiently extract 1-64 bits at a time
   - Supports seeded generation for reproducible tests and entropy-based generation
   - Convenience methods: `next_bit()`, `next_2_bits()`, `next_3_bits()`

2. **Symbol Maps** (`src/symbol_maps.rs`):
   - Bidirectional mapping tables for various modulation schemes
   - Implements Gray coding for QPSK, 8PSK, 16APSK, QAM16/32/64
   - Maps bit patterns to constellation points and vice versa
   - Used by carrier generators for modulation/demodulation

3. **RRCFilter** (`src/filter/rrc_filter.rs`):
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
- **Module Organization**: Domain-driven design with `filter::`, `generate::`, and `spectrum::` submodules
- **Public Re-exports**: Common types are re-exported from `lib.rs` for convenience
- **Testing**: Each module includes unit tests; use `PLOT=true` environment variable to enable visualization

### Signal Generation Workflow

Typical usage of the high-level API (`generate::Carrier`):

```rust
use signal_kit::{Carrier, ModType};

let carrier = Carrier::new(
    ModType::_QPSK,      // Modulation type
    0.1,                 // Normalized bandwidth
    0.1,                 // Center frequency
    10.0,                // SNR in dB
    0.35,                // RRC rolloff
    1e6,                 // Sample rate
    Some(42),            // Seed (optional)
);

let iq_samples = carrier.generate::<f64>(10000);
```

### Multi-Carrier and Channel Simulation

The library includes specialized support for transponder/channel simulation with multiple carriers (`generate::Channel`):

**Single Carrier (Original Behavior)**:
- Use `Carrier::generate()` to generate signal with noise included
- SNR is applied per-carrier, noise is independently generated

**Multi-Carrier with Shared AWGN (Recommended for Channels)**:
- Create multiple `Carrier` objects with different modulation, bandwidth, center frequency, and **target SNR**
- Combine multiple carriers using `Channel` struct
- `Channel::generate()` automatically scales each carrier to achieve its target SNR relative to the noise floor
- Noise is added once to the combined signal (realistic channel model)
- Power scaling formula: `Required_Power = 10^(SNR_dB/10) × Noise_Floor`

**Workflow**:

```rust
use signal_kit::{Carrier, Channel, ModType};

// Create carriers with different SNRs
let carrier1 = Carrier::new(
    ModType::_QPSK,
    0.1,        // 10% bandwidth
    0.2,        // Center freq
    10.0,       // Target 10 dB SNR
    0.35,       // RRC rolloff
    1e6,        // Sample rate
    Some(42),
);

let carrier2 = Carrier::new(
    ModType::_16QAM,
    0.15,       // 15% bandwidth
    -0.15,      // Different center freq
    5.0,        // Target 5 dB SNR
    0.35,
    1e6,
    Some(43),
);

// Create channel with shared noise floor
let mut channel = Channel::new(vec![carrier1, carrier2]);
channel.set_noise_floor_db(-85.0);  // Set shared noise floor in dB

// Generate: each carrier automatically scaled to achieve its SNR
let combined_iq = channel.generate::<f64>(10000);

// Or generate without noise for analysis
let combined_clean = channel.generate_clean::<f64>(10000);
```

**How SNR Scaling Works**:
1. User specifies target SNR for each carrier in `Carrier::snr_db`
2. User sets channel noise floor via `set_noise_floor_db()` or `set_noise_floor_linear()`
3. `Channel::generate()` calculates required power: `P_i = SNR_i_linear × N₀`
4. Each carrier is scaled to achieve this power (using `ComplexVec::scale_to_power()`)
5. Carriers are combined and AWGN is added with noise floor power
6. **Result**: Each carrier achieves its specified SNR in the frequency domain

**Key Benefits**:
- Realistic multi-carrier channel model (single noise source)
- Each carrier independently achieves its target SNR
- Supports different modulation types, bandwidths, and center frequencies
- Automatic power scaling handles constellation differences transparently

## Dependencies

### Core Dependencies
- `rustfft`: FFT operations for spectrum analysis and filtering
- `num-complex`: Complex number arithmetic for IQ samples
- `num-traits`: Generic numeric traits for Float types
- `rand`: Random number generation for AWGN and bit generation
- `wide`: SIMD operations for optimized vector operations

### Development/Test Dependencies
- `plotly`: Spectrum and constellation plotting (test-only, behind `PLOT=true`)

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
    - Exposes `generate()` for signal with noise
    - Exposes `generate_clean()` for noise-free signal
  - `PythonChannel` class wraps the Rust `Channel` struct
    - Manages multiple carriers with shared AWGN
    - Methods: `set_noise_floor_db()`, `set_noise_floor_linear()`, `set_seed()`, `generate()`, `generate_clean()`
  - Handles string-to-enum conversion for modulation types

- **python/signal_kit/**: Python package wrapper
  - Re-exports types from compiled `_signal_kit` module
  - Provides convenience functions (e.g., `combine_carriers()`)
  - Type hints via `py.typed` marker (PEP 561)

- **pyproject.toml**: Maturin configuration for Python packaging

### Python API

**Single Carrier (Simple Case)**:
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

# Generate IQ samples with noise (returns numpy array)
iq = carrier.generate(1000)  # Returns np.ndarray[np.complex128]
```

**Multi-Carrier with Shared AWGN (Channel Simulation)**:
```python
import signal_kit
import numpy as np

# Create multiple carriers
carrier1 = signal_kit.Carrier(
    modulation="QPSK",
    bandwidth=0.1,
    center_freq=0.1,
    snr_db=10.0,  # Target SNR in combined channel (informational)
    rolloff=0.35,
    sample_rate_hz=1e6,
    seed=42
)

carrier2 = signal_kit.Carrier(
    modulation="8PSK",
    bandwidth=0.15,
    center_freq=-0.15,
    snr_db=15.0,
    rolloff=0.35,
    sample_rate_hz=1e6,
    seed=43
)

# Create a channel with multiple carriers
channel = signal_kit.Channel([carrier1, carrier2])

# Set the noise floor in dB (determines SNR of all carriers)
channel.set_noise_floor_db(-100.0)

# Generate combined signal with shared AWGN
iq_combined = channel.generate(10000)  # Returns np.ndarray[np.complex128]

# Alternatively, generate clean signals without noise
iq_clean = channel.generate_clean(10000)
```

**Using clean generation for custom noise addition**:
```python
# Generate clean carriers and combine
carrier1 = signal_kit.Carrier(...)
iq1_clean = carrier1.generate_clean(1000)

carrier2 = signal_kit.Carrier(...)
iq2_clean = carrier2.generate_clean(1000)

# Custom combination and noise addition
combined_clean = iq1_clean + iq2_clean
# ... add your own AWGN using scipy.signal or custom implementation
```

### Data Transfer

- **Input to Rust**: Python NumPy arrays are passed as read-only or read-write views
- **Output from Rust**: Generated samples are transferred to Python with ownership, no copying of the actual data
- **Supported Types**: `complex64` and `complex128` map to Rust's `Complex<f32>` and `Complex<f64>`

## Testing with Visualization

Many tests support visualization when the `PLOT` environment variable is set:

```bash
# Run tests without plots (default)
cargo test

# Run specific test with plots enabled
PLOT=true cargo test test_welch_cw_tone -- --nocapture

# Run all tests with plots
PLOT=true cargo test --lib -- --nocapture
```

Tests that support plotting will skip visualization by default and show:
```
Skipping [test name] plot (set PLOT=true to enable)
```

## Edition Note

This project uses Rust edition 2024, which requires a recent Rust toolchain.
