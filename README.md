# SignalKit.rs

A Rust library for digital signal processing in communications systems.

## Features

- **Signal Generation**: Random bit generation, continuous wave (CW) signals
- **Modulation**: QPSK symbol mapping and demapping
- **Pulse Shaping**: Root-Raised Cosine (RRC) filtering
- **Frequency Analysis**: FFT, IFFT, FFT shift, and frequency utilities
- **Complex Signal Processing**: Convolution, normalization, magnitude calculations
- **Generic Types**: All components work with `f32` and `f64`

## Core Components

| Module | Description |
|--------|-------------|
| `BitGenerator` | Random bit generation with seeding support |
| `MapDemap` | QPSK modulation/demodulation |
| `RRCFilter` | Root-Raised Cosine pulse shaping |
| `FFT` | Forward/inverse FFT operations |
| `ComplexVec` | Complex vector operations and convolution |
| `CW` | Continuous wave signal generator |

## Quick Start

### Rust

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run specific module tests
cargo test symbol_mapper
cargo test rrc_filter
cargo test fft
```

### Python

Signal-kit now includes Python bindings! To get started:

```bash
# Run the automated setup script
./scripts/setup_python_env.sh

# Or manually set up:
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install maturin pytest numpy

# 3. Build signal-kit
maturin develop

# 4. Use in Python
python3 -c "import signal_kit; carrier = signal_kit.Carrier(...)"

# 5. Run tests
pytest tests/python/
```

See `scripts/README.md` for detailed setup instructions.

## Dependencies

- `rustfft` - FFT operations
- `num-complex` - Complex number arithmetic
- `num-traits` - Generic numeric traits
- `rand` - Random number generation

## Requirements

Rust edition 2024 or later.
