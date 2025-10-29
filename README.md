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

## Dependencies

- `rustfft` - FFT operations
- `num-complex` - Complex number arithmetic
- `num-traits` - Generic numeric traits
- `rand` - Random number generation

## Requirements

Rust edition 2024 or later.
