# SignalKit.rs

A Rust library for digital signal processing in communications systems with Python bindings.

## Features

- **Signal Generation**: AWGN, PSK/FSK/QAM carriers, multi-carrier channels
- **Modulation**: BPSK, QPSK, 8PSK, 16APSK, 16/32/64-QAM with Gray coding
- **Filtering**: Root-Raised Cosine pulse shaping, FFT-based resampling
- **Spectrum Analysis**: Welch PSD estimation, windowing functions
- **Channel Simulation**: Multi-carrier transponders with shared noise floor
- **Python Integration**: Zero-copy NumPy interface via PyO3

## Quick Start

### Rust

```rust
use signal_kit::{Carrier, ModType};

// Generate a QPSK signal
let carrier = Carrier::new(
    ModType::_QPSK,
    0.1,        // bandwidth
    0.0,        // center freq
    10.0,       // SNR (dB)
    0.35,       // rolloff
    1e6,        // sample rate
    Some(42),   // seed
);

let iq_samples = carrier.generate::<f64>(10000);
```

Build and test:
```bash
cargo build --release
cargo test
```

### Python

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

# Generate IQ samples (returns NumPy array)
iq = carrier.generate(10000)

# Multi-carrier channel
channel = signal_kit.Channel([carrier1, carrier2])
channel.set_noise_floor_db(-100.0)
combined = channel.generate(10000)
```

Setup:
```bash
pip install maturin
maturin develop
pytest tests/python/
```

## Project Structure

```
src/
├── filter/         # RRC filtering, FFT resampling
├── generate/       # Signal/noise generation, carriers, channels
├── spectrum/       # Welch PSD, window functions
└── [core modules]  # ComplexVec, FFT, symbol maps, vectors
```

Import patterns:
```rust
// Convenient re-exports
use signal_kit::{Carrier, Channel, ComplexVec, ModType};

// Direct module access
use signal_kit::generate::{AWGN, PskCarrier};
use signal_kit::filter::RRCFilter;
use signal_kit::spectrum::welch;
```

## Visualization

Enable plotting in tests:
```bash
PLOT=true cargo test test_welch_cw_tone -- --nocapture
```

## Requirements

- Rust edition 2024+
- For Python bindings: maturin, numpy

## License

See LICENSE file.