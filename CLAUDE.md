# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SignalKit.rs is a Rust library for digital signal processing, focused on communication systems. It implements components for generating, modulating, filtering, and processing digital signals commonly used in wireless communications.

## Build and Test Commands

```bash
# Build the project
cargo build

# Build with optimizations
cargo build --release

# Run the main binary
cargo run

# Run with optimizations
cargo run --release

# Run all tests
cargo test

# Run tests for a specific module
cargo test symbol_mapper
cargo test rrc_filter
cargo test random_bit_generator

# Run a specific test
cargo test test_qpsk_mapper
```

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

## Edition Note

This project uses Rust edition 2024, which requires a recent Rust toolchain.
