# CAF (Cross Ambiguity Function) Implementation Plan

## Overview
Implement an efficient Cross Ambiguity Function (CAF) processor for time-delay and Doppler frequency estimation in signal processing applications.

## Design Decisions (User Confirmed)
- **Module Location**: Standalone `src/caf.rs` (top-level module)
- **Processing Modes**: Coarse vs Fine differentiated by step size (coarse=larger steps, fine=smaller steps)
- **Interpolation**: Implement both Parabolic and Sinc methods, make it configurable
- **Parallelization**: Use rayon's global thread pool (users can set `RAYON_NUM_THREADS` env var)
- **Visualization**: Add plotly-based 3D surface, 2D heatmap, and 1D slice plots

## 1. Add Dependencies

### Update `Cargo.toml`
```toml
[dependencies]
rayon = "1.8"
```

## 2. Core CAF Module (`src/caf.rs`)

### API Structure

```rust
/// Interpolation method for peak refinement
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    Parabolic2D,    // Fast, good for smooth peaks
    Sinc2D,         // More accurate, computationally expensive
    None,           // No interpolation
}

/// Parameters for CAF computation
#[derive(Debug, Clone)]
pub struct CafParams {
    pub time_step: usize,        // 1 for fine mode, >1 for coarse mode
    pub freq_step_hz: f64,       // Frequency resolution for Doppler search
    pub max_doppler_hz: f64,     // Maximum Doppler shift to search (±)
    pub sample_rate_hz: f64,     // Sample rate of signals
}

/// CAF surface representation
#[derive(Debug, Clone)]
pub struct CafSurface<T: Float> {
    pub surface: Vec<Vec<T>>,    // 2D array: surface[doppler_idx][time_idx]
    pub time_delays: Vec<T>,     // Time delay values (samples)
    pub doppler_shifts: Vec<T>,  // Doppler shift values (Hz)
}

/// Coarse peak location
#[derive(Debug, Clone)]
pub struct Peak {
    pub magnitude: f64,
    pub delay_idx: usize,
    pub doppler_idx: usize,
    pub delay_samples: f64,
    pub doppler_hz: f64,
}

/// Refined peak with sub-sample accuracy
#[derive(Debug, Clone)]
pub struct RefinedPeak {
    pub magnitude: f64,
    pub delay_samples: f64,      // Sub-sample precision
    pub doppler_hz: f64,         // Sub-bin precision
}

/// Compute Cross Ambiguity Function using FFT-based algorithm
///
/// Uses rayon for parallel processing across Doppler frequencies.
/// For each Doppler shift:
///   1. Apply frequency shift to reference signal
///   2. FFT both signals
///   3. Multiply conjugate in frequency domain
///   4. IFFT to get correlation
///
/// # Arguments
/// * `signal1` - Reference signal (complex samples)
/// * `signal2` - Target signal (complex samples)
/// * `params` - CAF computation parameters
///
/// # Returns
/// CAF surface with time-delay vs Doppler
pub fn compute_caf<T: Float>(
    signal1: &[Complex<T>],
    signal2: &[Complex<T>],
    params: &CafParams,
) -> CafSurface<T>

/// Find peak in CAF surface using 2D max search
///
/// # Arguments
/// * `surface` - CAF surface from compute_caf()
///
/// # Returns
/// Peak location with magnitude and coordinates
pub fn find_peak<T: Float>(surface: &CafSurface<T>) -> Peak

/// Interpolate peak for sub-sample accuracy
///
/// # Arguments
/// * `surface` - CAF surface
/// * `coarse_peak` - Peak from find_peak()
/// * `method` - Interpolation method to use
///
/// # Returns
/// Refined peak with sub-sample precision
pub fn interpolate_peak<T: Float>(
    surface: &CafSurface<T>,
    coarse_peak: &Peak,
    method: InterpolationMethod,
) -> RefinedPeak
```

### Implementation Details

#### FFT-Based CAF Algorithm
```
For each Doppler frequency f_d in [-max_doppler, +max_doppler]:
    // Parallelize this loop with rayon::par_iter()
    1. freq_shift(signal1, f_d)  → signal1_shifted
    2. FFT(signal1_shifted) → S1_freq
    3. FFT(signal2) → S2_freq
    4. S1_freq.conj() * S2_freq → correlation_freq
    5. IFFT(correlation_freq) → correlation_time
    6. Store |correlation_time| in surface[doppler_idx][:]
    7. Downsample by time_step for coarse mode
```

#### 2D Parabolic Interpolation
- Fit parabola to 3×3 neighborhood around peak
- Extract sub-sample offset using quadratic coefficients
- Standard method in radar/sonar processing
- Fast and efficient for smooth peaks

#### 2D Sinc Interpolation
- Zero-pad FFT for higher resolution
- Theoretical optimum for bandlimited signals
- More computationally expensive
- Better accuracy for sharp peaks

## 3. CAF Surface Plotting (`src/caf.rs` - plot module)

### Plotting Functions (test-only, behind `PLOT=true`)

```rust
#[cfg(test)]
mod plot {
    use plotly::{Plot, Surface, Heatmap, Scatter, Layout, common::Mode};

    /// Plot 3D surface of CAF magnitude
    /// X-axis: Time delay (samples)
    /// Y-axis: Doppler shift (Hz)
    /// Z-axis: CAF magnitude (dB)
    pub fn plot_caf_surface_3d(
        surface: &CafSurface<f64>,
        peak: Option<&Peak>,
        title: &str
    )

    /// Plot 2D heatmap with optional peak marker
    /// X-axis: Time delay (samples)
    /// Y-axis: Doppler shift (Hz)
    /// Color: CAF magnitude (dB)
    pub fn plot_caf_heatmap(
        surface: &CafSurface<f64>,
        peak: Option<&Peak>,
        title: &str
    )

    /// Plot 1D slices through peak
    /// Subplot 1: CAF magnitude vs time delay at peak Doppler
    /// Subplot 2: CAF magnitude vs Doppler at peak time delay
    pub fn plot_caf_slices(
        surface: &CafSurface<f64>,
        peak: &Peak,
        title: &str
    )
}
```

### Visualization Features
- **3D Surface**: Interactive rotation, zoom, pan
- **Heatmap**: Color scale in dB, peak annotation with marker
- **Slices**: Cross-sections through peak for detailed analysis
- **Consistent styling**: Match existing plotly plots in codebase

## 4. Unit Tests (`src/caf.rs` test module)

### Test Strategy
Progressive complexity using seeded AWGN for reproducibility:

#### Test 1: Autocorrelation (Baseline)
```rust
#[test]
fn test_caf_autocorrelation()
```
- Same signal for both inputs
- Expected: Peak at (delay=0, doppler=0)
- Validates: Basic CAF computation and peak detection
- Plot: Should show single sharp peak at origin

#### Test 2: Time Delay Only
```rust
#[test]
fn test_caf_time_delay()
```
- Generate `signal_common` with AWGN
- `signal1 = signal_common + noise1`
- `signal2 = circular_shift(signal_common, N) + noise2`
- Expected: Peak at (delay=N, doppler=0)
- Validates: TDOA estimation
- Plot: Peak shifted in time dimension only

#### Test 3: Frequency Shift Only
```rust
#[test]
fn test_caf_frequency_shift()
```
- Generate `signal_common` with AWGN
- `signal1 = signal_common + noise1`
- `signal2 = freq_shift(signal_common, f_shift) + noise2`
- Expected: Peak at (delay=0, doppler=f_shift)
- Validates: Doppler estimation
- Plot: Peak shifted in frequency dimension only

#### Test 4: Combined TDOA + Doppler
```rust
#[test]
fn test_caf_tdoa_doppler_combined()
```
- `signal2 = freq_shift(circular_shift(signal_common, N), f_shift) + noise2`
- Expected: Peak at (delay=N, doppler=f_shift)
- Validates: Joint estimation
- Plot: Peak shifted in both dimensions

#### Test 5: Interpolation Accuracy
```rust
#[test]
fn test_caf_interpolation()
```
- Use fractional delay and fractional Doppler (between samples/bins)
- Test both Parabolic2D and Sinc2D methods
- Validate sub-sample accuracy
- Validates: Both interpolation methods
- Plot: Show refined peak vs coarse peak

### Test Patterns (Following Codebase Conventions)
```rust
#[test]
fn test_example() {
    use std::env;

    // Check PLOT environment variable
    let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
    let should_plot = plot.to_lowercase() == "true";

    if !should_plot {
        println!("Skipping test_example plot (set PLOT=true to enable)");
        return;
    }

    // Test implementation with visualization
    // ...
}
```

## 5. Module Integration

### Update `src/lib.rs`
```rust
// Add module declaration
pub mod caf;

// Add public re-exports for convenience
pub use caf::{
    compute_caf,
    find_peak,
    interpolate_peak,
    CafParams,
    CafSurface,
    Peak,
    RefinedPeak,
    InterpolationMethod,
};
```

## 6. Implementation Order

1. ✅ **Write plan to caf-plan.md**
2. **Add rayon dependency** to `Cargo.toml`
3. **Implement data structures** in `src/caf.rs`
   - `InterpolationMethod`, `CafParams`, `CafSurface<T>`, `Peak`, `RefinedPeak`
4. **Implement `compute_caf()`**
   - FFT-based algorithm
   - Rayon parallelization over Doppler frequencies
   - Support for coarse/fine modes via time_step
5. **Implement `find_peak()`**
   - 2D max search across surface
6. **Implement `interpolate_peak()`**
   - Parabolic2D method
   - Sinc2D method
7. **Implement plotting functions** (test module)
   - `plot_caf_surface_3d()`
   - `plot_caf_heatmap()`
   - `plot_caf_slices()`
8. **Write Test 1** - Autocorrelation
9. **Write Test 2** - Time delay only
10. **Write Test 3** - Frequency shift only
11. **Write Test 4** - Combined TDOA + Doppler
12. **Write Test 5** - Interpolation accuracy
13. **Add module to lib.rs** with public re-exports
14. **Run all tests** and verify correctness

## 7. Performance Considerations

- **FFT Efficiency**: Use `rustfft` which is already optimized in the codebase
- **Parallelization**: Rayon handles Doppler dimension (embarrassingly parallel)
- **Memory**: Pre-allocate CAF surface, reuse FFT buffers where possible
- **Coarse Mode**: Reduces computation by `time_step` factor in time dimension
- **Fine Mode**: Full resolution, use when peak region is known or for final refinement

## 8. Expected Performance

For typical parameters:
- Signal length: 10,000 samples
- Doppler range: ±1000 Hz with 10 Hz steps → 200 Doppler bins
- Time step: 1 (fine mode)

Expected: ~200 FFT pairs parallelized across available cores
- With 8 cores: ~25 FFTs per core
- With modern FFT and parallelization: should be very fast (<100ms typical)

Coarse mode (time_step=10):
- 10× faster in time dimension processing
- Useful for initial search before refinement

## 9. Usage Example

```rust
use signal_kit::{compute_caf, find_peak, interpolate_peak, CafParams, InterpolationMethod};

// Setup parameters
let params = CafParams {
    time_step: 1,           // Fine mode
    freq_step_hz: 10.0,     // 10 Hz resolution
    max_doppler_hz: 1000.0, // ±1 kHz search
    sample_rate_hz: 1e6,    // 1 MHz sample rate
};

// Compute CAF
let surface = compute_caf(&signal1, &signal2, &params);

// Find coarse peak
let peak = find_peak(&surface);

// Refine with interpolation
let refined = interpolate_peak(&surface, &peak, InterpolationMethod::Parabolic2D);

println!("TDOA: {:.3} samples", refined.delay_samples);
println!("Doppler: {:.3} Hz", refined.doppler_hz);
```

## 10. Future Enhancements (Not in Initial Implementation)

- Batch processing for multiple signal pairs
- GPU acceleration via CUDA/OpenCL
- Adaptive step size (coarse → fine automatic refinement)
- Additional interpolation methods (polynomial, spline)
- CAF normalization options (energy, peak)
- Confidence metrics for peak detection (SNR, sharpness)
- Multi-peak detection for multipath scenarios

## References

- **FFT-based cross-correlation**: Faster than time-domain for long signals
- **Rayon parallelization**: Efficient work-stealing thread pool
- **Parabolic interpolation**: Standard in radar processing (CFAR detectors)
- **Sinc interpolation**: Theoretically optimal for bandlimited signals
- **CAF applications**: TDOA geolocation, passive radar, sonar, communication sync
