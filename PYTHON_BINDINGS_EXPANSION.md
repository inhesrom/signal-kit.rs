# Python Bindings Expansion Plan

## Overview

This document outlines the plan to expand signal-kit's Python bindings from the current single `Carrier` class to expose all major DSP components available in the Rust library.

## Design Decisions

Based on requirements gathering:

1. **ComplexVec**: Expose as a Python class (not just NumPy arrays)
2. **Return Types**: All signal operations return `ComplexVec` objects (enables method chaining)
3. **Priority**: All components (analysis tools, signal generators, filters)
4. **Conversion**: Provide seamless NumPy ↔ ComplexVec conversion

## Current State

### Currently Exposed ✅
- `Carrier` class - High-level signal generator with modulation, filtering, frequency shift, and noise
  - Returns: NumPy array of complex128

### Available but Not Exposed ❌
- **15 additional components** across 7 categories (see below)

---

## Implementation Plan

### Phase 1: Core Data Structure

#### 1. Create `ComplexVec` Python Wrapper

**File**: `src/python_bindings.rs`

**Class**: `ComplexVec` (wraps Rust `ComplexVec<f64>`)

**Constructors**:
- `ComplexVec.from_array(numpy_array)` - Create from NumPy array
- `ComplexVec.new()` - Create empty

**Methods**:
- `to_array()` → NumPy array - Convert to NumPy
- `len()` → int - Number of samples
- `abs()` → NumPy array - Get magnitudes
- `normalize()` → ComplexVec - Normalize to unit power (returns new)
- `normalize_inplace()` → None - Normalize in-place
- `convolve(kernel, mode="same")` → ComplexVec - Convolution (mode: "full", "same", "valid")
- `convolve_inplace(kernel, mode="same")` → None - In-place convolution
- `freq_shift(freq_offset, sample_rate)` → ComplexVec - Frequency shift (returns new)

**Operators**:
- `__add__`, `__sub__` - Addition and subtraction
- `__getitem__`, `__setitem__` - Indexing
- `__len__` - Length
- `__repr__` - String representation

**Example Usage**:
```python
import signal_kit
import numpy as np

# Create from NumPy
signal = signal_kit.ComplexVec.from_array(np.array([1+1j, 2+2j, 3+3j]))

# Operations
mags = signal.abs()
normalized = signal.normalize()
shifted = signal.freq_shift(freq_offset=0.1, sample_rate=1e6)

# Convolution
kernel = signal_kit.ComplexVec.from_array([...])
result = signal.convolve(kernel, mode="same")

# Arithmetic
combined = signal1 + signal2

# Convert back to NumPy
array = signal.to_array()
```

#### 2. Update `Carrier.generate()` Return Type

**Change**: Return `ComplexVec` instead of raw NumPy array

**Backward Compatibility**: ComplexVec automatically converts to NumPy in numerical operations

**Before**:
```python
iq = carrier.generate(1000)  # Returns np.ndarray
```

**After**:
```python
iq = carrier.generate(1000)  # Returns ComplexVec
iq.normalize()  # Can now call methods
array = iq.to_array()  # Explicit conversion if needed
```

---

### Phase 2: Analysis Tools

#### 3. Add `welch()` Function for PSD Estimation

**File**: `src/python_bindings.rs`

**Function**: `welch(signal, sample_rate, nperseg, noverlap=None, nfft=None, window="hann", averaging="mean")`

**Parameters**:
- `signal`: ComplexVec or NumPy array
- `sample_rate`: float - Sample rate in Hz
- `nperseg`: int - Segment length
- `noverlap`: int - Overlap length (default: nperseg // 2)
- `nfft`: int - FFT length (default: nperseg)
- `window`: str - Window type: "hann", "hamming", "blackman", "rectangular"
- `averaging`: str - Averaging method: "mean", "median", "max", "min"

**Returns**: `(frequencies, psd)` - Both as NumPy float64 arrays

**Example**:
```python
freqs, psd = signal_kit.welch(
    iq,
    sample_rate=1e6,
    nperseg=1024,
    noverlap=512,
    window="hann",
    averaging="mean"
)

import matplotlib.pyplot as plt
plt.plot(freqs, 10*np.log10(psd))
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB)')
```

#### 4. Add FFT Functions Module

**File**: `src/python_bindings.rs`

**Functions**:

- `fft(signal)` → ComplexVec
  - Forward FFT (scaled by 1/N)
  - Accepts: ComplexVec or NumPy array

- `ifft(signal)` → ComplexVec
  - Inverse FFT
  - Accepts: ComplexVec or NumPy array

- `fftshift(signal)` → ComplexVec
  - Shift zero-frequency component to center
  - Accepts: ComplexVec or NumPy array

- `fftfreqs(start, stop, num_points)` → NumPy array (float64)
  - Generate frequency axis
  - Parameters: start (float), stop (float), num_points (int)

**Example**:
```python
# FFT analysis
spectrum = signal_kit.fft(iq)
spectrum_shifted = signal_kit.fftshift(spectrum)
freqs = signal_kit.fftfreqs(-0.5, 0.5, len(spectrum))

# Plot
import matplotlib.pyplot as plt
plt.plot(freqs, spectrum_shifted.abs())

# Inverse
time_domain = signal_kit.ifft(spectrum)
```

#### 5. Add Window Functions

**File**: `src/python_bindings.rs`

**Enum**: `WindowType`
- `WindowType.Hann`
- `WindowType.Hamming`
- `WindowType.Blackman`
- `WindowType.Rectangular`

**Function**: `generate_window(window_type, size)` → NumPy array (float64)

**Parameters**:
- `window_type`: str - "hann", "hamming", "blackman", "rectangular"
- `size`: int - Window length

**Example**:
```python
# Generate window
window = signal_kit.generate_window("hann", 1024)

# Apply to signal
windowed = iq.to_array() * window
```

---

### Phase 3: Signal Generators

#### 6. Add `AWGN` Noise Generator Class

**File**: `src/python_bindings.rs`

**Class**: `AWGN` (wraps Rust `AWGN`)

**Constructor**: `AWGN(sample_rate, noise_power, seed=None)`

**Parameters**:
- `sample_rate`: float - Sample rate in Hz
- `noise_power`: float - Noise power in dBW
- `seed`: int (optional) - Random seed for reproducibility

**Methods**:
- `generate(num_samples)` → ComplexVec - Generate noise samples

**Example**:
```python
# Create noise generator
noise_gen = signal_kit.AWGN(
    sample_rate=1e6,
    noise_power=-10.0,  # -10 dBW
    seed=42
)

# Generate noise
noise = noise_gen.generate(10000)

# Add to signal
carrier = signal_kit.Carrier(...)
clean = carrier.generate(10000)
noisy = clean + noise
```

#### 7. Add `CW` Tone Generator Class

**File**: `src/python_bindings.rs`

**Class**: `CW` (wraps Rust `CW`)

**Constructor**: `CW(freq_hz, sample_rate_hz)`

**Parameters**:
- `freq_hz`: float - Tone frequency in Hz
- `sample_rate_hz`: float - Sample rate in Hz

**Methods**:
- `generate(num_samples)` → ComplexVec - Generate tone (phase-continuous)

**Example**:
```python
# Create tone generator
cw = signal_kit.CW(freq_hz=1000, sample_rate_hz=1e6)

# Generate multiple blocks (phase-continuous)
tone1 = cw.generate(1000)
tone2 = cw.generate(1000)  # Phase continues from tone1
```

#### 8. Add `FskCarrier` Class

**File**: `src/python_bindings.rs`

**Class**: `FskCarrier` (wraps Rust `FskCarrier`)

**Constructor**: `FskCarrier(sample_rate, symbol_rate, carrier_freq, freq_deviation, seed=None)`

**Parameters**:
- `sample_rate`: float - Sample rate in Hz
- `symbol_rate`: float - Symbol rate in Hz
- `carrier_freq`: float - Normalized carrier frequency (-0.5 to 0.5)
- `freq_deviation`: float - Normalized frequency deviation
- `seed`: int (optional) - Random seed

**Methods**:
- `generate(num_samples)` → ComplexVec - Generate FSK modulated signal

**Example**:
```python
# Create FSK carrier
fsk = signal_kit.FskCarrier(
    sample_rate=1e6,
    symbol_rate=1e5,
    carrier_freq=0.1,
    freq_deviation=0.05,
    seed=42
)

# Generate signal
fsk_signal = fsk.generate(10000)
```

---

### Phase 4: Filtering

#### 9. Add `RRCFilter` Class

**File**: `src/python_bindings.rs`

**Class**: `RRCFilter` (wraps Rust `RRCFilter`)

**Constructor**: `RRCFilter(num_taps, sample_rate, symbol_rate, rolloff)`

**Parameters**:
- `num_taps`: int - Number of filter taps (should be odd)
- `sample_rate`: float - Sample rate in Hz
- `symbol_rate`: float - Symbol rate in Hz
- `rolloff`: float - Rolloff factor beta (0.0 to 1.0)

**Methods**:
- `build_filter()` → ComplexVec - Get filter coefficients

**Example**:
```python
# Design RRC filter
rrc = signal_kit.RRCFilter(
    num_taps=101,
    sample_rate=1e6,
    symbol_rate=1e5,
    rolloff=0.35
)

# Get coefficients
taps = rrc.build_filter()

# Apply to signal
filtered = signal.convolve(taps, mode="same")
```

#### 10. Add `resample()` Function (FFT Interpolator)

**File**: `src/python_bindings.rs`

**Function**: `resample(signal, factor)` → ComplexVec

**Parameters**:
- `signal`: ComplexVec or NumPy array - Input signal
- `factor`: float - Resampling factor
  - `factor > 1.0`: Upsampling (interpolation)
  - `factor < 1.0`: Downsampling (decimation)
  - `factor == 1.0`: No change

**Returns**: ComplexVec - Resampled signal

**Example**:
```python
# Upsample by 2x
upsampled = signal_kit.resample(iq, factor=2.0)

# Downsample by 2x
downsampled = signal_kit.resample(iq, factor=0.5)

# Arbitrary rate (e.g., 1.5x)
resampled = signal_kit.resample(iq, factor=1.5)
```

---

### Phase 5: Integration and Testing

#### 11. Update Python Package

**File**: `python/signal_kit/__init__.py`

**Exports**:
```python
"""
Signal-Kit: Digital Signal Processing Library for Communications

Core Classes:
    Carrier - High-level signal generator
    ComplexVec - Complex signal container with operations
    AWGN - Additive White Gaussian Noise generator
    CW - Continuous Wave tone generator
    FskCarrier - FSK modulation carrier
    RRCFilter - Root-Raised Cosine filter

Analysis Functions:
    welch - Power spectral density estimation
    fft - Forward FFT
    ifft - Inverse FFT
    fftshift - Shift zero-frequency to center
    fftfreqs - Generate frequency axis
    generate_window - Window function generation
    resample - FFT-based resampling
"""

from ._signal_kit import (
    # Core
    Carrier,
    ComplexVec,

    # Generators
    AWGN,
    CW,
    FskCarrier,

    # Filters
    RRCFilter,

    # Analysis
    welch,
    fft,
    ifft,
    fftshift,
    fftfreqs,
    generate_window,
    resample,

    # Version
    __version__,
)

__all__ = [
    "Carrier",
    "ComplexVec",
    "AWGN",
    "CW",
    "FskCarrier",
    "RRCFilter",
    "welch",
    "fft",
    "ifft",
    "fftshift",
    "fftfreqs",
    "generate_window",
    "resample",
]
```

#### 12. Add Comprehensive Tests

**Create Test Files**:

1. **`tests/python/test_complex_vec.py`** (~200 lines)
   - Test ComplexVec creation (from_array, new)
   - Test conversions (to_array)
   - Test operations (normalize, abs, convolve, freq_shift)
   - Test operators (+, -, [], len)
   - Test convolution modes (full, same, valid)

2. **`tests/python/test_analysis.py`** (~250 lines)
   - Test welch() with different parameters
   - Test all window types
   - Test averaging methods
   - Test FFT functions (fft, ifft, fftshift, fftfreqs)
   - Test FFT round-trip (fft → ifft)
   - Test window generation

3. **`tests/python/test_generators.py`** (~200 lines)
   - Test AWGN creation and generation
   - Test AWGN with/without seed (reproducibility)
   - Test CW tone generation
   - Test CW phase continuity
   - Test FskCarrier creation and generation
   - Test FSK with seed

4. **`tests/python/test_filters.py`** (~150 lines)
   - Test RRCFilter creation
   - Test RRC filter properties (energy, symmetry)
   - Test resample() upsampling
   - Test resample() downsampling
   - Test resample() arbitrary rates

**Total New Tests**: ~800 lines, 40+ test cases

---

## Expected Python API Summary

```python
import signal_kit
import numpy as np

# ===== Core Data Structure =====
signal = signal_kit.ComplexVec.from_array(np.array([1+1j, 2+2j]))
signal.normalize()
shifted = signal.freq_shift(0.1, 1e6)
result = signal1 + signal2

# ===== Signal Generation =====
# Carrier (existing, updated to return ComplexVec)
carrier = signal_kit.Carrier(
    modulation="QPSK",
    bandwidth=0.1,
    center_freq=0.0,
    snr_db=10.0,
    rolloff=0.35,
    sample_rate_hz=1e6,
    seed=42
)
iq = carrier.generate(1000)  # Returns ComplexVec

# AWGN noise
noise_gen = signal_kit.AWGN(sample_rate=1e6, noise_power=-10.0, seed=42)
noise = noise_gen.generate(1000)

# CW tone
cw = signal_kit.CW(freq_hz=1000, sample_rate_hz=1e6)
tone = cw.generate(1000)

# FSK modulation
fsk = signal_kit.FskCarrier(
    sample_rate=1e6,
    symbol_rate=1e5,
    carrier_freq=0.1,
    freq_deviation=0.05,
    seed=42
)
fsk_signal = fsk.generate(10000)

# ===== Analysis =====
# PSD estimation
freqs, psd = signal_kit.welch(
    iq,
    sample_rate=1e6,
    nperseg=1024,
    window="hann",
    averaging="mean"
)

# FFT operations
spectrum = signal_kit.fft(iq)
spectrum_shifted = signal_kit.fftshift(spectrum)
freqs = signal_kit.fftfreqs(-0.5, 0.5, len(spectrum))
time_domain = signal_kit.ifft(spectrum)

# Window generation
window = signal_kit.generate_window("hann", 1024)

# ===== Filtering =====
# RRC filter design
rrc = signal_kit.RRCFilter(
    num_taps=101,
    sample_rate=1e6,
    symbol_rate=1e5,
    rolloff=0.35
)
taps = rrc.build_filter()
filtered = iq.convolve(taps, mode="same")

# Resampling
upsampled = signal_kit.resample(iq, factor=2.0)
downsampled = signal_kit.resample(iq, factor=0.5)

# ===== Method Chaining =====
result = (carrier.generate(1000)
          .normalize()
          .freq_shift(0.1, 1e6)
          .convolve(rrc.build_filter()))
```

---

## Implementation Checklist

### Phase 1: Core Data Structure
- [ ] Implement `ComplexVec` Python wrapper
- [ ] Add conversion methods (from_array, to_array)
- [ ] Add operations (normalize, abs, convolve, freq_shift)
- [ ] Add operators (+, -, [], len)
- [ ] Update `Carrier.generate()` to return ComplexVec
- [ ] Write tests for ComplexVec (~200 lines)

### Phase 2: Analysis Tools
- [ ] Implement `welch()` function
- [ ] Implement FFT functions (fft, ifft, fftshift, fftfreqs)
- [ ] Implement window functions (generate_window)
- [ ] Write tests for analysis tools (~250 lines)

### Phase 3: Signal Generators
- [ ] Implement `AWGN` class
- [ ] Implement `CW` class
- [ ] Implement `FskCarrier` class
- [ ] Write tests for generators (~200 lines)

### Phase 4: Filtering
- [ ] Implement `RRCFilter` class
- [ ] Implement `resample()` function
- [ ] Write tests for filters (~150 lines)

### Phase 5: Integration
- [ ] Update `python/signal_kit/__init__.py`
- [ ] Run full test suite
- [ ] Update documentation (README.md, CLAUDE.md)
- [ ] Build and verify with maturin

---

## Success Criteria

1. ✅ All 15 new components exposed to Python
2. ✅ ComplexVec acts as primary return type
3. ✅ All operations support method chaining
4. ✅ Seamless NumPy conversion (to_array/from_array)
5. ✅ 40+ new test cases pass
6. ✅ Documentation updated
7. ✅ Backward compatibility maintained (Carrier still works)

---

## Estimated Scope

- **Code Changes**: ~1500 lines of Rust (python_bindings.rs)
- **Tests**: ~800 lines of Python
- **Documentation**: Updates to README.md, CLAUDE.md, scripts/README.md
- **Development Time**: 4-6 hours for full implementation + testing

---

## Notes

- All signal operations maintain phase continuity where applicable (CW, FskCarrier)
- ComplexVec uses f64 precision (complex128 in NumPy)
- Window functions are real-valued (float64)
- PSD and frequency arrays are real-valued (float64)
- Error handling follows PyO3 conventions (PyResult, PyErr)
- All classes support optional seed parameters for reproducibility
