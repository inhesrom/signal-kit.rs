//! Build-time selected SIMD operations for real-time IQ blocks.

use bytemuck::{cast_slice, cast_slice_mut};
use num_complex::Complex;
use num_traits::Float;

/// Describes the DSP vector backend selected by the build script.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Scalar fallback for non-SIMD builds and unsupported targets.
    Scalar,
    /// SSE2 backend using 128-bit registers.
    Sse2,
    /// AVX2/FMA backend using 256-bit registers.
    Avx2Fma,
    /// AVX-512F backend using 512-bit registers.
    Avx512F,
}

impl SimdBackend {
    /// Returns the backend register width in bits.
    pub const fn register_bits(self) -> usize {
        match self {
            Self::Scalar => 32,
            Self::Sse2 => 128,
            Self::Avx2Fma => 256,
            Self::Avx512F => 512,
        }
    }

    /// Returns the number of complex f32 IQ samples per vector register.
    pub const fn iq_lanes(self) -> usize {
        self.iq_lanes_f32()
    }

    /// Returns the number of complex f32 IQ samples per vector register.
    pub const fn iq_lanes_f32(self) -> usize {
        match self {
            Self::Scalar => 1,
            _ => self.register_bits() / 64,
        }
    }

    /// Returns the number of complex f64 IQ samples per vector register.
    pub const fn iq_lanes_f64(self) -> usize {
        match self {
            Self::Scalar => 1,
            _ => self.register_bits() / 128,
        }
    }
}

/// A small planner exposing the selected IQ backend and lane geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IqVectorPlan {
    backend: SimdBackend,
    register_bits: usize,
    iq_lanes_f32: usize,
    iq_lanes_f64: usize,
}

impl IqVectorPlan {
    /// Builds a plan for the backend selected at compile time.
    pub const fn selected() -> Self {
        Self {
            backend: SELECTED_BACKEND,
            register_bits: REGISTER_BITS,
            iq_lanes_f32: IQ_LANES_F32,
            iq_lanes_f64: IQ_LANES_F64,
        }
    }

    /// Returns the backend selected for this build.
    pub const fn backend(self) -> SimdBackend {
        self.backend
    }

    /// Returns the selected register width in bits.
    pub const fn register_bits(self) -> usize {
        self.register_bits
    }

    /// Returns complex f32 IQ samples processed per vector register.
    pub const fn iq_lanes(self) -> usize {
        self.iq_lanes_f32
    }

    /// Returns complex f32 IQ samples processed per vector register.
    pub const fn iq_lanes_f32(self) -> usize {
        self.iq_lanes_f32
    }

    /// Returns complex f64 IQ samples processed per vector register.
    pub const fn iq_lanes_f64(self) -> usize {
        self.iq_lanes_f64
    }

    /// Returns complex IQ samples processed per vector register for `T`.
    pub fn iq_lanes_for<T: IqSample>(self) -> usize {
        T::iq_lanes_for_backend(self.backend)
    }

    /// Writes `left + right` into `output` without allocating.
    pub fn iq_add_to<T: IqSample>(self, left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
        iq_add_to(left, right, output);
    }

    /// Adds `input` into `target` without allocating.
    pub fn iq_add_in_place<T: IqSample>(self, target: &mut [Complex<T>], input: &[Complex<T>]) {
        iq_add_in_place(target, input);
    }

    /// Scales `target` by `scale` without allocating.
    pub fn iq_scale_in_place<T: IqSample>(self, target: &mut [Complex<T>], scale: T) {
        iq_scale_in_place(target, scale);
    }

    /// Adds `scale * input` into `target` without allocating.
    pub fn iq_add_scaled_in_place<T: IqSample>(self, target: &mut [Complex<T>], scale: Complex<T>, input: &[Complex<T>]) {
        iq_add_scaled_in_place(target, scale, input);
    }

    /// Writes `left * right` into `output` without allocating.
    pub fn iq_multiply_to<T: IqSample>(self, left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
        iq_multiply_to(left, right, output);
    }

    /// Writes `conjugate(left) * right` into `output` without allocating.
    pub fn iq_conjugate_multiply_to<T: IqSample>(self, left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
        iq_conjugate_multiply_to(left, right, output);
    }

    /// Returns the sum of `conjugate(left) * right` for each pair of IQ samples.
    pub fn iq_conjugate_multiply_sum<T: IqSample>(self, left: &[Complex<T>], right: &[Complex<T>]) -> Complex<T> {
        iq_conjugate_multiply_sum(left, right)
    }

    /// Returns the sum of per-sample IQ power values.
    pub fn iq_sum_squared_magnitudes<T: IqSample>(self, input: &[Complex<T>]) -> T {
        iq_sum_squared_magnitudes(input)
    }

    /// Writes per-sample magnitudes into `output` without allocating.
    pub fn iq_calculate_magnitudes_to<T: IqSample>(self, input: &[Complex<T>], output: &mut [T]) {
        iq_calculate_magnitudes_to(input, output);
    }

    /// Writes the full direct IQ convolution into `output` without allocating.
    ///
    /// # Panics
    /// Panics when `input` or `kernel` is empty, or when `output.len()` is not
    /// `input.len() + kernel.len() - 1`.
    pub fn iq_convolve_full_to<T: IqSample>(self, input: &[Complex<T>], kernel: &[Complex<T>], output: &mut [Complex<T>]) {
        iq_convolve_full_to(input, kernel, output);
    }

    /// Writes a direct IQ convolution range into `output` without allocating.
    pub(crate) fn iq_convolve_range_to<T: IqSample>(
        self,
        input: &[Complex<T>],
        kernel: &[Complex<T>],
        full_output_start: usize,
        output: &mut [Complex<T>],
    ) {
        iq_convolve_range_to(input, kernel, full_output_start, output);
    }
}

impl Default for IqVectorPlan {
    fn default() -> Self {
        Self::selected()
    }
}

/// Backend selected by the current build.
pub const SELECTED_BACKEND: SimdBackend = selected_backend();

/// Register width selected by the current build.
pub const REGISTER_BITS: usize = SELECTED_BACKEND.register_bits();

/// Complex f32 samples processed per selected register.
pub const IQ_LANES: usize = IQ_LANES_F32;

/// Complex f32 samples processed per selected register.
pub const IQ_LANES_F32: usize = SELECTED_BACKEND.iq_lanes_f32();

/// Complex f64 samples processed per selected register.
pub const IQ_LANES_F64: usize = SELECTED_BACKEND.iq_lanes_f64();

#[cfg(signal_kit_simd_avx512)]
type SelectedBackend = x86::Avx512Backend;

#[cfg(signal_kit_simd_avx2)]
type SelectedBackend = x86::Avx2Backend;

#[cfg(signal_kit_simd_sse2)]
type SelectedBackend = x86::Sse2Backend;

#[cfg(any(signal_kit_simd_scalar, not(any(signal_kit_simd_avx512, signal_kit_simd_avx2, signal_kit_simd_sse2))))]
type SelectedBackend = scalar::ScalarBackend;

/// Scalar type supported by the IQ SIMD API.
///
/// This trait is sealed so only `f32` and `f64` can use the generic SIMD
/// functions.
pub trait IqSample: private::Sealed + Float + bytemuck::Pod {
    /// Returns complex IQ samples processed per register for `backend`.
    #[doc(hidden)]
    fn iq_lanes_for_backend(backend: SimdBackend) -> usize;

    /// Writes `left + right` into `output` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_add_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]);

    /// Adds `input` into `target` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_add_in_place(target: &mut [Complex<Self>], input: &[Complex<Self>]);

    /// Scales `target` by `scale` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_scale_in_place(target: &mut [Complex<Self>], scale: Self);

    /// Adds `scale * input` into `target` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_add_scaled_in_place(target: &mut [Complex<Self>], scale: Complex<Self>, input: &[Complex<Self>]);

    /// Writes `left * right` into `output` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]);

    /// Writes `conjugate(left) * right` into `output` using the selected backend.
    #[doc(hidden)]
    fn selected_iq_conjugate_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]);

    /// Returns the selected-backend sum of `conjugate(left) * right`.
    #[doc(hidden)]
    fn selected_iq_conjugate_multiply_sum(left: &[Complex<Self>], right: &[Complex<Self>]) -> Complex<Self>;

    /// Returns the selected-backend sum of squared magnitudes.
    #[doc(hidden)]
    fn selected_iq_sum_squared_magnitudes(input: &[Complex<Self>]) -> Self;

    /// Writes selected-backend per-sample magnitudes into `output`.
    #[doc(hidden)]
    fn selected_iq_calculate_magnitudes_to(input: &[Complex<Self>], output: &mut [Self]);

    /// Writes a selected-backend direct IQ convolution range into `output`.
    #[doc(hidden)]
    fn selected_iq_convolve_range_to(input: &[Complex<Self>], kernel: &[Complex<Self>], full_output_start: usize, output: &mut [Complex<Self>]);
}

impl IqSample for f32 {
    fn iq_lanes_for_backend(backend: SimdBackend) -> usize {
        backend.iq_lanes_f32()
    }

    fn selected_iq_add_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_add_to(left, right, output);
    }

    fn selected_iq_add_in_place(target: &mut [Complex<Self>], input: &[Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_add_in_place(target, input);
    }

    fn selected_iq_scale_in_place(target: &mut [Complex<Self>], scale: Self) {
        <SelectedBackend as IqBackend>::iq_scale_in_place(target, scale);
    }

    fn selected_iq_add_scaled_in_place(target: &mut [Complex<Self>], scale: Complex<Self>, input: &[Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_add_scaled_in_place(target, scale, input);
    }

    fn selected_iq_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_multiply_to(left, right, output);
    }

    fn selected_iq_conjugate_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_conjugate_multiply_to(left, right, output);
    }

    fn selected_iq_conjugate_multiply_sum(left: &[Complex<Self>], right: &[Complex<Self>]) -> Complex<Self> {
        <SelectedBackend as IqBackend>::iq_conjugate_multiply_sum(left, right)
    }

    fn selected_iq_sum_squared_magnitudes(input: &[Complex<Self>]) -> Self {
        <SelectedBackend as IqBackend>::iq_sum_squared_magnitudes(input)
    }

    fn selected_iq_calculate_magnitudes_to(input: &[Complex<Self>], output: &mut [Self]) {
        <SelectedBackend as IqBackend>::iq_calculate_magnitudes_to(input, output);
    }

    fn selected_iq_convolve_range_to(input: &[Complex<Self>], kernel: &[Complex<Self>], full_output_start: usize, output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackend>::iq_convolve_range_to(input, kernel, full_output_start, output);
    }
}

impl IqSample for f64 {
    fn iq_lanes_for_backend(backend: SimdBackend) -> usize {
        backend.iq_lanes_f64()
    }

    fn selected_iq_add_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_add_to(left, right, output);
    }

    fn selected_iq_add_in_place(target: &mut [Complex<Self>], input: &[Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_add_in_place(target, input);
    }

    fn selected_iq_scale_in_place(target: &mut [Complex<Self>], scale: Self) {
        <SelectedBackend as IqBackendF64>::iq_scale_in_place(target, scale);
    }

    fn selected_iq_add_scaled_in_place(target: &mut [Complex<Self>], scale: Complex<Self>, input: &[Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_add_scaled_in_place(target, scale, input);
    }

    fn selected_iq_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_multiply_to(left, right, output);
    }

    fn selected_iq_conjugate_multiply_to(left: &[Complex<Self>], right: &[Complex<Self>], output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_conjugate_multiply_to(left, right, output);
    }

    fn selected_iq_conjugate_multiply_sum(left: &[Complex<Self>], right: &[Complex<Self>]) -> Complex<Self> {
        <SelectedBackend as IqBackendF64>::iq_conjugate_multiply_sum(left, right)
    }

    fn selected_iq_sum_squared_magnitudes(input: &[Complex<Self>]) -> Self {
        <SelectedBackend as IqBackendF64>::iq_sum_squared_magnitudes(input)
    }

    fn selected_iq_calculate_magnitudes_to(input: &[Complex<Self>], output: &mut [Self]) {
        <SelectedBackend as IqBackendF64>::iq_calculate_magnitudes_to(input, output);
    }

    fn selected_iq_convolve_range_to(input: &[Complex<Self>], kernel: &[Complex<Self>], full_output_start: usize, output: &mut [Complex<Self>]) {
        <SelectedBackend as IqBackendF64>::iq_convolve_range_to(input, kernel, full_output_start, output);
    }
}

mod private {
    /// Seals the public SIMD sample trait to crate-supported scalar types.
    pub trait Sealed {}

    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Returns a plan for the backend selected at compile time.
pub const fn selected_iq_vector_plan() -> IqVectorPlan {
    IqVectorPlan::selected()
}

/// Writes `left + right` into `output` without allocating.
pub fn iq_add_to<T: IqSample>(left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
    assert_binary_output_len(left, right, output);
    T::selected_iq_add_to(left, right, output);
}

/// Adds `input` into `target` without allocating.
pub fn iq_add_in_place<T: IqSample>(target: &mut [Complex<T>], input: &[Complex<T>]) {
    assert_equal_len(target, input, "target", "input");
    T::selected_iq_add_in_place(target, input);
}

/// Scales `target` by `scale` without allocating.
pub fn iq_scale_in_place<T: IqSample>(target: &mut [Complex<T>], scale: T) {
    T::selected_iq_scale_in_place(target, scale);
}

/// Adds `scale * input` into `target` without allocating.
pub fn iq_add_scaled_in_place<T: IqSample>(target: &mut [Complex<T>], scale: Complex<T>, input: &[Complex<T>]) {
    assert_equal_len(target, input, "target", "input");
    T::selected_iq_add_scaled_in_place(target, scale, input);
}

/// Writes `left * right` into `output` without allocating.
pub fn iq_multiply_to<T: IqSample>(left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
    assert_binary_output_len(left, right, output);
    T::selected_iq_multiply_to(left, right, output);
}

/// Writes `conjugate(left) * right` into `output` without allocating.
pub fn iq_conjugate_multiply_to<T: IqSample>(left: &[Complex<T>], right: &[Complex<T>], output: &mut [Complex<T>]) {
    assert_binary_output_len(left, right, output);
    T::selected_iq_conjugate_multiply_to(left, right, output);
}

/// Returns the sum of `conjugate(left) * right` for each pair of IQ samples.
pub fn iq_conjugate_multiply_sum<T: IqSample>(left: &[Complex<T>], right: &[Complex<T>]) -> Complex<T> {
    assert_equal_len(left, right, "left", "right");
    T::selected_iq_conjugate_multiply_sum(left, right)
}

/// Returns the sum of squared magnitudes for all IQ samples.
pub fn iq_sum_squared_magnitudes<T: IqSample>(input: &[Complex<T>]) -> T {
    T::selected_iq_sum_squared_magnitudes(input)
}

/// Writes per-sample magnitudes into `output` without allocating.
pub fn iq_calculate_magnitudes_to<T: IqSample>(input: &[Complex<T>], output: &mut [T]) {
    assert_output_len(input.len(), output.len(), "magnitude output");
    T::selected_iq_calculate_magnitudes_to(input, output);
}

/// Writes the full direct IQ convolution into `output` without allocating.
///
/// # Panics
/// Panics when `input` or `kernel` is empty, or when `output.len()` is not
/// `input.len() + kernel.len() - 1`.
pub fn iq_convolve_full_to<T: IqSample>(input: &[Complex<T>], kernel: &[Complex<T>], output: &mut [Complex<T>]) {
    assert_full_convolution_output_len(input, kernel, output);
    T::selected_iq_convolve_range_to(input, kernel, 0, output);
}

/// Writes a direct IQ convolution range from the full output into `output`.
pub(crate) fn iq_convolve_range_to<T: IqSample>(
    input: &[Complex<T>],
    kernel: &[Complex<T>],
    full_output_start: usize,
    output: &mut [Complex<T>],
) {
    assert_convolution_range(input, kernel, full_output_start, output.len());
    T::selected_iq_convolve_range_to(input, kernel, full_output_start, output);
}

/// Frequency shifts IQ blocks while preserving phase across calls.
#[derive(Debug, Clone)]
pub struct FrequencyShifter<T = f32> {
    radians_per_sample: T,
    phase: T,
    oscillator: Vec<Complex<T>>,
    scratch: Vec<Complex<T>>,
}

impl<T: IqSample> FrequencyShifter<T> {
    /// Creates a frequency shifter from offset and sample-rate values in hertz.
    pub fn new(freq_offset_hz: T, sample_rate_hz: T) -> Self {
        assert!(sample_rate_hz > T::zero(), "sample_rate_hz must be positive");
        Self {
            radians_per_sample: T::from(std::f64::consts::TAU).unwrap() * freq_offset_hz / sample_rate_hz,
            phase: T::zero(),
            oscillator: Vec::new(),
            scratch: Vec::new(),
        }
    }

    /// Returns the current oscillator phase in radians.
    pub fn phase(&self) -> T {
        self.phase
    }

    /// Shifts a block in place and preserves phase for the next block.
    pub fn process_block(&mut self, samples: &mut [Complex<T>]) {
        if samples.is_empty() {
            return;
        }
        self.fill_oscillator(samples.len());
        self.scratch.clear();
        self.scratch.extend_from_slice(samples);
        iq_multiply_to(&self.scratch, &self.oscillator, samples);
    }

    /// Fills the reusable oscillator buffer for one output block.
    fn fill_oscillator(&mut self, len: usize) {
        self.oscillator.clear();
        self.oscillator.reserve(len);
        let mut phase = self.phase;
        for _ in 0..len {
            self.oscillator.push(unit_complex(phase));
            phase = wrap_phase(phase + self.radians_per_sample);
        }
        self.phase = phase;
    }
}

/// Applies a complex FIR filter while preserving delay state across calls.
#[derive(Debug, Clone)]
pub struct FirFilter<T = f32> {
    taps: Vec<Complex<T>>,
    delay: Vec<Complex<T>>,
    reversed_taps: Vec<Complex<T>>,
    history: Vec<Complex<T>>,
    input_scratch: Vec<Complex<T>>,
}

impl<T: IqSample> FirFilter<T> {
    /// Creates a streaming FIR filter with `taps[0]` applied to the current sample.
    pub fn new(taps: Vec<Complex<T>>) -> Self {
        assert!(!taps.is_empty(), "taps must not be empty");
        let delay = vec![zero_complex(); taps.len() - 1];
        let reversed_taps = taps.iter().rev().copied().collect();
        Self {
            taps,
            delay,
            reversed_taps,
            history: Vec::new(),
            input_scratch: Vec::new(),
        }
    }

    /// Returns the FIR tap slice.
    pub fn taps(&self) -> &[Complex<T>] {
        &self.taps
    }

    /// Returns the current delay-state slice with the newest sample first.
    pub fn delay(&self) -> &[Complex<T>] {
        &self.delay
    }

    /// Filters `input` into `output` while preserving delay state.
    pub fn process_block_to(&mut self, input: &[Complex<T>], output: &mut [Complex<T>]) {
        assert_output_len(input.len(), output.len(), "fir output");
        self.write_history(input);
        self.write_block_output(output);
        self.update_delay_from_history();
    }

    /// Filters a block in place while preserving delay state.
    pub fn process_block_in_place(&mut self, samples: &mut [Complex<T>]) {
        self.input_scratch.clear();
        self.input_scratch.extend_from_slice(samples);
        self.write_history_from_scratch();
        self.write_block_output(samples);
        self.update_delay_from_history();
    }

    /// Filters one sample and advances the delay line.
    pub fn process_sample(&mut self, sample: Complex<T>) -> Complex<T> {
        let output = self.calculate_output(sample);
        self.push_delay(sample);
        output
    }

    /// Calculates the FIR output for one sample using the current delay line.
    fn calculate_output(&self, sample: Complex<T>) -> Complex<T> {
        let mut sum = self.taps[0] * sample;
        for (tap, delayed_sample) in self.taps.iter().skip(1).zip(self.delay.iter()) {
            sum = sum + *tap * *delayed_sample;
        }
        sum
    }

    /// Pushes one sample into the delay line.
    fn push_delay(&mut self, sample: Complex<T>) {
        if self.delay.is_empty() {
            return;
        }
        self.delay.rotate_right(1);
        self.delay[0] = sample;
    }

    /// Builds chronological history from the delay line and input slice.
    fn write_history(&mut self, input: &[Complex<T>]) {
        self.history.clear();
        self.history.extend(self.delay.iter().rev().copied());
        self.history.extend_from_slice(input);
    }

    /// Builds chronological history from the reusable in-place scratch buffer.
    fn write_history_from_scratch(&mut self) {
        self.history.clear();
        self.history.extend(self.delay.iter().rev().copied());
        self.history.extend_from_slice(&self.input_scratch);
    }

    /// Writes the current history block through the SIMD convolution kernel.
    fn write_block_output(&mut self, output: &mut [Complex<T>]) {
        if output.is_empty() {
            return;
        }
        iq_convolve_range_to(&self.history, &self.reversed_taps, self.taps.len() - 1, output);
    }

    /// Copies the newest samples from chronological history into delay order.
    fn update_delay_from_history(&mut self) {
        if self.delay.is_empty() {
            return;
        }
        let delay_start = self.history.len() - self.delay.len();
        for (delay_sample, history_sample) in self.delay.iter_mut().zip(self.history[delay_start..].iter().rev()) {
            *delay_sample = *history_sample;
        }
    }
}

trait IqBackend {
    /// Writes `left + right` into `output`.
    fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]);

    /// Adds `input` into `target`.
    fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]);

    /// Scales `target` by `scale`.
    fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32);

    /// Adds `scale * input` into `target`.
    fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]);

    /// Writes `left * right` into `output`.
    fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]);

    /// Writes `conjugate(left) * right` into `output`.
    fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]);

    /// Returns sum of `conjugate(left) * right`.
    fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32>;

    /// Returns sum of squared magnitudes.
    fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32;

    /// Writes per-sample magnitudes into `output`.
    fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]);

    /// Writes a direct IQ convolution range into `output`.
    fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]);
}

trait IqBackendF64 {
    /// Writes `left + right` into `output`.
    fn iq_add_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]);

    /// Adds `input` into `target`.
    fn iq_add_in_place(target: &mut [Complex<f64>], input: &[Complex<f64>]);

    /// Scales `target` by `scale`.
    fn iq_scale_in_place(target: &mut [Complex<f64>], scale: f64);

    /// Adds `scale * input` into `target`.
    fn iq_add_scaled_in_place(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]);

    /// Writes `left * right` into `output`.
    fn iq_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]);

    /// Writes `conjugate(left) * right` into `output`.
    fn iq_conjugate_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]);

    /// Returns sum of `conjugate(left) * right`.
    fn iq_conjugate_multiply_sum(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64>;

    /// Returns sum of squared magnitudes.
    fn iq_sum_squared_magnitudes(input: &[Complex<f64>]) -> f64;

    /// Writes per-sample magnitudes into `output`.
    fn iq_calculate_magnitudes_to(input: &[Complex<f64>], output: &mut [f64]);

    /// Writes a direct IQ convolution range into `output`.
    fn iq_convolve_range_to(input: &[Complex<f64>], kernel: &[Complex<f64>], full_output_start: usize, output: &mut [Complex<f64>]);
}

/// Returns the build-selected backend as a const value.
const fn selected_backend() -> SimdBackend {
    #[cfg(signal_kit_simd_avx512)]
    {
        return SimdBackend::Avx512F;
    }
    #[cfg(signal_kit_simd_avx2)]
    {
        return SimdBackend::Avx2Fma;
    }
    #[cfg(signal_kit_simd_sse2)]
    {
        return SimdBackend::Sse2;
    }
    #[allow(unreachable_code)]
    SimdBackend::Scalar
}

/// Asserts that two input slices and one output slice have matching lengths.
fn assert_binary_output_len<T>(left: &[Complex<T>], right: &[Complex<T>], output: &[Complex<T>]) {
    assert_equal_len(left, right, "left", "right");
    assert_output_len(left.len(), output.len(), "output");
}

/// Asserts that two slices have matching lengths.
fn assert_equal_len<T, U>(left: &[T], right: &[U], left_name: &str, right_name: &str) {
    assert_eq!(left.len(), right.len(), "{left_name} and {right_name} lengths must match");
}

/// Asserts that an output length matches the expected length.
fn assert_output_len(expected: usize, actual: usize, output_name: &str) {
    assert_eq!(expected, actual, "{output_name} length must match input length");
}

/// Asserts that the full convolution output length is exact.
fn assert_full_convolution_output_len<T>(input: &[Complex<T>], kernel: &[Complex<T>], output: &[Complex<T>]) {
    let expected = full_convolution_len(input.len(), kernel.len());
    assert_eq!(
        expected,
        output.len(),
        "full convolution output length must equal input.len() + kernel.len() - 1"
    );
}

/// Asserts that a convolution range fits inside the full output.
fn assert_convolution_range<T>(input: &[Complex<T>], kernel: &[Complex<T>], full_output_start: usize, output_len: usize) {
    let full_len = full_convolution_len(input.len(), kernel.len());
    let range_end = full_output_start
        .checked_add(output_len)
        .expect("convolution output range length overflowed");
    assert!(range_end <= full_len, "convolution output range must fit within full output");
}

/// Returns the full convolution output length.
fn full_convolution_len(input_len: usize, kernel_len: usize) -> usize {
    assert!(input_len > 0, "input must not be empty");
    assert!(kernel_len > 0, "kernel must not be empty");
    input_len
        .checked_add(kernel_len)
        .and_then(|length| length.checked_sub(1))
        .expect("full convolution length overflowed")
}

/// Writes each requested convolution output using an overlap summation function.
fn write_convolution_range_to<T, SumOverlap>(
    input: &[Complex<T>],
    kernel: &[Complex<T>],
    full_output_start: usize,
    output: &mut [Complex<T>],
    mut sum_overlap: SumOverlap,
) where
    T: IqSample,
    SumOverlap: FnMut(&[Complex<T>], &[Complex<T>]) -> Complex<T>,
{
    for (output_offset, output_sample) in output.iter_mut().enumerate() {
        let full_output_index = full_output_start + output_offset;
        let (input_overlap, kernel_overlap) = convolution_overlap_slices(input, kernel, full_output_index);
        *output_sample = sum_overlap(input_overlap, kernel_overlap);
    }
}

/// Returns the overlapping input and kernel slices for one full output index.
fn convolution_overlap_slices<'input, 'kernel, T: IqSample>(
    input: &'input [Complex<T>],
    kernel: &'kernel [Complex<T>],
    full_index: usize,
) -> (&'input [Complex<T>], &'kernel [Complex<T>]) {
    let overlap = convolution_overlap(full_index, input.len(), kernel.len());
    let input_end = overlap.input_start + overlap.len;
    let kernel_end = overlap.kernel_start + overlap.len;
    (&input[overlap.input_start..input_end], &kernel[overlap.kernel_start..kernel_end])
}

/// Returns the overlap geometry for one full convolution output index.
fn convolution_overlap(full_index: usize, input_len: usize, kernel_len: usize) -> ConvolutionOverlap {
    let kernel_start = (kernel_len - 1).saturating_sub(full_index);
    let kernel_end = kernel_overlap_end(full_index, input_len, kernel_len);
    ConvolutionOverlap {
        input_start: full_index + kernel_start + 1 - kernel_len,
        kernel_start,
        len: kernel_end - kernel_start,
    }
}

/// Returns the exclusive kernel overlap end for one full output index.
fn kernel_overlap_end(full_index: usize, input_len: usize, kernel_len: usize) -> usize {
    let full_len = input_len + kernel_len - 1;
    kernel_len.min(full_len - full_index)
}

/// Stores contiguous overlap geometry for direct convolution.
#[derive(Debug, Clone, Copy)]
struct ConvolutionOverlap {
    input_start: usize,
    kernel_start: usize,
    len: usize,
}

/// Sums an interleaved `[re, im, ...]` slice into one complex value.
fn sum_interleaved_complex_lanes<T: IqSample>(values: &[T]) -> Complex<T> {
    let mut sum = zero_complex::<T>();
    for lane in values.chunks_exact(2) {
        sum = sum + Complex::new(lane[0], lane[1]);
    }
    sum
}

/// Multiplies the conjugate of `left` by `right`.
fn conjugate_multiply_sample<T: IqSample>(left: Complex<T>, right: Complex<T>) -> Complex<T> {
    Complex::new(left.re * right.re + left.im * right.im, left.re * right.im - left.im * right.re)
}

/// Builds a zero complex sample.
fn zero_complex<T: IqSample>() -> Complex<T> {
    Complex::new(T::zero(), T::zero())
}

/// Casts complex f32 samples to their interleaved f32 representation.
fn as_f32_slice(input: &[Complex<f32>]) -> &[f32] {
    cast_slice(input)
}

/// Casts mutable complex f32 samples to their interleaved f32 representation.
fn as_f32_slice_mut(input: &mut [Complex<f32>]) -> &mut [f32] {
    cast_slice_mut(input)
}

/// Casts complex f64 samples to their interleaved f64 representation.
fn as_f64_slice(input: &[Complex<f64>]) -> &[f64] {
    cast_slice(input)
}

/// Casts mutable complex f64 samples to their interleaved f64 representation.
fn as_f64_slice_mut(input: &mut [Complex<f64>]) -> &mut [f64] {
    cast_slice_mut(input)
}

/// Returns the largest vectorized prefix length for a lane count.
fn vectorized_end(len: usize, lanes: usize) -> usize {
    len - (len % lanes)
}

/// Builds a unit complex value for the provided phase.
fn unit_complex<T: IqSample>(phase: T) -> Complex<T> {
    let (sin_phase, cos_phase) = phase.sin_cos();
    Complex::new(cos_phase, sin_phase)
}

/// Wraps phase into one turn to keep oscillator state bounded.
fn wrap_phase<T: IqSample>(phase: T) -> T {
    let turn = T::from(std::f64::consts::TAU).unwrap();
    let wrapped = phase % turn;
    if wrapped < T::zero() { wrapped + turn } else { wrapped }
}

mod scalar {
    #![allow(dead_code)]

    use super::*;

    /// Scalar backend marker.
    pub(super) struct ScalarBackend;

    impl IqBackend for ScalarBackend {
        fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            iq_add_to(left, right, output);
        }

        fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
            iq_add_in_place(target, input);
        }

        fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32) {
            iq_scale_in_place(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
            iq_add_scaled_in_place(target, scale, input);
        }

        fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            iq_multiply_to(left, right, output);
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            iq_conjugate_multiply_to(left, right, output);
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
            iq_conjugate_multiply_sum(left, right)
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
            iq_sum_squared_magnitudes(input)
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
            iq_calculate_magnitudes_to(input, output);
        }

        fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
            iq_convolve_range_to(input, kernel, full_output_start, output);
        }
    }

    impl IqBackendF64 for ScalarBackend {
        fn iq_add_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            iq_add_to_f64(left, right, output);
        }

        fn iq_add_in_place(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
            iq_add_in_place_f64(target, input);
        }

        fn iq_scale_in_place(target: &mut [Complex<f64>], scale: f64) {
            iq_scale_in_place_f64(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
            iq_add_scaled_in_place_f64(target, scale, input);
        }

        fn iq_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            iq_multiply_to_f64(left, right, output);
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            iq_conjugate_multiply_to_f64(left, right, output);
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
            iq_conjugate_multiply_sum_f64(left, right)
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f64>]) -> f64 {
            iq_sum_squared_magnitudes_f64(input)
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f64>], output: &mut [f64]) {
            iq_calculate_magnitudes_to_f64(input, output);
        }

        fn iq_convolve_range_to(input: &[Complex<f64>], kernel: &[Complex<f64>], full_output_start: usize, output: &mut [Complex<f64>]) {
            iq_convolve_range_to_f64(input, kernel, full_output_start, output);
        }
    }

    /// Writes scalar complex additions into `output`.
    pub(super) fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = *left_sample + *right_sample;
        }
    }

    /// Adds scalar complex samples into `target`.
    pub(super) fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
        for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
            *target_sample += *input_sample;
        }
    }

    /// Scales scalar complex samples in place.
    pub(super) fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32) {
        for target_sample in target {
            *target_sample *= scale;
        }
    }

    /// Adds scalar `scale * input` products into `target`.
    pub(super) fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
        for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
            *target_sample += scale * *input_sample;
        }
    }

    /// Writes scalar complex products into `output`.
    pub(super) fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = *left_sample * *right_sample;
        }
    }

    /// Writes scalar conjugate-multiply products into `output`.
    pub(super) fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = conjugate_multiply_sample(*left_sample, *right_sample);
        }
    }

    /// Returns a scalar sum of conjugate-multiply products.
    pub(super) fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        conjugate_multiply_sum_tail(left, right, 0)
    }

    /// Returns a scalar sum of squared magnitudes.
    pub(super) fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
        input.iter().map(|sample| sample.norm_sqr()).sum()
    }

    /// Writes scalar magnitudes into `output`.
    pub(super) fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
        for (sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = sample.norm();
        }
    }

    /// Writes scalar direct convolution outputs into `output`.
    pub(super) fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| {
            multiply_sum_tail(input_overlap, kernel_overlap, 0)
        });
    }

    /// Writes scalar f32 additions from `start` to the end.
    pub(super) fn add_f32_tail(left: &[f32], right: &[f32], output: &mut [f32], start: usize) {
        for index in start..left.len() {
            output[index] = left[index] + right[index];
        }
    }

    /// Adds scalar f32 samples into `target` from `start` to the end.
    pub(super) fn add_assign_f32_tail(target: &mut [f32], input: &[f32], start: usize) {
        for index in start..target.len() {
            target[index] += input[index];
        }
    }

    /// Adds scalar complex add-scaled results from `start` to the end.
    pub(super) fn add_scaled_tail(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>], start: usize) {
        for index in start..target.len() {
            target[index] += scale * input[index];
        }
    }

    /// Writes scalar complex products from `start` to the end.
    pub(super) fn multiply_tail(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>], start: usize) {
        for index in start..left.len() {
            output[index] = left[index] * right[index];
        }
    }

    /// Writes scalar conjugate-multiply products from `start` to the end.
    pub(super) fn conjugate_multiply_tail(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>], start: usize) {
        for index in start..left.len() {
            output[index] = conjugate_multiply_sample(left[index], right[index]);
        }
    }

    /// Returns a scalar f32 squared-magnitude tail sum from `start` to the end.
    pub(super) fn sum_squared_f32_tail(input: &[f32], start: usize) -> f32 {
        input.iter().skip(start).map(|value| value * value).sum()
    }

    /// Writes scalar magnitudes from `start` to the end.
    pub(super) fn calculate_magnitudes_tail(input: &[Complex<f32>], output: &mut [f32], start: usize) {
        for index in start..input.len() {
            output[index] = input[index].norm();
        }
    }

    /// Returns a scalar sum of complex products from `start` to the end.
    pub(super) fn multiply_sum_tail(left: &[Complex<f32>], right: &[Complex<f32>], start: usize) -> Complex<f32> {
        let mut sum = Complex::new(0.0, 0.0);
        for index in start..left.len() {
            sum += left[index] * right[index];
        }
        sum
    }

    /// Returns a scalar conjugate-multiply sum from `start` to the end.
    pub(super) fn conjugate_multiply_sum_tail(left: &[Complex<f32>], right: &[Complex<f32>], start: usize) -> Complex<f32> {
        let mut sum = Complex::new(0.0, 0.0);
        for index in start..left.len() {
            sum += conjugate_multiply_sample(left[index], right[index]);
        }
        sum
    }

    /// Writes scalar f64 complex additions into `output`.
    pub(super) fn iq_add_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = *left_sample + *right_sample;
        }
    }

    /// Adds scalar f64 complex samples into `target`.
    pub(super) fn iq_add_in_place_f64(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
        for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
            *target_sample += *input_sample;
        }
    }

    /// Scales scalar f64 complex samples in place.
    pub(super) fn iq_scale_in_place_f64(target: &mut [Complex<f64>], scale: f64) {
        for target_sample in target {
            *target_sample *= scale;
        }
    }

    /// Adds scalar f64 `scale * input` products into `target`.
    pub(super) fn iq_add_scaled_in_place_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
        for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
            *target_sample += scale * *input_sample;
        }
    }

    /// Writes scalar f64 complex products into `output`.
    pub(super) fn iq_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = *left_sample * *right_sample;
        }
    }

    /// Writes scalar f64 conjugate-multiply products into `output`.
    pub(super) fn iq_conjugate_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
            *output_sample = conjugate_multiply_sample(*left_sample, *right_sample);
        }
    }

    /// Returns a scalar f64 sum of conjugate-multiply products.
    pub(super) fn iq_conjugate_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        conjugate_multiply_sum_tail_f64(left, right, 0)
    }

    /// Returns a scalar f64 sum of squared magnitudes.
    pub(super) fn iq_sum_squared_magnitudes_f64(input: &[Complex<f64>]) -> f64 {
        input.iter().map(|sample| sample.norm_sqr()).sum()
    }

    /// Writes scalar f64 magnitudes into `output`.
    pub(super) fn iq_calculate_magnitudes_to_f64(input: &[Complex<f64>], output: &mut [f64]) {
        for (sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = sample.norm();
        }
    }

    /// Writes scalar f64 direct convolution outputs into `output`.
    pub(super) fn iq_convolve_range_to_f64(
        input: &[Complex<f64>],
        kernel: &[Complex<f64>],
        full_output_start: usize,
        output: &mut [Complex<f64>],
    ) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| {
            multiply_sum_tail_f64(input_overlap, kernel_overlap, 0)
        });
    }

    /// Writes scalar f64 additions from `start` to the end.
    pub(super) fn add_f64_tail(left: &[f64], right: &[f64], output: &mut [f64], start: usize) {
        for index in start..left.len() {
            output[index] = left[index] + right[index];
        }
    }

    /// Adds scalar f64 samples into `target` from `start` to the end.
    pub(super) fn add_assign_f64_tail(target: &mut [f64], input: &[f64], start: usize) {
        for index in start..target.len() {
            target[index] += input[index];
        }
    }

    /// Adds scalar f64 complex add-scaled results from `start` to the end.
    pub(super) fn add_scaled_tail_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>], start: usize) {
        for index in start..target.len() {
            target[index] += scale * input[index];
        }
    }

    /// Writes scalar f64 complex products from `start` to the end.
    pub(super) fn multiply_tail_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>], start: usize) {
        for index in start..left.len() {
            output[index] = left[index] * right[index];
        }
    }

    /// Writes scalar f64 conjugate-multiply products from `start` to the end.
    pub(super) fn conjugate_multiply_tail_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>], start: usize) {
        for index in start..left.len() {
            output[index] = conjugate_multiply_sample(left[index], right[index]);
        }
    }

    /// Returns a scalar f64 squared-magnitude tail sum from `start` to the end.
    pub(super) fn sum_squared_f64_tail(input: &[f64], start: usize) -> f64 {
        input.iter().skip(start).map(|value| value * value).sum()
    }

    /// Writes scalar f64 magnitudes from `start` to the end.
    pub(super) fn calculate_magnitudes_tail_f64(input: &[Complex<f64>], output: &mut [f64], start: usize) {
        for index in start..input.len() {
            output[index] = input[index].norm();
        }
    }

    /// Returns a scalar f64 sum of complex products from `start` to the end.
    pub(super) fn multiply_sum_tail_f64(left: &[Complex<f64>], right: &[Complex<f64>], start: usize) -> Complex<f64> {
        let mut sum = Complex::new(0.0, 0.0);
        for index in start..left.len() {
            sum += left[index] * right[index];
        }
        sum
    }

    /// Returns a scalar f64 conjugate-multiply sum from `start` to the end.
    pub(super) fn conjugate_multiply_sum_tail_f64(left: &[Complex<f64>], right: &[Complex<f64>], start: usize) -> Complex<f64> {
        let mut sum = Complex::new(0.0, 0.0);
        for index in start..left.len() {
            sum += conjugate_multiply_sample(left[index], right[index]);
        }
        sum
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #![allow(dead_code)]

    use super::*;

    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;

    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const SSE_F32_LANES: usize = 4;
    const AVX2_F32_LANES: usize = 8;
    const AVX512_F32_LANES: usize = 16;
    const SSE_F64_LANES: usize = 2;
    const AVX2_F64_LANES: usize = 4;
    const AVX512_F64_LANES: usize = 8;
    const SSE_COMPLEX_LANES: usize = 2;
    const AVX2_COMPLEX_LANES: usize = 4;
    const AVX512_COMPLEX_LANES: usize = 8;
    const SSE_COMPLEX_LANES_F64: usize = 1;
    const AVX2_COMPLEX_LANES_F64: usize = 2;
    const AVX512_COMPLEX_LANES_F64: usize = 4;
    const SSE_SIGN: [f32; SSE_F32_LANES] = [-1.0, 1.0, -1.0, 1.0];
    const AVX2_SIGN: [f32; AVX2_F32_LANES] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    const AVX512_SIGN: [f32; AVX512_F32_LANES] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    const SSE_CONJUGATE_SIGN: [f32; SSE_F32_LANES] = [1.0, -1.0, 1.0, -1.0];
    const AVX2_CONJUGATE_SIGN: [f32; AVX2_F32_LANES] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    const AVX512_CONJUGATE_SIGN: [f32; AVX512_F32_LANES] =
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
    const SSE_SIGN_F64: [f64; SSE_F64_LANES] = [-1.0, 1.0];
    const AVX2_SIGN_F64: [f64; AVX2_F64_LANES] = [-1.0, 1.0, -1.0, 1.0];
    const AVX512_SIGN_F64: [f64; AVX512_F64_LANES] = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0];
    const SSE_CONJUGATE_SIGN_F64: [f64; SSE_F64_LANES] = [1.0, -1.0];
    const AVX2_CONJUGATE_SIGN_F64: [f64; AVX2_F64_LANES] = [1.0, -1.0, 1.0, -1.0];
    const AVX512_CONJUGATE_SIGN_F64: [f64; AVX512_F64_LANES] = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

    /// SSE2 backend marker.
    pub(super) struct Sse2Backend;

    /// AVX2/FMA backend marker.
    pub(super) struct Avx2Backend;

    /// AVX-512F backend marker.
    pub(super) struct Avx512Backend;

    impl IqBackend for Sse2Backend {
        fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { sse2_iq_add_to(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
            unsafe { sse2_iq_add_in_place(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32) {
            scalar::iq_scale_in_place(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
            unsafe { sse2_iq_add_scaled_in_place(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { sse2_iq_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { sse2_iq_conjugate_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
            unsafe { sse2_iq_conjugate_multiply_sum(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
            unsafe { sse2_iq_sum_squared_magnitudes(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
            unsafe { sse2_iq_calculate_magnitudes_to(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
            unsafe { sse2_iq_convolve_range_to(input, kernel, full_output_start, output) };
        }
    }

    impl IqBackend for Avx2Backend {
        fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx2_iq_add_to(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
            unsafe { avx2_iq_add_in_place(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32) {
            scalar::iq_scale_in_place(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
            unsafe { avx2_iq_add_scaled_in_place(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx2_iq_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx2_iq_conjugate_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
            unsafe { avx2_iq_conjugate_multiply_sum(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
            unsafe { avx2_iq_sum_squared_magnitudes(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
            unsafe { avx2_iq_calculate_magnitudes_to(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
            unsafe { avx2_iq_convolve_range_to(input, kernel, full_output_start, output) };
        }
    }

    impl IqBackend for Avx512Backend {
        fn iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx512_iq_add_to(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
            unsafe { avx512_iq_add_in_place(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f32>], scale: f32) {
            scalar::iq_scale_in_place(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
            unsafe { avx512_iq_add_scaled_in_place(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx512_iq_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
            unsafe { avx512_iq_conjugate_multiply_to(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
            unsafe { avx512_iq_conjugate_multiply_sum(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
            unsafe { avx512_iq_sum_squared_magnitudes(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
            unsafe { avx512_iq_calculate_magnitudes_to(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
            unsafe { avx512_iq_convolve_range_to(input, kernel, full_output_start, output) };
        }
    }

    impl IqBackendF64 for Sse2Backend {
        fn iq_add_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { sse2_iq_add_to_f64(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
            unsafe { sse2_iq_add_in_place_f64(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f64>], scale: f64) {
            scalar::iq_scale_in_place_f64(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
            unsafe { sse2_iq_add_scaled_in_place_f64(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { sse2_iq_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { sse2_iq_conjugate_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
            unsafe { sse2_iq_conjugate_multiply_sum_f64(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f64>]) -> f64 {
            unsafe { sse2_iq_sum_squared_magnitudes_f64(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f64>], output: &mut [f64]) {
            unsafe { sse2_iq_calculate_magnitudes_to_f64(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f64>], kernel: &[Complex<f64>], full_output_start: usize, output: &mut [Complex<f64>]) {
            unsafe { sse2_iq_convolve_range_to_f64(input, kernel, full_output_start, output) };
        }
    }

    impl IqBackendF64 for Avx2Backend {
        fn iq_add_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx2_iq_add_to_f64(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
            unsafe { avx2_iq_add_in_place_f64(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f64>], scale: f64) {
            scalar::iq_scale_in_place_f64(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
            unsafe { avx2_iq_add_scaled_in_place_f64(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx2_iq_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx2_iq_conjugate_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
            unsafe { avx2_iq_conjugate_multiply_sum_f64(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f64>]) -> f64 {
            unsafe { avx2_iq_sum_squared_magnitudes_f64(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f64>], output: &mut [f64]) {
            unsafe { avx2_iq_calculate_magnitudes_to_f64(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f64>], kernel: &[Complex<f64>], full_output_start: usize, output: &mut [Complex<f64>]) {
            unsafe { avx2_iq_convolve_range_to_f64(input, kernel, full_output_start, output) };
        }
    }

    impl IqBackendF64 for Avx512Backend {
        fn iq_add_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx512_iq_add_to_f64(left, right, output) };
        }

        fn iq_add_in_place(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
            unsafe { avx512_iq_add_in_place_f64(target, input) };
        }

        fn iq_scale_in_place(target: &mut [Complex<f64>], scale: f64) {
            scalar::iq_scale_in_place_f64(target, scale);
        }

        fn iq_add_scaled_in_place(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
            unsafe { avx512_iq_add_scaled_in_place_f64(target, scale, input) };
        }

        fn iq_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx512_iq_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_to(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
            unsafe { avx512_iq_conjugate_multiply_to_f64(left, right, output) };
        }

        fn iq_conjugate_multiply_sum(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
            unsafe { avx512_iq_conjugate_multiply_sum_f64(left, right) }
        }

        fn iq_sum_squared_magnitudes(input: &[Complex<f64>]) -> f64 {
            unsafe { avx512_iq_sum_squared_magnitudes_f64(input) }
        }

        fn iq_calculate_magnitudes_to(input: &[Complex<f64>], output: &mut [f64]) {
            unsafe { avx512_iq_calculate_magnitudes_to_f64(input, output) };
        }

        fn iq_convolve_range_to(input: &[Complex<f64>], kernel: &[Complex<f64>], full_output_start: usize, output: &mut [Complex<f64>]) {
            unsafe { avx512_iq_convolve_range_to_f64(input, kernel, full_output_start, output) };
        }
    }

    /// Writes SSE2 f32 additions into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { add_f32_vectors(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output)) };
        scalar::add_f32_tail(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output), end);
    }

    /// Adds SSE2 f32 samples into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
        let end = unsafe { add_assign_f32_vectors(as_f32_slice_mut(target), as_f32_slice(input)) };
        scalar::add_assign_f32_tail(as_f32_slice_mut(target), as_f32_slice(input), end);
    }

    /// Adds SSE2 complex add-scaled results into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
        let end = unsafe { sse2_add_scaled_vectors(target, scale, input) };
        scalar::add_scaled_tail(target, scale, input, end);
    }

    /// Writes SSE2 complex products into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { sse2_multiply_vectors(left, right, output) };
        scalar::multiply_tail(left, right, output, end);
    }

    /// Writes SSE2 conjugate-multiply products into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { sse2_conjugate_multiply_vectors(left, right, output) };
        scalar::conjugate_multiply_tail(left, right, output, end);
    }

    /// Returns an SSE2 sum of conjugate-multiply products.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { sse2_complex_conjugate_multiply_sum_vectors(left, right) };
        sum + scalar::conjugate_multiply_sum_tail(left, right, end)
    }

    /// Returns an SSE2 sum of squared magnitudes.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
        let (end, sum) = unsafe { sum_squared_f32_vectors(as_f32_slice(input)) };
        sum + scalar::sum_squared_f32_tail(as_f32_slice(input), end)
    }

    /// Writes SSE2 magnitudes into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
        let end = unsafe { sse2_magnitude_vectors(input, output) };
        scalar::calculate_magnitudes_tail(input, output, end);
    }

    /// Writes SSE2 direct convolution outputs into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            sse2_complex_multiply_sum(input_overlap, kernel_overlap)
        });
    }

    /// Writes AVX2 f32 additions into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx2_add_f32_vectors(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output)) };
        scalar::add_f32_tail(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output), end);
    }

    /// Adds AVX2 f32 samples into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
        let end = unsafe { avx2_add_assign_f32_vectors(as_f32_slice_mut(target), as_f32_slice(input)) };
        scalar::add_assign_f32_tail(as_f32_slice_mut(target), as_f32_slice(input), end);
    }

    /// Adds AVX2 complex add-scaled results into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
        let end = unsafe { avx2_add_scaled_vectors(target, scale, input) };
        scalar::add_scaled_tail(target, scale, input, end);
    }

    /// Writes AVX2 complex products into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx2_multiply_vectors(left, right, output) };
        scalar::multiply_tail(left, right, output, end);
    }

    /// Writes AVX2 conjugate-multiply products into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx2_conjugate_multiply_vectors(left, right, output) };
        scalar::conjugate_multiply_tail(left, right, output, end);
    }

    /// Returns an AVX2 sum of conjugate-multiply products.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { avx2_complex_conjugate_multiply_sum_vectors(left, right) };
        sum + scalar::conjugate_multiply_sum_tail(left, right, end)
    }

    /// Returns an AVX2 sum of squared magnitudes.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
        let (end, sum) = unsafe { avx2_sum_squared_f32_vectors(as_f32_slice(input)) };
        sum + scalar::sum_squared_f32_tail(as_f32_slice(input), end)
    }

    /// Writes AVX2 magnitudes into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
        let end = unsafe { avx2_magnitude_vectors(input, output) };
        scalar::calculate_magnitudes_tail(input, output, end);
    }

    /// Writes AVX2 direct convolution outputs into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_convolve_range_to(input: &[Complex<f32>], kernel: &[Complex<f32>], full_output_start: usize, output: &mut [Complex<f32>]) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            avx2_complex_multiply_sum(input_overlap, kernel_overlap)
        });
    }

    /// Writes AVX-512 f32 additions into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx512_add_f32_vectors(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output)) };
        scalar::add_f32_tail(as_f32_slice(left), as_f32_slice(right), as_f32_slice_mut(output), end);
    }

    /// Adds AVX-512 f32 samples into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_in_place(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
        let end = unsafe { avx512_add_assign_f32_vectors(as_f32_slice_mut(target), as_f32_slice(input)) };
        scalar::add_assign_f32_tail(as_f32_slice_mut(target), as_f32_slice(input), end);
    }

    /// Adds AVX-512 complex add-scaled results into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_scaled_in_place(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
        let end = unsafe { avx512_add_scaled_vectors(target, scale, input) };
        scalar::add_scaled_tail(target, scale, input, end);
    }

    /// Writes AVX-512 complex products into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx512_multiply_vectors(left, right, output) };
        scalar::multiply_tail(left, right, output, end);
    }

    /// Writes AVX-512 conjugate-multiply products into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_conjugate_multiply_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
        let end = unsafe { avx512_conjugate_multiply_vectors(left, right, output) };
        scalar::conjugate_multiply_tail(left, right, output, end);
    }

    /// Returns an AVX-512 sum of conjugate-multiply products.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_conjugate_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { avx512_complex_conjugate_multiply_sum_vectors(left, right) };
        sum + scalar::conjugate_multiply_sum_tail(left, right, end)
    }

    /// Returns an AVX-512 sum of squared magnitudes.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_sum_squared_magnitudes(input: &[Complex<f32>]) -> f32 {
        let (end, sum) = unsafe { avx512_sum_squared_f32_vectors(as_f32_slice(input)) };
        sum + scalar::sum_squared_f32_tail(as_f32_slice(input), end)
    }

    /// Writes AVX-512 magnitudes into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_calculate_magnitudes_to(input: &[Complex<f32>], output: &mut [f32]) {
        let end = unsafe { avx512_magnitude_vectors(input, output) };
        scalar::calculate_magnitudes_tail(input, output, end);
    }

    /// Writes AVX-512 direct convolution outputs into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_convolve_range_to(
        input: &[Complex<f32>],
        kernel: &[Complex<f32>],
        full_output_start: usize,
        output: &mut [Complex<f32>],
    ) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            avx512_complex_multiply_sum(input_overlap, kernel_overlap)
        });
    }

    /// Writes SSE2 f64 additions into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { add_f64_vectors(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output)) };
        scalar::add_f64_tail(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output), end);
    }

    /// Adds SSE2 f64 samples into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_in_place_f64(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
        let end = unsafe { add_assign_f64_vectors(as_f64_slice_mut(target), as_f64_slice(input)) };
        scalar::add_assign_f64_tail(as_f64_slice_mut(target), as_f64_slice(input), end);
    }

    /// Adds SSE2 f64 complex add-scaled results into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_add_scaled_in_place_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
        let end = unsafe { sse2_add_scaled_vectors_f64(target, scale, input) };
        scalar::add_scaled_tail_f64(target, scale, input, end);
    }

    /// Writes SSE2 f64 complex products into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { sse2_multiply_vectors_f64(left, right, output) };
        scalar::multiply_tail_f64(left, right, output, end);
    }

    /// Writes SSE2 f64 conjugate-multiply products into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_conjugate_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { sse2_conjugate_multiply_vectors_f64(left, right, output) };
        scalar::conjugate_multiply_tail_f64(left, right, output, end);
    }

    /// Returns an SSE2 f64 sum of conjugate-multiply products.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_conjugate_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { sse2_complex_conjugate_multiply_sum_vectors_f64(left, right) };
        sum + scalar::conjugate_multiply_sum_tail_f64(left, right, end)
    }

    /// Returns an SSE2 f64 sum of squared magnitudes.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_sum_squared_magnitudes_f64(input: &[Complex<f64>]) -> f64 {
        let (end, sum) = unsafe { sum_squared_f64_vectors(as_f64_slice(input)) };
        sum + scalar::sum_squared_f64_tail(as_f64_slice(input), end)
    }

    /// Writes SSE2 f64 magnitudes into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_calculate_magnitudes_to_f64(input: &[Complex<f64>], output: &mut [f64]) {
        let end = unsafe { sse2_magnitude_vectors_f64(input, output) };
        scalar::calculate_magnitudes_tail_f64(input, output, end);
    }

    /// Writes SSE2 f64 direct convolution outputs into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_iq_convolve_range_to_f64(
        input: &[Complex<f64>],
        kernel: &[Complex<f64>],
        full_output_start: usize,
        output: &mut [Complex<f64>],
    ) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            sse2_complex_multiply_sum_f64(input_overlap, kernel_overlap)
        });
    }

    /// Writes AVX2 f64 additions into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx2_add_f64_vectors(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output)) };
        scalar::add_f64_tail(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output), end);
    }

    /// Adds AVX2 f64 samples into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_in_place_f64(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
        let end = unsafe { avx2_add_assign_f64_vectors(as_f64_slice_mut(target), as_f64_slice(input)) };
        scalar::add_assign_f64_tail(as_f64_slice_mut(target), as_f64_slice(input), end);
    }

    /// Adds AVX2 f64 complex add-scaled results into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_add_scaled_in_place_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
        let end = unsafe { avx2_add_scaled_vectors_f64(target, scale, input) };
        scalar::add_scaled_tail_f64(target, scale, input, end);
    }

    /// Writes AVX2 f64 complex products into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx2_multiply_vectors_f64(left, right, output) };
        scalar::multiply_tail_f64(left, right, output, end);
    }

    /// Writes AVX2 f64 conjugate-multiply products into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_conjugate_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx2_conjugate_multiply_vectors_f64(left, right, output) };
        scalar::conjugate_multiply_tail_f64(left, right, output, end);
    }

    /// Returns an AVX2 f64 sum of conjugate-multiply products.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_conjugate_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { avx2_complex_conjugate_multiply_sum_vectors_f64(left, right) };
        sum + scalar::conjugate_multiply_sum_tail_f64(left, right, end)
    }

    /// Returns an AVX2 f64 sum of squared magnitudes.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_sum_squared_magnitudes_f64(input: &[Complex<f64>]) -> f64 {
        let (end, sum) = unsafe { avx2_sum_squared_f64_vectors(as_f64_slice(input)) };
        sum + scalar::sum_squared_f64_tail(as_f64_slice(input), end)
    }

    /// Writes AVX2 f64 magnitudes into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_calculate_magnitudes_to_f64(input: &[Complex<f64>], output: &mut [f64]) {
        let end = unsafe { avx2_magnitude_vectors_f64(input, output) };
        scalar::calculate_magnitudes_tail_f64(input, output, end);
    }

    /// Writes AVX2 f64 direct convolution outputs into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_iq_convolve_range_to_f64(
        input: &[Complex<f64>],
        kernel: &[Complex<f64>],
        full_output_start: usize,
        output: &mut [Complex<f64>],
    ) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            avx2_complex_multiply_sum_f64(input_overlap, kernel_overlap)
        });
    }

    /// Writes AVX-512 f64 additions into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx512_add_f64_vectors(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output)) };
        scalar::add_f64_tail(as_f64_slice(left), as_f64_slice(right), as_f64_slice_mut(output), end);
    }

    /// Adds AVX-512 f64 samples into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_in_place_f64(target: &mut [Complex<f64>], input: &[Complex<f64>]) {
        let end = unsafe { avx512_add_assign_f64_vectors(as_f64_slice_mut(target), as_f64_slice(input)) };
        scalar::add_assign_f64_tail(as_f64_slice_mut(target), as_f64_slice(input), end);
    }

    /// Adds AVX-512 f64 complex add-scaled results into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_add_scaled_in_place_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) {
        let end = unsafe { avx512_add_scaled_vectors_f64(target, scale, input) };
        scalar::add_scaled_tail_f64(target, scale, input, end);
    }

    /// Writes AVX-512 f64 complex products into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx512_multiply_vectors_f64(left, right, output) };
        scalar::multiply_tail_f64(left, right, output, end);
    }

    /// Writes AVX-512 f64 conjugate-multiply products into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_conjugate_multiply_to_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) {
        let end = unsafe { avx512_conjugate_multiply_vectors_f64(left, right, output) };
        scalar::conjugate_multiply_tail_f64(left, right, output, end);
    }

    /// Returns an AVX-512 f64 sum of conjugate-multiply products.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_conjugate_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { avx512_complex_conjugate_multiply_sum_vectors_f64(left, right) };
        sum + scalar::conjugate_multiply_sum_tail_f64(left, right, end)
    }

    /// Returns an AVX-512 f64 sum of squared magnitudes.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_sum_squared_magnitudes_f64(input: &[Complex<f64>]) -> f64 {
        let (end, sum) = unsafe { avx512_sum_squared_f64_vectors(as_f64_slice(input)) };
        sum + scalar::sum_squared_f64_tail(as_f64_slice(input), end)
    }

    /// Writes AVX-512 f64 magnitudes into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_calculate_magnitudes_to_f64(input: &[Complex<f64>], output: &mut [f64]) {
        let end = unsafe { avx512_magnitude_vectors_f64(input, output) };
        scalar::calculate_magnitudes_tail_f64(input, output, end);
    }

    /// Writes AVX-512 f64 direct convolution outputs into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_iq_convolve_range_to_f64(
        input: &[Complex<f64>],
        kernel: &[Complex<f64>],
        full_output_start: usize,
        output: &mut [Complex<f64>],
    ) {
        write_convolution_range_to(input, kernel, full_output_start, output, |input_overlap, kernel_overlap| unsafe {
            avx512_complex_multiply_sum_f64(input_overlap, kernel_overlap)
        });
    }

    /// Writes SSE2 f32 additions and returns the processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn add_f32_vectors(left: &[f32], right: &[f32], output: &mut [f32]) -> usize {
        let end = vectorized_end(left.len(), SSE_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm_add_ps(load_sse(left, index), load_sse(right, index)) };
            unsafe { _mm_storeu_ps(output.as_mut_ptr().add(index), sum) };
            index += SSE_F32_LANES;
        }
        end
    }

    /// Adds SSE2 f32 samples and returns the processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn add_assign_f32_vectors(target: &mut [f32], input: &[f32]) -> usize {
        let end = vectorized_end(target.len(), SSE_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm_add_ps(load_sse(target, index), load_sse(input, index)) };
            unsafe { _mm_storeu_ps(target.as_mut_ptr().add(index), sum) };
            index += SSE_F32_LANES;
        }
        end
    }

    /// Adds SSE2 complex add-scaled values and returns processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_add_scaled_vectors(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) -> usize {
        let end = vectorized_end(input.len(), SSE_COMPLEX_LANES);
        let scale_re = _mm_set1_ps(scale.re);
        let scale_im = unsafe { _mm_mul_ps(_mm_set1_ps(scale.im), load_sse_sign()) };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_add_scaled(target, input, index, scale_re, scale_im) };
            index += SSE_COMPLEX_LANES;
        }
        end
    }

    /// Writes SSE2 complex products and returns processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES);
        let sign = unsafe { load_sse_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_multiply(left, right, output, index, sign) };
            index += SSE_COMPLEX_LANES;
        }
        end
    }

    /// Writes SSE2 conjugate-multiply products and returns processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_conjugate_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES);
        let sign = unsafe { load_sse_sign() };
        let conjugate_sign = unsafe { load_sse_conjugate_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_conjugate_multiply(left, right, output, index, sign, conjugate_sign) };
            index += SSE_COMPLEX_LANES;
        }
        end
    }

    /// Returns the SSE2 f32 square sum and processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn sum_squared_f32_vectors(input: &[f32]) -> (usize, f32) {
        let end = vectorized_end(input.len(), SSE_F32_LANES);
        let mut accum = _mm_setzero_ps();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_sse(input, index) };
            accum = _mm_add_ps(accum, _mm_mul_ps(values, values));
            index += SSE_F32_LANES;
        }
        (end, unsafe { horizontal_sum_sse(accum) })
    }

    /// Writes SSE2 magnitudes and returns processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_magnitude_vectors(input: &[Complex<f32>], output: &mut [f32]) -> usize {
        let end = vectorized_end(input.len(), SSE_COMPLEX_LANES);
        let input_f32 = as_f32_slice(input);
        let mut index = 0;
        while index < end {
            let magnitudes = unsafe { sse2_magnitudes(input_f32, index) };
            output[index..index + SSE_COMPLEX_LANES].copy_from_slice(&magnitudes);
            index += SSE_COMPLEX_LANES;
        }
        end
    }

    /// Returns the SSE2 sum of pairwise complex products.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { sse2_complex_multiply_sum_vectors(left, right) };
        sum + scalar::multiply_sum_tail(left, right, end)
    }

    /// Returns the SSE2 vector sum and processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES);
        let sign = unsafe { load_sse_sign() };
        let mut accum = _mm_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_sse(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_sse(as_f32_slice(right), index * 2) };
            accum = _mm_add_ps(accum, unsafe { complex_multiply_sse(left_vector, right_vector, sign) });
            index += SSE_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_sse(accum) })
    }

    /// Returns the SSE2 conjugate-multiply vector sum and processed samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_conjugate_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES);
        let sign = unsafe { load_sse_sign() };
        let conjugate_sign = unsafe { load_sse_conjugate_sign() };
        let mut accum = _mm_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_sse(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_sse(as_f32_slice(right), index * 2) };
            accum = _mm_add_ps(accum, unsafe {
                complex_conjugate_multiply_sse(left_vector, right_vector, sign, conjugate_sign)
            });
            index += SSE_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_sse(accum) })
    }

    /// Writes AVX2 f32 additions and returns the processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_f32_vectors(left: &[f32], right: &[f32], output: &mut [f32]) -> usize {
        let end = vectorized_end(left.len(), AVX2_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm256_add_ps(load_avx2(left, index), load_avx2(right, index)) };
            unsafe { _mm256_storeu_ps(output.as_mut_ptr().add(index), sum) };
            index += AVX2_F32_LANES;
        }
        end
    }

    /// Adds AVX2 f32 samples and returns the processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_assign_f32_vectors(target: &mut [f32], input: &[f32]) -> usize {
        let end = vectorized_end(target.len(), AVX2_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm256_add_ps(load_avx2(target, index), load_avx2(input, index)) };
            unsafe { _mm256_storeu_ps(target.as_mut_ptr().add(index), sum) };
            index += AVX2_F32_LANES;
        }
        end
    }

    /// Adds AVX2 complex add-scaled values and returns processed complex samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_add_scaled_vectors(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) -> usize {
        let end = vectorized_end(input.len(), AVX2_COMPLEX_LANES);
        let scale_re = _mm256_set1_ps(scale.re);
        let scale_im = unsafe { _mm256_mul_ps(_mm256_set1_ps(scale.im), load_avx2_sign()) };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_add_scaled(target, input, index, scale_re, scale_im) };
            index += AVX2_COMPLEX_LANES;
        }
        end
    }

    /// Writes AVX2 complex products and returns processed complex samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES);
        let sign = unsafe { load_avx2_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_multiply(left, right, output, index, sign) };
            index += AVX2_COMPLEX_LANES;
        }
        end
    }

    /// Writes AVX2 conjugate-multiply products and returns processed complex samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_conjugate_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES);
        let sign = unsafe { load_avx2_sign() };
        let conjugate_sign = unsafe { load_avx2_conjugate_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_conjugate_multiply(left, right, output, index, sign, conjugate_sign) };
            index += AVX2_COMPLEX_LANES;
        }
        end
    }

    /// Returns the AVX2 f32 square sum and processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sum_squared_f32_vectors(input: &[f32]) -> (usize, f32) {
        let end = vectorized_end(input.len(), AVX2_F32_LANES);
        let mut accum = _mm256_setzero_ps();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_avx2(input, index) };
            accum = _mm256_add_ps(accum, _mm256_mul_ps(values, values));
            index += AVX2_F32_LANES;
        }
        (end, unsafe { horizontal_sum_avx2(accum) })
    }

    /// Writes AVX2 magnitudes and returns processed complex samples.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_magnitude_vectors(input: &[Complex<f32>], output: &mut [f32]) -> usize {
        let end = vectorized_end(input.len(), AVX2_COMPLEX_LANES);
        let input_f32 = as_f32_slice(input);
        let mut index = 0;
        while index < end {
            let magnitudes = unsafe { avx2_magnitudes(input_f32, index) };
            output[index..index + AVX2_COMPLEX_LANES].copy_from_slice(&magnitudes);
            index += AVX2_COMPLEX_LANES;
        }
        end
    }

    /// Returns the AVX2 sum of pairwise complex products.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { avx2_complex_multiply_sum_vectors(left, right) };
        sum + scalar::multiply_sum_tail(left, right, end)
    }

    /// Returns the AVX2 vector sum and processed complex samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES);
        let sign = unsafe { load_avx2_sign() };
        let mut accum = _mm256_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx2(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_avx2(as_f32_slice(right), index * 2) };
            accum = _mm256_add_ps(accum, complex_multiply_avx2(left_vector, right_vector, sign));
            index += AVX2_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_avx2(accum) })
    }

    /// Returns the AVX2 conjugate-multiply vector sum and processed samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_conjugate_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES);
        let sign = unsafe { load_avx2_sign() };
        let conjugate_sign = unsafe { load_avx2_conjugate_sign() };
        let mut accum = _mm256_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx2(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_avx2(as_f32_slice(right), index * 2) };
            accum = _mm256_add_ps(accum, complex_conjugate_multiply_avx2(left_vector, right_vector, sign, conjugate_sign));
            index += AVX2_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_avx2(accum) })
    }

    /// Writes AVX-512 f32 additions and returns the processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_f32_vectors(left: &[f32], right: &[f32], output: &mut [f32]) -> usize {
        let end = vectorized_end(left.len(), AVX512_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm512_add_ps(load_avx512(left, index), load_avx512(right, index)) };
            unsafe { _mm512_storeu_ps(output.as_mut_ptr().add(index), sum) };
            index += AVX512_F32_LANES;
        }
        end
    }

    /// Adds AVX-512 f32 samples and returns the processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_assign_f32_vectors(target: &mut [f32], input: &[f32]) -> usize {
        let end = vectorized_end(target.len(), AVX512_F32_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm512_add_ps(load_avx512(target, index), load_avx512(input, index)) };
            unsafe { _mm512_storeu_ps(target.as_mut_ptr().add(index), sum) };
            index += AVX512_F32_LANES;
        }
        end
    }

    /// Adds AVX-512 complex add-scaled values and returns processed complex samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_scaled_vectors(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) -> usize {
        let end = vectorized_end(input.len(), AVX512_COMPLEX_LANES);
        let scale_re = _mm512_set1_ps(scale.re);
        let scale_im = unsafe { _mm512_mul_ps(_mm512_set1_ps(scale.im), load_avx512_sign()) };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_add_scaled(target, input, index, scale_re, scale_im) };
            index += AVX512_COMPLEX_LANES;
        }
        end
    }

    /// Writes AVX-512 complex products and returns processed complex samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES);
        let sign = unsafe { load_avx512_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_multiply(left, right, output, index, sign) };
            index += AVX512_COMPLEX_LANES;
        }
        end
    }

    /// Writes AVX-512 conjugate-multiply products and returns processed samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_conjugate_multiply_vectors(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) -> usize {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES);
        let sign = unsafe { load_avx512_sign() };
        let conjugate_sign = unsafe { load_avx512_conjugate_sign() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_conjugate_multiply(left, right, output, index, sign, conjugate_sign) };
            index += AVX512_COMPLEX_LANES;
        }
        end
    }

    /// Returns the AVX-512 f32 square sum and processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_sum_squared_f32_vectors(input: &[f32]) -> (usize, f32) {
        let end = vectorized_end(input.len(), AVX512_F32_LANES);
        let mut accum = _mm512_setzero_ps();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_avx512(input, index) };
            accum = _mm512_add_ps(accum, _mm512_mul_ps(values, values));
            index += AVX512_F32_LANES;
        }
        (end, unsafe { horizontal_sum_avx512(accum) })
    }

    /// Writes AVX-512 magnitudes and returns processed complex samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_magnitude_vectors(input: &[Complex<f32>], output: &mut [f32]) -> usize {
        let end = vectorized_end(input.len(), AVX512_COMPLEX_LANES);
        let input_f32 = as_f32_slice(input);
        let mut index = 0;
        while index < end {
            let magnitudes = unsafe { avx512_magnitudes(input_f32, index) };
            output[index..index + AVX512_COMPLEX_LANES].copy_from_slice(&magnitudes);
            index += AVX512_COMPLEX_LANES;
        }
        end
    }

    /// Returns the AVX-512 sum of pairwise complex products.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_multiply_sum(left: &[Complex<f32>], right: &[Complex<f32>]) -> Complex<f32> {
        let (end, sum) = unsafe { avx512_complex_multiply_sum_vectors(left, right) };
        sum + scalar::multiply_sum_tail(left, right, end)
    }

    /// Returns the AVX-512 vector sum and processed complex samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES);
        let sign = unsafe { load_avx512_sign() };
        let mut accum = _mm512_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx512(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_avx512(as_f32_slice(right), index * 2) };
            accum = _mm512_add_ps(accum, complex_multiply_avx512(left_vector, right_vector, sign));
            index += AVX512_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_avx512(accum) })
    }

    /// Returns the AVX-512 conjugate-multiply vector sum and processed samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_conjugate_multiply_sum_vectors(left: &[Complex<f32>], right: &[Complex<f32>]) -> (usize, Complex<f32>) {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES);
        let sign = unsafe { load_avx512_sign() };
        let conjugate_sign = unsafe { load_avx512_conjugate_sign() };
        let mut accum = _mm512_setzero_ps();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx512(as_f32_slice(left), index * 2) };
            let right_vector = unsafe { load_avx512(as_f32_slice(right), index * 2) };
            accum = _mm512_add_ps(accum, complex_conjugate_multiply_avx512(left_vector, right_vector, sign, conjugate_sign));
            index += AVX512_COMPLEX_LANES;
        }
        (end, unsafe { horizontal_complex_sum_avx512(accum) })
    }

    /// Loads four f32 values from an unaligned slice offset.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse(input: &[f32], index: usize) -> __m128 {
        unsafe { _mm_loadu_ps(input.as_ptr().add(index)) }
    }

    /// Loads the SSE complex sign pattern.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse_sign() -> __m128 {
        unsafe { _mm_loadu_ps(SSE_SIGN.as_ptr()) }
    }

    /// Loads the SSE conjugate sign pattern.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse_conjugate_sign() -> __m128 {
        unsafe { _mm_loadu_ps(SSE_CONJUGATE_SIGN.as_ptr()) }
    }

    /// Stores one SSE add-scaled vector into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_add_scaled(target: &mut [Complex<f32>], input: &[Complex<f32>], index: usize, scale_re: __m128, scale_im: __m128) {
        let input_vector = unsafe { load_sse(as_f32_slice(input), index * 2) };
        let product = unsafe { complex_scale_sse(input_vector, scale_re, scale_im) };
        let target_vector = unsafe { load_sse(as_f32_slice(target), index * 2) };
        let output = _mm_add_ps(target_vector, product);
        unsafe { _mm_storeu_ps(as_f32_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one SSE complex multiplication vector into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_multiply(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>], index: usize, sign: __m128) {
        let left_vector = unsafe { load_sse(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_sse(as_f32_slice(right), index * 2) };
        let product = unsafe { complex_multiply_sse(left_vector, right_vector, sign) };
        unsafe { _mm_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one SSE conjugate-multiply vector into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_conjugate_multiply(
        left: &[Complex<f32>],
        right: &[Complex<f32>],
        output: &mut [Complex<f32>],
        index: usize,
        sign: __m128,
        conjugate_sign: __m128,
    ) {
        let left_vector = unsafe { load_sse(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_sse(as_f32_slice(right), index * 2) };
        let product = unsafe { complex_conjugate_multiply_sse(left_vector, right_vector, sign, conjugate_sign) };
        unsafe { _mm_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed SSE complex values by a broadcast complex scalar.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_scale_sse(input: __m128, scale_re: __m128, scale_im: __m128) -> __m128 {
        let swapped = _mm_shuffle_ps(input, input, 0xb1);
        let real_product = _mm_mul_ps(input, scale_re);
        let imag_product = _mm_mul_ps(swapped, scale_im);
        _mm_add_ps(real_product, imag_product)
    }

    /// Multiplies packed SSE complex values pairwise.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_multiply_sse(left: __m128, right: __m128, sign: __m128) -> __m128 {
        let right_re = _mm_shuffle_ps(right, right, 0xa0);
        let right_im = _mm_mul_ps(_mm_shuffle_ps(right, right, 0xf5), sign);
        unsafe { complex_scale_sse(left, right_re, right_im) }
    }

    /// Multiplies packed conjugated SSE complex values pairwise.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_conjugate_multiply_sse(left: __m128, right: __m128, sign: __m128, conjugate_sign: __m128) -> __m128 {
        let conjugated_left = _mm_mul_ps(left, conjugate_sign);
        unsafe { complex_multiply_sse(conjugated_left, right, sign) }
    }

    /// Horizontally sums an SSE f32 vector.
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_sum_sse(input: __m128) -> f32 {
        let mut values = [0.0; SSE_F32_LANES];
        unsafe { _mm_storeu_ps(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed SSE complex lanes.
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_complex_sum_sse(input: __m128) -> Complex<f32> {
        let mut values = [0.0; SSE_F32_LANES];
        unsafe { _mm_storeu_ps(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates two SSE complex magnitudes.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_magnitudes(input_f32: &[f32], index: usize) -> [f32; SSE_COMPLEX_LANES] {
        let values = unsafe { load_sse(input_f32, index * 2) };
        let squares = _mm_mul_ps(values, values);
        let paired = _mm_add_ps(squares, _mm_shuffle_ps(squares, squares, 0xb1));
        let roots = _mm_sqrt_ps(paired);
        let mut lanes = [0.0; SSE_F32_LANES];
        unsafe { _mm_storeu_ps(lanes.as_mut_ptr(), roots) };
        [lanes[0], lanes[2]]
    }

    /// Loads eight f32 values from an unaligned slice offset.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2(input: &[f32], index: usize) -> __m256 {
        unsafe { _mm256_loadu_ps(input.as_ptr().add(index)) }
    }

    /// Loads the AVX2 complex sign pattern.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2_sign() -> __m256 {
        unsafe { _mm256_loadu_ps(AVX2_SIGN.as_ptr()) }
    }

    /// Loads the AVX2 conjugate sign pattern.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2_conjugate_sign() -> __m256 {
        unsafe { _mm256_loadu_ps(AVX2_CONJUGATE_SIGN.as_ptr()) }
    }

    /// Stores one AVX2 add-scaled vector into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_add_scaled(target: &mut [Complex<f32>], input: &[Complex<f32>], index: usize, scale_re: __m256, scale_im: __m256) {
        let input_vector = unsafe { load_avx2(as_f32_slice(input), index * 2) };
        let product = complex_scale_avx2(input_vector, scale_re, scale_im);
        let target_vector = unsafe { load_avx2(as_f32_slice(target), index * 2) };
        let output = _mm256_add_ps(target_vector, product);
        unsafe { _mm256_storeu_ps(as_f32_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one AVX2 complex multiplication vector into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_multiply(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>], index: usize, sign: __m256) {
        let left_vector = unsafe { load_avx2(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_avx2(as_f32_slice(right), index * 2) };
        let product = complex_multiply_avx2(left_vector, right_vector, sign);
        unsafe { _mm256_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one AVX2 conjugate-multiply vector into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_conjugate_multiply(
        left: &[Complex<f32>],
        right: &[Complex<f32>],
        output: &mut [Complex<f32>],
        index: usize,
        sign: __m256,
        conjugate_sign: __m256,
    ) {
        let left_vector = unsafe { load_avx2(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_avx2(as_f32_slice(right), index * 2) };
        let product = complex_conjugate_multiply_avx2(left_vector, right_vector, sign, conjugate_sign);
        unsafe { _mm256_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed AVX2 complex values by a broadcast complex scalar.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_scale_avx2(input: __m256, scale_re: __m256, scale_im: __m256) -> __m256 {
        let swapped = _mm256_permute_ps(input, 0xb1);
        let real_product = _mm256_mul_ps(input, scale_re);
        _mm256_fmadd_ps(swapped, scale_im, real_product)
    }

    /// Multiplies packed AVX2 complex values pairwise.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_multiply_avx2(left: __m256, right: __m256, sign: __m256) -> __m256 {
        let right_re = _mm256_permute_ps(right, 0xa0);
        let right_im = _mm256_mul_ps(_mm256_permute_ps(right, 0xf5), sign);
        complex_scale_avx2(left, right_re, right_im)
    }

    /// Multiplies packed conjugated AVX2 complex values pairwise.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_conjugate_multiply_avx2(left: __m256, right: __m256, sign: __m256, conjugate_sign: __m256) -> __m256 {
        let conjugated_left = _mm256_mul_ps(left, conjugate_sign);
        complex_multiply_avx2(conjugated_left, right, sign)
    }

    /// Horizontally sums an AVX2 f32 vector.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2(input: __m256) -> f32 {
        let mut values = [0.0; AVX2_F32_LANES];
        unsafe { _mm256_storeu_ps(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed AVX2 complex lanes.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_complex_sum_avx2(input: __m256) -> Complex<f32> {
        let mut values = [0.0; AVX2_F32_LANES];
        unsafe { _mm256_storeu_ps(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates four AVX2 complex magnitudes.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_magnitudes(input_f32: &[f32], index: usize) -> [f32; AVX2_COMPLEX_LANES] {
        let values = unsafe { load_avx2(input_f32, index * 2) };
        let squares = _mm256_mul_ps(values, values);
        let paired = _mm256_add_ps(squares, _mm256_permute_ps(squares, 0xb1));
        let roots = _mm256_sqrt_ps(paired);
        let mut lanes = [0.0; AVX2_F32_LANES];
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), roots) };
        [lanes[0], lanes[2], lanes[4], lanes[6]]
    }

    /// Loads sixteen f32 values from an unaligned slice offset.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512(input: &[f32], index: usize) -> __m512 {
        unsafe { _mm512_loadu_ps(input.as_ptr().add(index)) }
    }

    /// Loads the AVX-512 complex sign pattern.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512_sign() -> __m512 {
        unsafe { _mm512_loadu_ps(AVX512_SIGN.as_ptr()) }
    }

    /// Loads the AVX-512 conjugate sign pattern.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512_conjugate_sign() -> __m512 {
        unsafe { _mm512_loadu_ps(AVX512_CONJUGATE_SIGN.as_ptr()) }
    }

    /// Stores one AVX-512 add-scaled vector into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_add_scaled(target: &mut [Complex<f32>], input: &[Complex<f32>], index: usize, scale_re: __m512, scale_im: __m512) {
        let input_vector = unsafe { load_avx512(as_f32_slice(input), index * 2) };
        let product = complex_scale_avx512(input_vector, scale_re, scale_im);
        let target_vector = unsafe { load_avx512(as_f32_slice(target), index * 2) };
        let output = _mm512_add_ps(target_vector, product);
        unsafe { _mm512_storeu_ps(as_f32_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one AVX-512 complex multiplication vector into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_multiply(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>], index: usize, sign: __m512) {
        let left_vector = unsafe { load_avx512(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_avx512(as_f32_slice(right), index * 2) };
        let product = complex_multiply_avx512(left_vector, right_vector, sign);
        unsafe { _mm512_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one AVX-512 conjugate-multiply vector into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_conjugate_multiply(
        left: &[Complex<f32>],
        right: &[Complex<f32>],
        output: &mut [Complex<f32>],
        index: usize,
        sign: __m512,
        conjugate_sign: __m512,
    ) {
        let left_vector = unsafe { load_avx512(as_f32_slice(left), index * 2) };
        let right_vector = unsafe { load_avx512(as_f32_slice(right), index * 2) };
        let product = complex_conjugate_multiply_avx512(left_vector, right_vector, sign, conjugate_sign);
        unsafe { _mm512_storeu_ps(as_f32_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed AVX-512 complex values by a broadcast complex scalar.
    #[target_feature(enable = "avx512f")]
    fn complex_scale_avx512(input: __m512, scale_re: __m512, scale_im: __m512) -> __m512 {
        let swapped = _mm512_permute_ps(input, 0xb1);
        let real_product = _mm512_mul_ps(input, scale_re);
        _mm512_fmadd_ps(swapped, scale_im, real_product)
    }

    /// Multiplies packed AVX-512 complex values pairwise.
    #[target_feature(enable = "avx512f")]
    fn complex_multiply_avx512(left: __m512, right: __m512, sign: __m512) -> __m512 {
        let right_re = _mm512_permute_ps(right, 0xa0);
        let right_im = _mm512_mul_ps(_mm512_permute_ps(right, 0xf5), sign);
        complex_scale_avx512(left, right_re, right_im)
    }

    /// Multiplies packed conjugated AVX-512 complex values pairwise.
    #[target_feature(enable = "avx512f")]
    fn complex_conjugate_multiply_avx512(left: __m512, right: __m512, sign: __m512, conjugate_sign: __m512) -> __m512 {
        let conjugated_left = _mm512_mul_ps(left, conjugate_sign);
        complex_multiply_avx512(conjugated_left, right, sign)
    }

    /// Horizontally sums an AVX-512 f32 vector.
    #[target_feature(enable = "avx512f")]
    unsafe fn horizontal_sum_avx512(input: __m512) -> f32 {
        let mut values = [0.0; AVX512_F32_LANES];
        unsafe { _mm512_storeu_ps(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed AVX-512 complex lanes.
    #[target_feature(enable = "avx512f")]
    unsafe fn horizontal_complex_sum_avx512(input: __m512) -> Complex<f32> {
        let mut values = [0.0; AVX512_F32_LANES];
        unsafe { _mm512_storeu_ps(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates eight AVX-512 complex magnitudes.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_magnitudes(input_f32: &[f32], index: usize) -> [f32; AVX512_COMPLEX_LANES] {
        let values = unsafe { load_avx512(input_f32, index * 2) };
        let squares = _mm512_mul_ps(values, values);
        let paired = _mm512_add_ps(squares, _mm512_permute_ps(squares, 0xb1));
        let roots = _mm512_sqrt_ps(paired);
        let mut lanes = [0.0; AVX512_F32_LANES];
        unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), roots) };
        [lanes[0], lanes[2], lanes[4], lanes[6], lanes[8], lanes[10], lanes[12], lanes[14]]
    }

    /// Writes SSE2 f64 additions and returns the processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn add_f64_vectors(left: &[f64], right: &[f64], output: &mut [f64]) -> usize {
        let end = vectorized_end(left.len(), SSE_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm_add_pd(load_sse_f64(left, index), load_sse_f64(right, index)) };
            unsafe { _mm_storeu_pd(output.as_mut_ptr().add(index), sum) };
            index += SSE_F64_LANES;
        }
        end
    }

    /// Adds SSE2 f64 samples and returns the processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn add_assign_f64_vectors(target: &mut [f64], input: &[f64]) -> usize {
        let end = vectorized_end(target.len(), SSE_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm_add_pd(load_sse_f64(target, index), load_sse_f64(input, index)) };
            unsafe { _mm_storeu_pd(target.as_mut_ptr().add(index), sum) };
            index += SSE_F64_LANES;
        }
        end
    }

    /// Adds SSE2 f64 complex add-scaled values and returns processed samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_add_scaled_vectors_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) -> usize {
        let end = vectorized_end(input.len(), SSE_COMPLEX_LANES_F64);
        let scale_re = _mm_set1_pd(scale.re);
        let scale_im = unsafe { _mm_mul_pd(_mm_set1_pd(scale.im), load_sse_sign_f64()) };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_add_scaled_f64(target, input, index, scale_re, scale_im) };
            index += SSE_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes SSE2 f64 complex products and returns processed samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES_F64);
        let sign = unsafe { load_sse_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_multiply_f64(left, right, output, index, sign) };
            index += SSE_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes SSE2 f64 conjugate-multiply products and returns samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_conjugate_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES_F64);
        let sign = unsafe { load_sse_sign_f64() };
        let conjugate_sign = unsafe { load_sse_conjugate_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_sse_conjugate_multiply_f64(left, right, output, index, sign, conjugate_sign) };
            index += SSE_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the SSE2 f64 square sum and processed prefix length.
    #[target_feature(enable = "sse2")]
    unsafe fn sum_squared_f64_vectors(input: &[f64]) -> (usize, f64) {
        let end = vectorized_end(input.len(), SSE_F64_LANES);
        let mut accum = _mm_setzero_pd();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_sse_f64(input, index) };
            accum = _mm_add_pd(accum, _mm_mul_pd(values, values));
            index += SSE_F64_LANES;
        }
        (end, unsafe { horizontal_sum_sse_f64(accum) })
    }

    /// Writes SSE2 f64 magnitudes and returns processed samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_magnitude_vectors_f64(input: &[Complex<f64>], output: &mut [f64]) -> usize {
        let end = vectorized_end(input.len(), SSE_COMPLEX_LANES_F64);
        let input_f64 = as_f64_slice(input);
        let mut index = 0;
        while index < end {
            output[index] = unsafe { sse2_magnitude_f64(input_f64, index) };
            index += SSE_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the SSE2 f64 sum of pairwise complex products.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { sse2_complex_multiply_sum_vectors_f64(left, right) };
        sum + scalar::multiply_sum_tail_f64(left, right, end)
    }

    /// Returns the SSE2 f64 vector sum and processed complex samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES_F64);
        let sign = unsafe { load_sse_sign_f64() };
        let mut accum = _mm_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_sse_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_sse_f64(as_f64_slice(right), index * 2) };
            accum = _mm_add_pd(accum, unsafe { complex_multiply_sse_f64(left_vector, right_vector, sign) });
            index += SSE_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_sse_f64(accum) })
    }

    /// Returns the SSE2 f64 conjugate-multiply vector sum and samples.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_complex_conjugate_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), SSE_COMPLEX_LANES_F64);
        let sign = unsafe { load_sse_sign_f64() };
        let conjugate_sign = unsafe { load_sse_conjugate_sign_f64() };
        let mut accum = _mm_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_sse_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_sse_f64(as_f64_slice(right), index * 2) };
            accum = _mm_add_pd(accum, unsafe {
                complex_conjugate_multiply_sse_f64(left_vector, right_vector, sign, conjugate_sign)
            });
            index += SSE_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_sse_f64(accum) })
    }

    /// Writes AVX2 f64 additions and returns the processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_f64_vectors(left: &[f64], right: &[f64], output: &mut [f64]) -> usize {
        let end = vectorized_end(left.len(), AVX2_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm256_add_pd(load_avx2_f64(left, index), load_avx2_f64(right, index)) };
            unsafe { _mm256_storeu_pd(output.as_mut_ptr().add(index), sum) };
            index += AVX2_F64_LANES;
        }
        end
    }

    /// Adds AVX2 f64 samples and returns the processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_assign_f64_vectors(target: &mut [f64], input: &[f64]) -> usize {
        let end = vectorized_end(target.len(), AVX2_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm256_add_pd(load_avx2_f64(target, index), load_avx2_f64(input, index)) };
            unsafe { _mm256_storeu_pd(target.as_mut_ptr().add(index), sum) };
            index += AVX2_F64_LANES;
        }
        end
    }

    /// Adds AVX2 f64 complex add-scaled values and returns samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_add_scaled_vectors_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) -> usize {
        let end = vectorized_end(input.len(), AVX2_COMPLEX_LANES_F64);
        let scale_re = _mm256_set1_pd(scale.re);
        let scale_im = unsafe { _mm256_mul_pd(_mm256_set1_pd(scale.im), load_avx2_sign_f64()) };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_add_scaled_f64(target, input, index, scale_re, scale_im) };
            index += AVX2_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes AVX2 f64 complex products and returns processed samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx2_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_multiply_f64(left, right, output, index, sign) };
            index += AVX2_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes AVX2 f64 conjugate-multiply products and returns samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_conjugate_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx2_sign_f64() };
        let conjugate_sign = unsafe { load_avx2_conjugate_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx2_conjugate_multiply_f64(left, right, output, index, sign, conjugate_sign) };
            index += AVX2_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the AVX2 f64 square sum and processed prefix length.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_sum_squared_f64_vectors(input: &[f64]) -> (usize, f64) {
        let end = vectorized_end(input.len(), AVX2_F64_LANES);
        let mut accum = _mm256_setzero_pd();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_avx2_f64(input, index) };
            accum = _mm256_add_pd(accum, _mm256_mul_pd(values, values));
            index += AVX2_F64_LANES;
        }
        (end, unsafe { horizontal_sum_avx2_f64(accum) })
    }

    /// Writes AVX2 f64 magnitudes and returns processed samples.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_magnitude_vectors_f64(input: &[Complex<f64>], output: &mut [f64]) -> usize {
        let end = vectorized_end(input.len(), AVX2_COMPLEX_LANES_F64);
        let input_f64 = as_f64_slice(input);
        let mut index = 0;
        while index < end {
            let magnitudes = unsafe { avx2_magnitudes_f64(input_f64, index) };
            output[index..index + AVX2_COMPLEX_LANES_F64].copy_from_slice(&magnitudes);
            index += AVX2_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the AVX2 f64 sum of pairwise complex products.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { avx2_complex_multiply_sum_vectors_f64(left, right) };
        sum + scalar::multiply_sum_tail_f64(left, right, end)
    }

    /// Returns the AVX2 f64 vector sum and processed complex samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx2_sign_f64() };
        let mut accum = _mm256_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx2_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_avx2_f64(as_f64_slice(right), index * 2) };
            accum = _mm256_add_pd(accum, complex_multiply_avx2_f64(left_vector, right_vector, sign));
            index += AVX2_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_avx2_f64(accum) })
    }

    /// Returns the AVX2 f64 conjugate-multiply vector sum and samples.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn avx2_complex_conjugate_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), AVX2_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx2_sign_f64() };
        let conjugate_sign = unsafe { load_avx2_conjugate_sign_f64() };
        let mut accum = _mm256_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx2_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_avx2_f64(as_f64_slice(right), index * 2) };
            accum = _mm256_add_pd(
                accum,
                complex_conjugate_multiply_avx2_f64(left_vector, right_vector, sign, conjugate_sign),
            );
            index += AVX2_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_avx2_f64(accum) })
    }

    /// Writes AVX-512 f64 additions and returns the processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_f64_vectors(left: &[f64], right: &[f64], output: &mut [f64]) -> usize {
        let end = vectorized_end(left.len(), AVX512_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm512_add_pd(load_avx512_f64(left, index), load_avx512_f64(right, index)) };
            unsafe { _mm512_storeu_pd(output.as_mut_ptr().add(index), sum) };
            index += AVX512_F64_LANES;
        }
        end
    }

    /// Adds AVX-512 f64 samples and returns the processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_assign_f64_vectors(target: &mut [f64], input: &[f64]) -> usize {
        let end = vectorized_end(target.len(), AVX512_F64_LANES);
        let mut index = 0;
        while index < end {
            let sum = unsafe { _mm512_add_pd(load_avx512_f64(target, index), load_avx512_f64(input, index)) };
            unsafe { _mm512_storeu_pd(target.as_mut_ptr().add(index), sum) };
            index += AVX512_F64_LANES;
        }
        end
    }

    /// Adds AVX-512 f64 complex add-scaled values and returns samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_add_scaled_vectors_f64(target: &mut [Complex<f64>], scale: Complex<f64>, input: &[Complex<f64>]) -> usize {
        let end = vectorized_end(input.len(), AVX512_COMPLEX_LANES_F64);
        let scale_re = _mm512_set1_pd(scale.re);
        let scale_im = unsafe { _mm512_mul_pd(_mm512_set1_pd(scale.im), load_avx512_sign_f64()) };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_add_scaled_f64(target, input, index, scale_re, scale_im) };
            index += AVX512_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes AVX-512 f64 complex products and returns processed samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx512_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_multiply_f64(left, right, output, index, sign) };
            index += AVX512_COMPLEX_LANES_F64;
        }
        end
    }

    /// Writes AVX-512 f64 conjugate-multiply products and returns samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_conjugate_multiply_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>]) -> usize {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx512_sign_f64() };
        let conjugate_sign = unsafe { load_avx512_conjugate_sign_f64() };
        let mut index = 0;
        while index < end {
            unsafe { store_avx512_conjugate_multiply_f64(left, right, output, index, sign, conjugate_sign) };
            index += AVX512_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the AVX-512 f64 square sum and processed prefix length.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_sum_squared_f64_vectors(input: &[f64]) -> (usize, f64) {
        let end = vectorized_end(input.len(), AVX512_F64_LANES);
        let mut accum = _mm512_setzero_pd();
        let mut index = 0;
        while index < end {
            let values = unsafe { load_avx512_f64(input, index) };
            accum = _mm512_add_pd(accum, _mm512_mul_pd(values, values));
            index += AVX512_F64_LANES;
        }
        (end, unsafe { horizontal_sum_avx512_f64(accum) })
    }

    /// Writes AVX-512 f64 magnitudes and returns processed samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_magnitude_vectors_f64(input: &[Complex<f64>], output: &mut [f64]) -> usize {
        let end = vectorized_end(input.len(), AVX512_COMPLEX_LANES_F64);
        let input_f64 = as_f64_slice(input);
        let mut index = 0;
        while index < end {
            let magnitudes = unsafe { avx512_magnitudes_f64(input_f64, index) };
            output[index..index + AVX512_COMPLEX_LANES_F64].copy_from_slice(&magnitudes);
            index += AVX512_COMPLEX_LANES_F64;
        }
        end
    }

    /// Returns the AVX-512 f64 sum of pairwise complex products.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_multiply_sum_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> Complex<f64> {
        let (end, sum) = unsafe { avx512_complex_multiply_sum_vectors_f64(left, right) };
        sum + scalar::multiply_sum_tail_f64(left, right, end)
    }

    /// Returns the AVX-512 f64 vector sum and processed complex samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx512_sign_f64() };
        let mut accum = _mm512_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx512_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_avx512_f64(as_f64_slice(right), index * 2) };
            accum = _mm512_add_pd(accum, complex_multiply_avx512_f64(left_vector, right_vector, sign));
            index += AVX512_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_avx512_f64(accum) })
    }

    /// Returns the AVX-512 f64 conjugate-multiply vector sum and samples.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_complex_conjugate_multiply_sum_vectors_f64(left: &[Complex<f64>], right: &[Complex<f64>]) -> (usize, Complex<f64>) {
        let end = vectorized_end(left.len(), AVX512_COMPLEX_LANES_F64);
        let sign = unsafe { load_avx512_sign_f64() };
        let conjugate_sign = unsafe { load_avx512_conjugate_sign_f64() };
        let mut accum = _mm512_setzero_pd();
        let mut index = 0;
        while index < end {
            let left_vector = unsafe { load_avx512_f64(as_f64_slice(left), index * 2) };
            let right_vector = unsafe { load_avx512_f64(as_f64_slice(right), index * 2) };
            accum = _mm512_add_pd(
                accum,
                complex_conjugate_multiply_avx512_f64(left_vector, right_vector, sign, conjugate_sign),
            );
            index += AVX512_COMPLEX_LANES_F64;
        }
        (end, unsafe { horizontal_complex_sum_avx512_f64(accum) })
    }

    /// Loads two f64 values from an unaligned slice offset.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse_f64(input: &[f64], index: usize) -> __m128d {
        unsafe { _mm_loadu_pd(input.as_ptr().add(index)) }
    }

    /// Loads the SSE f64 complex sign pattern.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse_sign_f64() -> __m128d {
        unsafe { _mm_loadu_pd(SSE_SIGN_F64.as_ptr()) }
    }

    /// Loads the SSE f64 conjugate sign pattern.
    #[target_feature(enable = "sse2")]
    unsafe fn load_sse_conjugate_sign_f64() -> __m128d {
        unsafe { _mm_loadu_pd(SSE_CONJUGATE_SIGN_F64.as_ptr()) }
    }

    /// Stores one SSE f64 add-scaled vector into `target`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_add_scaled_f64(target: &mut [Complex<f64>], input: &[Complex<f64>], index: usize, scale_re: __m128d, scale_im: __m128d) {
        let input_vector = unsafe { load_sse_f64(as_f64_slice(input), index * 2) };
        let product = unsafe { complex_scale_sse_f64(input_vector, scale_re, scale_im) };
        let target_vector = unsafe { load_sse_f64(as_f64_slice(target), index * 2) };
        let output = _mm_add_pd(target_vector, product);
        unsafe { _mm_storeu_pd(as_f64_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one SSE f64 complex multiplication vector into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_multiply_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>], index: usize, sign: __m128d) {
        let left_vector = unsafe { load_sse_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_sse_f64(as_f64_slice(right), index * 2) };
        let product = unsafe { complex_multiply_sse_f64(left_vector, right_vector, sign) };
        unsafe { _mm_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one SSE f64 conjugate-multiply vector into `output`.
    #[target_feature(enable = "sse2")]
    unsafe fn store_sse_conjugate_multiply_f64(
        left: &[Complex<f64>],
        right: &[Complex<f64>],
        output: &mut [Complex<f64>],
        index: usize,
        sign: __m128d,
        conjugate_sign: __m128d,
    ) {
        let left_vector = unsafe { load_sse_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_sse_f64(as_f64_slice(right), index * 2) };
        let product = unsafe { complex_conjugate_multiply_sse_f64(left_vector, right_vector, sign, conjugate_sign) };
        unsafe { _mm_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed SSE f64 complex values by a broadcast scalar.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_scale_sse_f64(input: __m128d, scale_re: __m128d, scale_im: __m128d) -> __m128d {
        let swapped = _mm_shuffle_pd(input, input, 0b01);
        let real_product = _mm_mul_pd(input, scale_re);
        let imag_product = _mm_mul_pd(swapped, scale_im);
        _mm_add_pd(real_product, imag_product)
    }

    /// Multiplies packed SSE f64 complex values pairwise.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_multiply_sse_f64(left: __m128d, right: __m128d, sign: __m128d) -> __m128d {
        let right_re = _mm_shuffle_pd(right, right, 0b00);
        let right_im = _mm_mul_pd(_mm_shuffle_pd(right, right, 0b11), sign);
        unsafe { complex_scale_sse_f64(left, right_re, right_im) }
    }

    /// Multiplies packed conjugated SSE f64 complex values pairwise.
    #[target_feature(enable = "sse2")]
    unsafe fn complex_conjugate_multiply_sse_f64(left: __m128d, right: __m128d, sign: __m128d, conjugate_sign: __m128d) -> __m128d {
        let conjugated_left = _mm_mul_pd(left, conjugate_sign);
        unsafe { complex_multiply_sse_f64(conjugated_left, right, sign) }
    }

    /// Horizontally sums an SSE f64 vector.
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_sum_sse_f64(input: __m128d) -> f64 {
        let mut values = [0.0; SSE_F64_LANES];
        unsafe { _mm_storeu_pd(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed SSE f64 complex lanes.
    #[target_feature(enable = "sse2")]
    unsafe fn horizontal_complex_sum_sse_f64(input: __m128d) -> Complex<f64> {
        let mut values = [0.0; SSE_F64_LANES];
        unsafe { _mm_storeu_pd(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates one SSE f64 complex magnitude.
    #[target_feature(enable = "sse2")]
    unsafe fn sse2_magnitude_f64(input_f64: &[f64], index: usize) -> f64 {
        let values = unsafe { load_sse_f64(input_f64, index * 2) };
        let squares = _mm_mul_pd(values, values);
        let paired = _mm_add_pd(squares, _mm_shuffle_pd(squares, squares, 0b01));
        let roots = _mm_sqrt_pd(paired);
        let mut lanes = [0.0; SSE_F64_LANES];
        unsafe { _mm_storeu_pd(lanes.as_mut_ptr(), roots) };
        lanes[0]
    }

    /// Loads four f64 values from an unaligned slice offset.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2_f64(input: &[f64], index: usize) -> __m256d {
        unsafe { _mm256_loadu_pd(input.as_ptr().add(index)) }
    }

    /// Loads the AVX2 f64 complex sign pattern.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2_sign_f64() -> __m256d {
        unsafe { _mm256_loadu_pd(AVX2_SIGN_F64.as_ptr()) }
    }

    /// Loads the AVX2 f64 conjugate sign pattern.
    #[target_feature(enable = "avx2")]
    unsafe fn load_avx2_conjugate_sign_f64() -> __m256d {
        unsafe { _mm256_loadu_pd(AVX2_CONJUGATE_SIGN_F64.as_ptr()) }
    }

    /// Stores one AVX2 f64 add-scaled vector into `target`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_add_scaled_f64(
        target: &mut [Complex<f64>],
        input: &[Complex<f64>],
        index: usize,
        scale_re: __m256d,
        scale_im: __m256d,
    ) {
        let input_vector = unsafe { load_avx2_f64(as_f64_slice(input), index * 2) };
        let product = complex_scale_avx2_f64(input_vector, scale_re, scale_im);
        let target_vector = unsafe { load_avx2_f64(as_f64_slice(target), index * 2) };
        let output = _mm256_add_pd(target_vector, product);
        unsafe { _mm256_storeu_pd(as_f64_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one AVX2 f64 complex multiplication vector into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_multiply_f64(left: &[Complex<f64>], right: &[Complex<f64>], output: &mut [Complex<f64>], index: usize, sign: __m256d) {
        let left_vector = unsafe { load_avx2_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_avx2_f64(as_f64_slice(right), index * 2) };
        let product = complex_multiply_avx2_f64(left_vector, right_vector, sign);
        unsafe { _mm256_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one AVX2 f64 conjugate-multiply vector into `output`.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    unsafe fn store_avx2_conjugate_multiply_f64(
        left: &[Complex<f64>],
        right: &[Complex<f64>],
        output: &mut [Complex<f64>],
        index: usize,
        sign: __m256d,
        conjugate_sign: __m256d,
    ) {
        let left_vector = unsafe { load_avx2_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_avx2_f64(as_f64_slice(right), index * 2) };
        let product = complex_conjugate_multiply_avx2_f64(left_vector, right_vector, sign, conjugate_sign);
        unsafe { _mm256_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed AVX2 f64 complex values by a broadcast scalar.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_scale_avx2_f64(input: __m256d, scale_re: __m256d, scale_im: __m256d) -> __m256d {
        let swapped = _mm256_permute_pd(input, 0b0101);
        let real_product = _mm256_mul_pd(input, scale_re);
        _mm256_fmadd_pd(swapped, scale_im, real_product)
    }

    /// Multiplies packed AVX2 f64 complex values pairwise.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_multiply_avx2_f64(left: __m256d, right: __m256d, sign: __m256d) -> __m256d {
        let right_re = _mm256_permute_pd(right, 0b0000);
        let right_im = _mm256_mul_pd(_mm256_permute_pd(right, 0b1111), sign);
        complex_scale_avx2_f64(left, right_re, right_im)
    }

    /// Multiplies packed conjugated AVX2 f64 complex values pairwise.
    #[target_feature(enable = "avx2")]
    #[target_feature(enable = "fma")]
    fn complex_conjugate_multiply_avx2_f64(left: __m256d, right: __m256d, sign: __m256d, conjugate_sign: __m256d) -> __m256d {
        let conjugated_left = _mm256_mul_pd(left, conjugate_sign);
        complex_multiply_avx2_f64(conjugated_left, right, sign)
    }

    /// Horizontally sums an AVX2 f64 vector.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_sum_avx2_f64(input: __m256d) -> f64 {
        let mut values = [0.0; AVX2_F64_LANES];
        unsafe { _mm256_storeu_pd(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed AVX2 f64 complex lanes.
    #[target_feature(enable = "avx2")]
    unsafe fn horizontal_complex_sum_avx2_f64(input: __m256d) -> Complex<f64> {
        let mut values = [0.0; AVX2_F64_LANES];
        unsafe { _mm256_storeu_pd(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates two AVX2 f64 complex magnitudes.
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_magnitudes_f64(input_f64: &[f64], index: usize) -> [f64; AVX2_COMPLEX_LANES_F64] {
        let values = unsafe { load_avx2_f64(input_f64, index * 2) };
        let squares = _mm256_mul_pd(values, values);
        let paired = _mm256_add_pd(squares, _mm256_permute_pd(squares, 0b0101));
        let roots = _mm256_sqrt_pd(paired);
        let mut lanes = [0.0; AVX2_F64_LANES];
        unsafe { _mm256_storeu_pd(lanes.as_mut_ptr(), roots) };
        [lanes[0], lanes[2]]
    }

    /// Loads eight f64 values from an unaligned slice offset.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512_f64(input: &[f64], index: usize) -> __m512d {
        unsafe { _mm512_loadu_pd(input.as_ptr().add(index)) }
    }

    /// Loads the AVX-512 f64 complex sign pattern.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512_sign_f64() -> __m512d {
        unsafe { _mm512_loadu_pd(AVX512_SIGN_F64.as_ptr()) }
    }

    /// Loads the AVX-512 f64 conjugate sign pattern.
    #[target_feature(enable = "avx512f")]
    unsafe fn load_avx512_conjugate_sign_f64() -> __m512d {
        unsafe { _mm512_loadu_pd(AVX512_CONJUGATE_SIGN_F64.as_ptr()) }
    }

    /// Stores one AVX-512 f64 add-scaled vector into `target`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_add_scaled_f64(
        target: &mut [Complex<f64>],
        input: &[Complex<f64>],
        index: usize,
        scale_re: __m512d,
        scale_im: __m512d,
    ) {
        let input_vector = unsafe { load_avx512_f64(as_f64_slice(input), index * 2) };
        let product = complex_scale_avx512_f64(input_vector, scale_re, scale_im);
        let target_vector = unsafe { load_avx512_f64(as_f64_slice(target), index * 2) };
        let output = _mm512_add_pd(target_vector, product);
        unsafe { _mm512_storeu_pd(as_f64_slice_mut(target).as_mut_ptr().add(index * 2), output) };
    }

    /// Stores one AVX-512 f64 complex multiplication vector into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_multiply_f64(
        left: &[Complex<f64>],
        right: &[Complex<f64>],
        output: &mut [Complex<f64>],
        index: usize,
        sign: __m512d,
    ) {
        let left_vector = unsafe { load_avx512_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_avx512_f64(as_f64_slice(right), index * 2) };
        let product = complex_multiply_avx512_f64(left_vector, right_vector, sign);
        unsafe { _mm512_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Stores one AVX-512 f64 conjugate-multiply vector into `output`.
    #[target_feature(enable = "avx512f")]
    unsafe fn store_avx512_conjugate_multiply_f64(
        left: &[Complex<f64>],
        right: &[Complex<f64>],
        output: &mut [Complex<f64>],
        index: usize,
        sign: __m512d,
        conjugate_sign: __m512d,
    ) {
        let left_vector = unsafe { load_avx512_f64(as_f64_slice(left), index * 2) };
        let right_vector = unsafe { load_avx512_f64(as_f64_slice(right), index * 2) };
        let product = complex_conjugate_multiply_avx512_f64(left_vector, right_vector, sign, conjugate_sign);
        unsafe { _mm512_storeu_pd(as_f64_slice_mut(output).as_mut_ptr().add(index * 2), product) };
    }

    /// Multiplies packed AVX-512 f64 complex values by a broadcast scalar.
    #[target_feature(enable = "avx512f")]
    fn complex_scale_avx512_f64(input: __m512d, scale_re: __m512d, scale_im: __m512d) -> __m512d {
        let swapped = _mm512_permute_pd(input, 0b01010101);
        let real_product = _mm512_mul_pd(input, scale_re);
        _mm512_fmadd_pd(swapped, scale_im, real_product)
    }

    /// Multiplies packed AVX-512 f64 complex values pairwise.
    #[target_feature(enable = "avx512f")]
    fn complex_multiply_avx512_f64(left: __m512d, right: __m512d, sign: __m512d) -> __m512d {
        let right_re = _mm512_permute_pd(right, 0b00000000);
        let right_im = _mm512_mul_pd(_mm512_permute_pd(right, 0b11111111), sign);
        complex_scale_avx512_f64(left, right_re, right_im)
    }

    /// Multiplies packed conjugated AVX-512 f64 complex values pairwise.
    #[target_feature(enable = "avx512f")]
    fn complex_conjugate_multiply_avx512_f64(left: __m512d, right: __m512d, sign: __m512d, conjugate_sign: __m512d) -> __m512d {
        let conjugated_left = _mm512_mul_pd(left, conjugate_sign);
        complex_multiply_avx512_f64(conjugated_left, right, sign)
    }

    /// Horizontally sums an AVX-512 f64 vector.
    #[target_feature(enable = "avx512f")]
    unsafe fn horizontal_sum_avx512_f64(input: __m512d) -> f64 {
        let mut values = [0.0; AVX512_F64_LANES];
        unsafe { _mm512_storeu_pd(values.as_mut_ptr(), input) };
        values.iter().sum()
    }

    /// Horizontally sums packed AVX-512 f64 complex lanes.
    #[target_feature(enable = "avx512f")]
    unsafe fn horizontal_complex_sum_avx512_f64(input: __m512d) -> Complex<f64> {
        let mut values = [0.0; AVX512_F64_LANES];
        unsafe { _mm512_storeu_pd(values.as_mut_ptr(), input) };
        sum_interleaved_complex_lanes(&values)
    }

    /// Calculates four AVX-512 f64 complex magnitudes.
    #[target_feature(enable = "avx512f")]
    unsafe fn avx512_magnitudes_f64(input_f64: &[f64], index: usize) -> [f64; AVX512_COMPLEX_LANES_F64] {
        let values = unsafe { load_avx512_f64(input_f64, index * 2) };
        let squares = _mm512_mul_pd(values, values);
        let paired = _mm512_add_pd(squares, _mm512_permute_pd(squares, 0b01010101));
        let roots = _mm512_sqrt_pd(paired);
        let mut lanes = [0.0; AVX512_F64_LANES];
        unsafe { _mm512_storeu_pd(lanes.as_mut_ptr(), roots) };
        [lanes[0], lanes[2], lanes[4], lanes[6]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-4;
    const EPSILON_F64: f64 = 1.0e-10;

    #[test]
    fn selected_plan_matches_backend_geometry() {
        let plan = selected_iq_vector_plan();
        assert_eq!(plan.backend(), SELECTED_BACKEND);
        assert_eq!(plan.register_bits(), REGISTER_BITS);
        assert_eq!(plan.iq_lanes(), IQ_LANES);
        assert_eq!(plan.iq_lanes_f32(), IQ_LANES_F32);
        assert_eq!(plan.iq_lanes_f64(), IQ_LANES_F64);
        assert_eq!(plan.iq_lanes_for::<f32>(), IQ_LANES_F32);
        assert_eq!(plan.iq_lanes_for::<f64>(), IQ_LANES_F64);
    }

    #[test]
    fn selected_backend_matches_scalar_reference() {
        for len in test_lengths() {
            assert_backend_matches_reference::<SelectedBackend>(len);
        }
    }

    #[test]
    fn selected_backend_matches_scalar_reference_f64() {
        for len in test_lengths_f64() {
            assert_backend_matches_reference_f64::<SelectedBackend>(len);
        }
    }

    #[test]
    fn scalar_backend_matches_reference() {
        for len in test_lengths() {
            assert_backend_matches_reference::<scalar::ScalarBackend>(len);
        }
    }

    #[test]
    fn scalar_backend_matches_reference_f64() {
        for len in test_lengths_f64() {
            assert_backend_matches_reference_f64::<scalar::ScalarBackend>(len);
        }
    }

    #[test]
    fn selected_convolve_full_matches_scalar_reference() {
        assert_convolve_full_cases_match_reference::<SelectedBackend>();
    }

    #[test]
    fn selected_convolve_full_matches_scalar_reference_f64() {
        assert_convolve_full_cases_match_reference_f64::<SelectedBackend>();
    }

    #[test]
    fn scalar_convolve_full_matches_scalar_reference() {
        assert_convolve_full_cases_match_reference::<scalar::ScalarBackend>();
    }

    #[test]
    fn scalar_convolve_full_matches_scalar_reference_f64() {
        assert_convolve_full_cases_match_reference_f64::<scalar::ScalarBackend>();
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn available_x86_backends_match_reference() {
        for len in test_lengths() {
            assert_backend_matches_reference::<x86::Sse2Backend>(len);
            assert_avx2_backend_matches_reference(len);
            assert_avx512_backend_matches_reference(len);
        }
        for len in test_lengths_f64() {
            assert_backend_matches_reference_f64::<x86::Sse2Backend>(len);
            assert_avx2_backend_matches_reference_f64(len);
            assert_avx512_backend_matches_reference_f64(len);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[test]
    fn available_x86_convolve_backends_match_reference() {
        assert_convolve_full_cases_match_reference::<x86::Sse2Backend>();
        assert_avx2_convolve_backend_matches_reference();
        assert_avx512_convolve_backend_matches_reference();
        assert_convolve_full_cases_match_reference_f64::<x86::Sse2Backend>();
        assert_avx2_convolve_backend_matches_reference_f64();
        assert_avx512_convolve_backend_matches_reference_f64();
    }

    #[test]
    fn convolve_range_matches_full_slice() {
        let input = make_iq(17, 0.11);
        let kernel = make_iq(5, -0.19);
        let mut full_output = zero_iq(input.len() + kernel.len() - 1);
        let mut range_output = zero_iq(7);
        iq_convolve_full_to(&input, &kernel, &mut full_output);
        iq_convolve_range_to(&input, &kernel, 3, &mut range_output);
        assert_complex_slices_close(&range_output, &full_output[3..10], EPSILON);
    }

    #[test]
    fn convolve_range_matches_full_slice_f64() {
        let input = make_iq_f64(17, 0.11);
        let kernel = make_iq_f64(5, -0.19);
        let mut full_output = zero_iq_f64(input.len() + kernel.len() - 1);
        let mut range_output = zero_iq_f64(7);
        iq_convolve_full_to(&input, &kernel, &mut full_output);
        iq_convolve_range_to(&input, &kernel, 3, &mut range_output);
        assert_complex_slices_close_f64(&range_output, &full_output[3..10], EPSILON_F64);
    }

    #[test]
    #[should_panic(expected = "input must not be empty")]
    fn convolve_full_panics_on_empty_input() {
        let kernel = make_iq(1, 0.0);
        iq_convolve_full_to(&[], &kernel, &mut []);
    }

    #[test]
    #[should_panic(expected = "kernel must not be empty")]
    fn convolve_full_panics_on_empty_kernel() {
        let input = make_iq(1, 0.0);
        iq_convolve_full_to(&input, &[], &mut []);
    }

    #[test]
    #[should_panic(expected = "full convolution output length must equal input.len() + kernel.len() - 1")]
    fn convolve_full_panics_on_bad_output_length() {
        let input = make_iq(4, 0.0);
        let kernel = make_iq(3, 0.0);
        let mut output = zero_iq(5);
        iq_convolve_full_to(&input, &kernel, &mut output);
    }

    #[test]
    fn frequency_shifter_keeps_phase_across_blocks() {
        let input = make_iq(97, 0.11);
        let mut whole = input.clone();
        let mut chunked = input.clone();
        FrequencyShifter::new(12_500.0, 1_000_000.0).process_block(&mut whole);
        shift_in_chunks(&mut chunked);
        assert_complex_slices_close(&whole, &chunked, EPSILON);
    }

    #[test]
    fn frequency_shifter_keeps_phase_across_blocks_f64() {
        let input = make_iq_f64(97, 0.11);
        let mut whole = input.clone();
        let mut chunked = input.clone();
        FrequencyShifter::new(12_500.0_f64, 1_000_000.0_f64).process_block(&mut whole);
        shift_in_chunks_f64(&mut chunked);
        assert_complex_slices_close_f64(&whole, &chunked, EPSILON_F64);
    }

    #[test]
    fn fir_filter_keeps_delay_across_blocks() {
        let input = make_iq(89, 0.07);
        let taps = make_iq(7, -0.13);
        let mut whole_output = vec![Complex::new(0.0, 0.0); input.len()];
        let mut chunked_output = vec![Complex::new(0.0, 0.0); input.len()];
        FirFilter::new(taps.clone()).process_block_to(&input, &mut whole_output);
        fir_in_chunks(&input, &mut chunked_output, taps);
        assert_complex_slices_close(&whole_output, &chunked_output, EPSILON);
    }

    #[test]
    fn fir_filter_keeps_delay_across_blocks_f64() {
        let input = make_iq_f64(89, 0.07);
        let taps = make_iq_f64(7, -0.13);
        let mut whole_output = vec![Complex::new(0.0, 0.0); input.len()];
        let mut chunked_output = vec![Complex::new(0.0, 0.0); input.len()];
        FirFilter::new(taps.clone()).process_block_to(&input, &mut whole_output);
        fir_in_chunks_f64(&input, &mut chunked_output, taps);
        assert_complex_slices_close_f64(&whole_output, &chunked_output, EPSILON_F64);
    }

    #[test]
    fn fir_filter_block_matches_sample_reference() {
        let input = make_iq(73, -0.03);
        let taps = make_iq(9, 0.17);
        let mut block_output = zero_iq(input.len());
        let mut sample_output = zero_iq(input.len());
        FirFilter::new(taps.clone()).process_block_to(&input, &mut block_output);
        write_fir_sample_reference(&input, &mut sample_output, taps);
        assert_complex_slices_close(&block_output, &sample_output, EPSILON);
    }

    #[test]
    fn fir_filter_block_matches_sample_reference_f64() {
        let input = make_iq_f64(73, -0.03);
        let taps = make_iq_f64(9, 0.17);
        let mut block_output = zero_iq_f64(input.len());
        let mut sample_output = zero_iq_f64(input.len());
        FirFilter::new(taps.clone()).process_block_to(&input, &mut block_output);
        write_fir_sample_reference_f64(&input, &mut sample_output, taps);
        assert_complex_slices_close_f64(&block_output, &sample_output, EPSILON_F64);
    }

    #[test]
    fn fir_in_place_matches_output_mode() {
        let input = make_iq(67, 0.05);
        let taps = make_iq(5, 0.17);
        let mut output_mode = vec![Complex::new(0.0, 0.0); input.len()];
        let mut in_place_mode = input.clone();
        FirFilter::new(taps.clone()).process_block_to(&input, &mut output_mode);
        FirFilter::new(taps).process_block_in_place(&mut in_place_mode);
        assert_complex_slices_close(&output_mode, &in_place_mode, EPSILON);
    }

    #[test]
    fn fir_in_place_matches_output_mode_f64() {
        let input = make_iq_f64(67, 0.05);
        let taps = make_iq_f64(5, 0.17);
        let mut output_mode = vec![Complex::new(0.0, 0.0); input.len()];
        let mut in_place_mode = input.clone();
        FirFilter::new(taps.clone()).process_block_to(&input, &mut output_mode);
        FirFilter::new(taps).process_block_in_place(&mut in_place_mode);
        assert_complex_slices_close_f64(&output_mode, &in_place_mode, EPSILON_F64);
    }

    /// Checks one backend against scalar reference operations for a block length.
    fn assert_backend_matches_reference<B: IqBackend>(len: usize) {
        assert_add_to_matches::<B>(len);
        assert_add_in_place_matches::<B>(len);
        assert_scale_in_place_matches::<B>(len);
        assert_add_scaled_in_place_matches::<B>(len);
        assert_multiply_to_matches::<B>(len);
        assert_conjugate_multiply_to_matches::<B>(len);
        assert_conjugate_multiply_sum_matches::<B>(len);
        assert_sum_squared_magnitudes_matches::<B>(len);
        assert_calculate_magnitudes_matches::<B>(len);
    }

    /// Checks one f64 backend against scalar reference operations for a length.
    fn assert_backend_matches_reference_f64<B: IqBackendF64>(len: usize) {
        assert_add_to_matches_f64::<B>(len);
        assert_add_in_place_matches_f64::<B>(len);
        assert_scale_in_place_matches_f64::<B>(len);
        assert_add_scaled_in_place_matches_f64::<B>(len);
        assert_multiply_to_matches_f64::<B>(len);
        assert_conjugate_multiply_to_matches_f64::<B>(len);
        assert_conjugate_multiply_sum_matches_f64::<B>(len);
        assert_sum_squared_magnitudes_matches_f64::<B>(len);
        assert_calculate_magnitudes_matches_f64::<B>(len);
    }

    /// Checks backend add-to output for a block length.
    fn assert_add_to_matches<B: IqBackend>(len: usize) {
        let left = make_iq(len, 0.13);
        let right = make_iq(len, -0.21);
        let mut actual = zero_iq(len);
        let mut expected = zero_iq(len);
        B::iq_add_to(&left, &right, &mut actual);
        scalar::iq_add_to(&left, &right, &mut expected);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend in-place add output for a block length.
    fn assert_add_in_place_matches<B: IqBackend>(len: usize) {
        let mut actual = make_iq(len, 0.13);
        let mut expected = actual.clone();
        let input = make_iq(len, -0.21);
        B::iq_add_in_place(&mut actual, &input);
        scalar::iq_add_in_place(&mut expected, &input);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend in-place scale output for a block length.
    fn assert_scale_in_place_matches<B: IqBackend>(len: usize) {
        let mut actual = make_iq(len, 0.13);
        let mut expected = actual.clone();
        B::iq_scale_in_place(&mut actual, -1.75);
        scalar::iq_scale_in_place(&mut expected, -1.75);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend add-scaled output for a block length.
    fn assert_add_scaled_in_place_matches<B: IqBackend>(len: usize) {
        let mut actual = make_iq(len, 0.13);
        let mut expected = actual.clone();
        let input = make_iq(len, -0.21);
        let scale = Complex::new(0.75, -0.25);
        B::iq_add_scaled_in_place(&mut actual, scale, &input);
        scalar::iq_add_scaled_in_place(&mut expected, scale, &input);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend multiply output for a block length.
    fn assert_multiply_to_matches<B: IqBackend>(len: usize) {
        let left = make_iq(len, 0.13);
        let right = make_iq(len, -0.21);
        let mut actual = zero_iq(len);
        let mut expected = zero_iq(len);
        B::iq_multiply_to(&left, &right, &mut actual);
        scalar::iq_multiply_to(&left, &right, &mut expected);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend conjugate-multiply output for a block length.
    fn assert_conjugate_multiply_to_matches<B: IqBackend>(len: usize) {
        let left = make_iq(len, 0.13);
        let right = make_iq(len, -0.21);
        let mut actual = zero_iq(len);
        let mut expected = zero_iq(len);
        B::iq_conjugate_multiply_to(&left, &right, &mut actual);
        scalar::iq_conjugate_multiply_to(&left, &right, &mut expected);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks backend conjugate-multiply sum output for a block length.
    fn assert_conjugate_multiply_sum_matches<B: IqBackend>(len: usize) {
        let left = make_iq(len, 0.13);
        let right = make_iq(len, -0.21);
        let actual = B::iq_conjugate_multiply_sum(&left, &right);
        let expected = scalar::iq_conjugate_multiply_sum(&left, &right);
        assert_complex_close(actual, expected, 2.0e-3);
    }

    /// Checks backend sum of squared magnitudes output for a block length.
    fn assert_sum_squared_magnitudes_matches<B: IqBackend>(len: usize) {
        let input = make_iq(len, 0.13);
        let actual = B::iq_sum_squared_magnitudes(&input);
        let expected = scalar::iq_sum_squared_magnitudes(&input);
        assert_close(actual, expected, 2.0e-3);
    }

    /// Checks backend magnitude output for a block length.
    fn assert_calculate_magnitudes_matches<B: IqBackend>(len: usize) {
        let input = make_iq(len, 0.13);
        let mut actual = vec![0.0; len];
        let mut expected = vec![0.0; len];
        B::iq_calculate_magnitudes_to(&input, &mut actual);
        scalar::iq_calculate_magnitudes_to(&input, &mut expected);
        assert_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks f64 backend add-to output for a block length.
    fn assert_add_to_matches_f64<B: IqBackendF64>(len: usize) {
        let left = make_iq_f64(len, 0.13);
        let right = make_iq_f64(len, -0.21);
        let mut actual = zero_iq_f64(len);
        let mut expected = zero_iq_f64(len);
        B::iq_add_to(&left, &right, &mut actual);
        scalar::iq_add_to_f64(&left, &right, &mut expected);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend in-place add output for a block length.
    fn assert_add_in_place_matches_f64<B: IqBackendF64>(len: usize) {
        let mut actual = make_iq_f64(len, 0.13);
        let mut expected = actual.clone();
        let input = make_iq_f64(len, -0.21);
        B::iq_add_in_place(&mut actual, &input);
        scalar::iq_add_in_place_f64(&mut expected, &input);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend in-place scale output for a block length.
    fn assert_scale_in_place_matches_f64<B: IqBackendF64>(len: usize) {
        let mut actual = make_iq_f64(len, 0.13);
        let mut expected = actual.clone();
        B::iq_scale_in_place(&mut actual, -1.75);
        scalar::iq_scale_in_place_f64(&mut expected, -1.75);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend add-scaled output for a block length.
    fn assert_add_scaled_in_place_matches_f64<B: IqBackendF64>(len: usize) {
        let mut actual = make_iq_f64(len, 0.13);
        let mut expected = actual.clone();
        let input = make_iq_f64(len, -0.21);
        let scale = Complex::new(0.75, -0.25);
        B::iq_add_scaled_in_place(&mut actual, scale, &input);
        scalar::iq_add_scaled_in_place_f64(&mut expected, scale, &input);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend multiply output for a block length.
    fn assert_multiply_to_matches_f64<B: IqBackendF64>(len: usize) {
        let left = make_iq_f64(len, 0.13);
        let right = make_iq_f64(len, -0.21);
        let mut actual = zero_iq_f64(len);
        let mut expected = zero_iq_f64(len);
        B::iq_multiply_to(&left, &right, &mut actual);
        scalar::iq_multiply_to_f64(&left, &right, &mut expected);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend conjugate-multiply output for a block length.
    fn assert_conjugate_multiply_to_matches_f64<B: IqBackendF64>(len: usize) {
        let left = make_iq_f64(len, 0.13);
        let right = make_iq_f64(len, -0.21);
        let mut actual = zero_iq_f64(len);
        let mut expected = zero_iq_f64(len);
        B::iq_conjugate_multiply_to(&left, &right, &mut actual);
        scalar::iq_conjugate_multiply_to_f64(&left, &right, &mut expected);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks f64 backend conjugate-multiply sum output for a block length.
    fn assert_conjugate_multiply_sum_matches_f64<B: IqBackendF64>(len: usize) {
        let left = make_iq_f64(len, 0.13);
        let right = make_iq_f64(len, -0.21);
        let actual = B::iq_conjugate_multiply_sum(&left, &right);
        let expected = scalar::iq_conjugate_multiply_sum_f64(&left, &right);
        assert_complex_close_f64(actual, expected, 1.0e-9);
    }

    /// Checks f64 backend sum of squared magnitudes output for a block length.
    fn assert_sum_squared_magnitudes_matches_f64<B: IqBackendF64>(len: usize) {
        let input = make_iq_f64(len, 0.13);
        let actual = B::iq_sum_squared_magnitudes(&input);
        let expected = scalar::iq_sum_squared_magnitudes_f64(&input);
        assert_close_f64(actual, expected, 1.0e-9);
    }

    /// Checks f64 backend magnitude output for a block length.
    fn assert_calculate_magnitudes_matches_f64<B: IqBackendF64>(len: usize) {
        let input = make_iq_f64(len, 0.13);
        let mut actual = vec![0.0; len];
        let mut expected = vec![0.0; len];
        B::iq_calculate_magnitudes_to(&input, &mut actual);
        scalar::iq_calculate_magnitudes_to_f64(&input, &mut expected);
        assert_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Checks convolution output against scalar reference cases.
    fn assert_convolve_full_cases_match_reference<B: IqBackend>() {
        for input_len in convolution_input_lengths() {
            for kernel_len in convolution_kernel_lengths() {
                assert_convolve_full_matches::<B>(input_len, kernel_len);
            }
        }
    }

    /// Checks f64 convolution output against scalar reference cases.
    fn assert_convolve_full_cases_match_reference_f64<B: IqBackendF64>() {
        for input_len in convolution_input_lengths_f64() {
            for kernel_len in convolution_kernel_lengths() {
                assert_convolve_full_matches_f64::<B>(input_len, kernel_len);
            }
        }
    }

    /// Checks full convolution output for one input and kernel length.
    fn assert_convolve_full_matches<B: IqBackend>(input_len: usize, kernel_len: usize) {
        let input = make_iq(input_len, 0.13);
        let kernel = make_iq(kernel_len, -0.21);
        let mut actual = zero_iq(input_len + kernel_len - 1);
        let mut expected = zero_iq(input_len + kernel_len - 1);
        B::iq_convolve_range_to(&input, &kernel, 0, &mut actual);
        scalar::iq_convolve_range_to(&input, &kernel, 0, &mut expected);
        assert_complex_slices_close(&actual, &expected, EPSILON);
    }

    /// Checks f64 full convolution output for one input and kernel length.
    fn assert_convolve_full_matches_f64<B: IqBackendF64>(input_len: usize, kernel_len: usize) {
        let input = make_iq_f64(input_len, 0.13);
        let kernel = make_iq_f64(kernel_len, -0.21);
        let mut actual = zero_iq_f64(input_len + kernel_len - 1);
        let mut expected = zero_iq_f64(input_len + kernel_len - 1);
        B::iq_convolve_range_to(&input, &kernel, 0, &mut actual);
        scalar::iq_convolve_range_to_f64(&input, &kernel, 0, &mut expected);
        assert_complex_slices_close_f64(&actual, &expected, EPSILON_F64);
    }

    /// Runs AVX2 backend checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx2_backend_matches_reference(len: usize) {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert_backend_matches_reference::<x86::Avx2Backend>(len);
        }
    }

    /// Runs AVX-512 backend checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx512_backend_matches_reference(len: usize) {
        if std::is_x86_feature_detected!("avx512f") {
            assert_backend_matches_reference::<x86::Avx512Backend>(len);
        }
    }

    /// Runs AVX2 f64 backend checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx2_backend_matches_reference_f64(len: usize) {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert_backend_matches_reference_f64::<x86::Avx2Backend>(len);
        }
    }

    /// Runs AVX-512 f64 backend checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx512_backend_matches_reference_f64(len: usize) {
        if std::is_x86_feature_detected!("avx512f") {
            assert_backend_matches_reference_f64::<x86::Avx512Backend>(len);
        }
    }

    /// Runs AVX2 convolution checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx2_convolve_backend_matches_reference() {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert_convolve_full_cases_match_reference::<x86::Avx2Backend>();
        }
    }

    /// Runs AVX-512 convolution checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx512_convolve_backend_matches_reference() {
        if std::is_x86_feature_detected!("avx512f") {
            assert_convolve_full_cases_match_reference::<x86::Avx512Backend>();
        }
    }

    /// Runs AVX2 f64 convolution checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx2_convolve_backend_matches_reference_f64() {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            assert_convolve_full_cases_match_reference_f64::<x86::Avx2Backend>();
        }
    }

    /// Runs AVX-512 f64 convolution checks when the current test CPU supports it.
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn assert_avx512_convolve_backend_matches_reference_f64() {
        if std::is_x86_feature_detected!("avx512f") {
            assert_convolve_full_cases_match_reference_f64::<x86::Avx512Backend>();
        }
    }

    /// Returns block lengths around and across the selected lane width.
    fn test_lengths() -> Vec<usize> {
        vec![0, 1, IQ_LANES.saturating_sub(1), IQ_LANES, IQ_LANES + 1, IQ_LANES * 3 + 2, 31]
    }

    /// Returns block lengths around and across the selected f64 lane width.
    fn test_lengths_f64() -> Vec<usize> {
        vec![
            0,
            1,
            IQ_LANES_F64.saturating_sub(1),
            IQ_LANES_F64,
            IQ_LANES_F64 + 1,
            IQ_LANES_F64 * 3 + 2,
            31,
        ]
    }

    /// Returns convolution input lengths around lanes and larger blocks.
    fn convolution_input_lengths() -> Vec<usize> {
        let mut lengths = Vec::new();
        for len in [1, 2, IQ_LANES.saturating_sub(1), IQ_LANES, IQ_LANES + 1, IQ_LANES * 3 + 2, 31, 128] {
            push_unique_nonzero(&mut lengths, len);
        }
        lengths
    }

    /// Returns f64 convolution input lengths around lanes and larger blocks.
    fn convolution_input_lengths_f64() -> Vec<usize> {
        let mut lengths = Vec::new();
        for len in [
            1,
            2,
            IQ_LANES_F64.saturating_sub(1),
            IQ_LANES_F64,
            IQ_LANES_F64 + 1,
            IQ_LANES_F64 * 3 + 2,
            31,
            128,
        ] {
            push_unique_nonzero(&mut lengths, len);
        }
        lengths
    }

    /// Returns kernel lengths used by direct convolution checks.
    fn convolution_kernel_lengths() -> [usize; 4] {
        [1, 2, 5, 51]
    }

    /// Pushes a nonzero length when it is not already present.
    fn push_unique_nonzero(lengths: &mut Vec<usize>, len: usize) {
        if len != 0 && !lengths.contains(&len) {
            lengths.push(len);
        }
    }

    /// Builds deterministic IQ samples for tests.
    fn make_iq(len: usize, offset: f32) -> Vec<Complex<f32>> {
        (0..len)
            .map(|index| {
                let value = index as f32 + offset;
                Complex::new(value.sin() * 0.5, value.cos() * -0.25)
            })
            .collect()
    }

    /// Builds deterministic f64 IQ samples for tests.
    fn make_iq_f64(len: usize, offset: f64) -> Vec<Complex<f64>> {
        (0..len)
            .map(|index| {
                let value = index as f64 + offset;
                Complex::new(value.sin() * 0.5, value.cos() * -0.25)
            })
            .collect()
    }

    /// Builds zeroed IQ samples for tests.
    fn zero_iq(len: usize) -> Vec<Complex<f32>> {
        vec![Complex::new(0.0, 0.0); len]
    }

    /// Builds zeroed f64 IQ samples for tests.
    fn zero_iq_f64(len: usize) -> Vec<Complex<f64>> {
        vec![Complex::new(0.0, 0.0); len]
    }

    /// Applies a frequency shifter through uneven block sizes.
    fn shift_in_chunks(samples: &mut [Complex<f32>]) {
        let mut shifter = FrequencyShifter::new(12_500.0, 1_000_000.0);
        for chunk in samples.chunks_mut(11) {
            shifter.process_block(chunk);
        }
    }

    /// Applies an f64 frequency shifter through uneven block sizes.
    fn shift_in_chunks_f64(samples: &mut [Complex<f64>]) {
        let mut shifter = FrequencyShifter::new(12_500.0_f64, 1_000_000.0_f64);
        for chunk in samples.chunks_mut(11) {
            shifter.process_block(chunk);
        }
    }

    /// Applies a FIR filter through uneven block sizes.
    fn fir_in_chunks(input: &[Complex<f32>], output: &mut [Complex<f32>], taps: Vec<Complex<f32>>) {
        let mut filter = FirFilter::new(taps);
        for (input_chunk, output_chunk) in input.chunks(13).zip(output.chunks_mut(13)) {
            filter.process_block_to(input_chunk, output_chunk);
        }
    }

    /// Applies an f64 FIR filter through uneven block sizes.
    fn fir_in_chunks_f64(input: &[Complex<f64>], output: &mut [Complex<f64>], taps: Vec<Complex<f64>>) {
        let mut filter = FirFilter::new(taps);
        for (input_chunk, output_chunk) in input.chunks(13).zip(output.chunks_mut(13)) {
            filter.process_block_to(input_chunk, output_chunk);
        }
    }

    /// Writes scalar f32 FIR output using one-sample processing.
    fn write_fir_sample_reference(input: &[Complex<f32>], output: &mut [Complex<f32>], taps: Vec<Complex<f32>>) {
        let mut filter = FirFilter::new(taps);
        for (sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = filter.process_sample(*sample);
        }
    }

    /// Writes scalar f64 FIR output using one-sample processing.
    fn write_fir_sample_reference_f64(input: &[Complex<f64>], output: &mut [Complex<f64>], taps: Vec<Complex<f64>>) {
        let mut filter = FirFilter::new(taps);
        for (sample, output_sample) in input.iter().zip(output.iter_mut()) {
            *output_sample = filter.process_sample(*sample);
        }
    }

    /// Asserts two complex slices are close.
    fn assert_complex_slices_close(actual: &[Complex<f32>], expected: &[Complex<f32>], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_sample, expected_sample) in actual.iter().zip(expected.iter()) {
            assert_close(actual_sample.re, expected_sample.re, epsilon);
            assert_close(actual_sample.im, expected_sample.im, epsilon);
        }
    }

    /// Asserts two complex values are close.
    fn assert_complex_close(actual: Complex<f32>, expected: Complex<f32>, epsilon: f32) {
        assert_close(actual.re, expected.re, epsilon);
        assert_close(actual.im, expected.im, epsilon);
    }

    /// Asserts two f64 complex slices are close.
    fn assert_complex_slices_close_f64(actual: &[Complex<f64>], expected: &[Complex<f64>], epsilon: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual_sample, expected_sample) in actual.iter().zip(expected.iter()) {
            assert_close_f64(actual_sample.re, expected_sample.re, epsilon);
            assert_close_f64(actual_sample.im, expected_sample.im, epsilon);
        }
    }

    /// Asserts two f64 complex values are close.
    fn assert_complex_close_f64(actual: Complex<f64>, expected: Complex<f64>, epsilon: f64) {
        assert_close_f64(actual.re, expected.re, epsilon);
        assert_close_f64(actual.im, expected.im, epsilon);
    }

    /// Asserts two f32 slices are close.
    fn assert_slices_close(actual: &[f32], expected: &[f32], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
            assert_close(*actual_value, *expected_value, epsilon);
        }
    }

    /// Asserts two f64 slices are close.
    fn assert_slices_close_f64(actual: &[f64], expected: &[f64], epsilon: f64) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
            assert_close_f64(*actual_value, *expected_value, epsilon);
        }
    }

    /// Asserts two f32 values are close.
    fn assert_close(actual: f32, expected: f32, epsilon: f32) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "actual {actual} expected {expected} epsilon {epsilon}"
        );
    }

    /// Asserts two f64 values are close.
    fn assert_close_f64(actual: f64, expected: f64, epsilon: f64) {
        assert!(
            (actual - expected).abs() <= epsilon,
            "actual {actual} expected {expected} epsilon {epsilon}"
        );
    }
}
