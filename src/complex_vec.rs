#![allow(dead_code)]

use crate::vector_simd;
use num_complex::Complex;
use num_traits::Float;
use std::ops::{Add, Deref, DerefMut, Index, IndexMut, Mul, Sub};

/// Convolution mode
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvMode {
    /// Full convolution: output_size = input_size + kernel_size - 1
    /// Best for pulse shaping - preserves all convolution outputs
    Full,
    /// Same mode: output_size = input_size
    /// Centers the kernel to keep output same size as input
    Same,
    /// Valid mode: output_size = input_size - kernel_size + 1
    /// Only outputs where kernel fully overlaps signal
    Valid,
}

#[derive(Clone)]
pub struct ComplexVec<T> {
    vector: Vec<Complex<T>>,
}

impl<T> ComplexVec<T>
where
    T: Float,
{
    pub fn new() -> Self {
        ComplexVec {
            vector: Vec::<Complex<T>>::new(),
        }
    }

    //Move vector into (ownership transfer)
    pub fn from_vec(vector: Vec<Complex<T>>) -> Self {
        ComplexVec { vector }
    }

    pub fn replace_vec(&mut self, vector: Vec<Complex<T>>) {
        self.vector = vector; // Ownership transferred, old vector dropped
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Complex<T>> {
        self.vector.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Complex<T>> {
        self.vector.iter_mut()
    }

    pub fn extend<I: IntoIterator<Item = Complex<T>>>(&mut self, iter: I) {
        self.vector.extend(iter);
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    // Returns real vector of sqrt(r^2 + i^2)
    pub fn abs(&mut self) -> Vec<T> {
        self.vector.iter().map(|x| x.norm()).collect()
    }

    // Normalize to unit magnitude
    pub fn normalize(&mut self) -> ComplexVec<T> {
        ComplexVec::from_vec(
            self.vector
                .iter()
                .map(|c| {
                    let mag = c.norm();
                    if mag > T::zero() { *c / mag } else { *c }
                })
                .collect(),
        )
    }

    // In-place normalize to unit magnitude
    pub fn normalize_inplace(&mut self) {
        for c in self.vector.iter_mut() {
            let mag = c.norm();
            if mag > T::zero() {
                *c = *c / mag;
            }
        }
    }

    /// Convolve with a kernel using specified mode
    /// - ConvMode::Full: output_size = input_size + kernel_size - 1 (default, best for pulse shaping)
    /// - ConvMode::Same: output_size = input_size (keeps same size as input)
    /// - ConvMode::Valid: output_size = input_size - kernel_size + 1 (only fully overlapped outputs)
    pub fn convolve(&self, kernel: &ComplexVec<T>, mode: ConvMode) -> ComplexVec<T> {
        let input_len = self.vector.len();
        let kernel_len = kernel.vector.len();

        // Compute convolution in full mode first
        let full_size = input_len + kernel_len - 1;
        let mut full_result = vec![Complex::new(T::zero(), T::zero()); full_size];

        // Full convolution computation
        // Note: This implements cross-correlation (not true convolution with kernel flip)
        // This matches the behavior of the original implementation
        for i in 0..full_size {
            let mut sum = Complex::new(T::zero(), T::zero());
            for j in 0..kernel_len {
                // Correlation: y[i] = sum(x[i+j] * h[j])  for valid indices
                // For full mode, we need to account for zero-padding
                if i + j >= kernel_len - 1 && i + j < input_len + kernel_len - 1 {
                    let signal_idx = i + j - (kernel_len - 1);
                    if signal_idx < input_len {
                        sum = sum + self.vector[signal_idx] * kernel.vector[j];
                    }
                }
            }
            full_result[i] = sum;
        }

        // Extract the appropriate portion based on mode
        let result = match mode {
            ConvMode::Full => full_result,
            ConvMode::Same => {
                // Center the output to match input size
                let delay = (kernel_len - 1) / 2;
                full_result[delay..delay + input_len].to_vec()
            }
            ConvMode::Valid => {
                // Only fully overlapped portion
                let valid_size = input_len - kernel_len + 1;
                full_result[kernel_len - 1..kernel_len - 1 + valid_size].to_vec()
            }
        };

        ComplexVec::from_vec(result)
    }

    pub fn convolve_inplace(&mut self, kernel: &ComplexVec<T>, mode: ConvMode) {
        let convolved = self.convolve(kernel, mode);
        self.vector = convolved.vector;
    }

    /// Frequency shift signal using time-domain multiplication by e^(j2πf_shift·n/fs)
    ///
    /// # Arguments
    /// * `freq_offset` - Frequency offset in Hz
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Panics
    /// Panics if sample_rate is 0 or negative
    pub fn freq_shift(&mut self, freq_offset: f64, sample_rate: f64) {
        assert!(sample_rate > 0.0, "sample_rate must be positive");

        let normalized_freq = freq_offset / sample_rate;
        let two_pi = T::from(std::f64::consts::PI).unwrap() * T::from(2.0).unwrap();

        for (n, sample) in self.vector.iter_mut().enumerate() {
            let phase = two_pi * T::from(normalized_freq).unwrap() * T::from(n as f64).unwrap();
            let phase_f64 = phase.to_f64().unwrap();

            // e^(j*phase) = cos(phase) + j*sin(phase)
            let cos_phase = T::from(phase_f64.cos()).unwrap();
            let sin_phase = T::from(phase_f64.sin()).unwrap();
            let rotator = Complex::new(cos_phase, sin_phase);

            *sample = *sample * rotator;
        }
    }

    /// Measure the average power of a signal
    ///
    /// # Arguments
    /// * `oversample_rate` - the rate at which the signal/carrier in band is oversampled at. i.e.
    /// 1000e3 sample_rate, 500e3 carrier bandwidth = 2.0 oversample_rate
    pub fn measure_power(&self, oversample_rate: Option<f64>) -> f64 {
        let mut sum_power = T::from(0.0).unwrap();
        for sample in self.vector.iter() {
            let power = (sample.re * sample.re) + (sample.im * sample.im);
            sum_power = sum_power + power;
        }
        let factor = oversample_rate.unwrap_or(1.0_f64);
        sum_power.to_f64().unwrap() / ((self.vector.len() as f64) / factor)
    }

    /// Scale signal to achieve a target average power
    ///
    /// # Arguments
    /// * `target_power` - desired average power (linear scale)
    /// * `oversample_rate` - optional oversample rate for in-band power measurement
    ///
    /// # Returns
    /// A new ComplexVec scaled to the target power
    pub fn scale_to_power(&self, target_power: f64, oversample_rate: Option<f64>) -> ComplexVec<T> {
        if target_power <= 0.0 {
            panic!("target_power must be positive");
        }

        let current_power = self.measure_power(oversample_rate);

        if current_power == 0.0 {
            // Signal is silent, return a copy as-is
            return ComplexVec::from_vec(self.vector.clone());
        }

        let scale = (target_power / current_power).sqrt();
        let scale_t = T::from(scale).unwrap();

        let result = self.vector.iter().map(|sample| sample * scale_t).collect();
        ComplexVec::from_vec(result)
    }
}

impl ComplexVec<f32> {
    /// Convolves with a kernel using the selected f32 SIMD backend.
    ///
    /// # Panics
    /// Panics when `self` or `kernel` is empty, or when `mode` is
    /// `ConvMode::Valid` and the kernel is longer than the input.
    pub fn convolve_simd(&self, kernel: &ComplexVec<f32>, mode: ConvMode) -> ComplexVec<f32> {
        let output_len = convolve_simd_output_len(self.len(), kernel.len(), mode);
        let mut output = vec![Complex::new(0.0, 0.0); output_len];
        self.convolve_simd_to(kernel, mode, &mut output);
        ComplexVec::from_vec(output)
    }

    /// Writes selected-mode convolution output using the selected f32 SIMD backend.
    ///
    /// # Panics
    /// Panics when `self` or `kernel` is empty, when `mode` is
    /// `ConvMode::Valid` and the kernel is longer than the input, or when
    /// `output.len()` does not match the selected convolution mode.
    pub fn convolve_simd_to(&self, kernel: &ComplexVec<f32>, mode: ConvMode, output: &mut [Complex<f32>]) {
        let expected_len = convolve_simd_output_len(self.len(), kernel.len(), mode);
        assert_eq!(output.len(), expected_len, "output length must match convolution mode");
        vector_simd::selected_iq_vector_plan().iq_convolve_range_to(
            &self.vector,
            &kernel.vector,
            convolve_simd_full_output_start(kernel.len(), mode),
            output,
        );
    }
}

/// Returns the output length for a f32 SIMD convolution mode.
fn convolve_simd_output_len(input_len: usize, kernel_len: usize, mode: ConvMode) -> usize {
    assert_convolve_simd_inputs(input_len, kernel_len);
    match mode {
        ConvMode::Full => input_len + kernel_len - 1,
        ConvMode::Same => input_len,
        ConvMode::Valid => valid_convolve_simd_output_len(input_len, kernel_len),
    }
}

/// Returns the full-output start offset for a f32 SIMD convolution mode.
fn convolve_simd_full_output_start(kernel_len: usize, mode: ConvMode) -> usize {
    match mode {
        ConvMode::Full => 0,
        ConvMode::Same => (kernel_len - 1) / 2,
        ConvMode::Valid => kernel_len - 1,
    }
}

/// Returns the valid-mode output length for f32 SIMD convolution.
fn valid_convolve_simd_output_len(input_len: usize, kernel_len: usize) -> usize {
    assert!(
        input_len >= kernel_len,
        "valid convolution requires input length to be at least kernel length"
    );
    input_len - kernel_len + 1
}

/// Asserts that f32 SIMD convolution inputs are non-empty.
fn assert_convolve_simd_inputs(input_len: usize, kernel_len: usize) {
    assert!(input_len > 0, "input must not be empty");
    assert!(kernel_len > 0, "kernel must not be empty");
}

impl<T> Index<usize> for ComplexVec<T>
where
    T: Float,
{
    type Output = Complex<T>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.vector[idx]
    }
}

impl<T> IndexMut<usize> for ComplexVec<T>
where
    T: Float,
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.vector[idx]
    }
}

impl<T> Deref for ComplexVec<T>
where
    T: Float,
{
    type Target = [Complex<T>];

    fn deref(&self) -> &Self::Target {
        &self.vector
    }
}

impl<T> DerefMut for ComplexVec<T>
where
    T: Float,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.vector
    }
}

impl<T> Extend<Complex<T>> for ComplexVec<T>
where
    T: Float,
{
    fn extend<I: IntoIterator<Item = Complex<T>>>(&mut self, iter: I) {
        self.vector.extend(iter);
    }
}

impl<T> Add for ComplexVec<T>
where
    T: Float,
{
    type Output = Self;

    fn add(self, other: ComplexVec<T>) -> Self {
        assert_eq!(self.vector.len(), other.vector.len(), "ComplexVec addition requires equal lengths");
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a + b).collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Add for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn add(self, other: &ComplexVec<T>) -> ComplexVec<T> {
        assert_eq!(self.vector.len(), other.vector.len(), "ComplexVec addition requires equal lengths");
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a + b).collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Sub for ComplexVec<T>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, other: ComplexVec<T>) -> Self {
        assert_eq!(self.vector.len(), other.vector.len(), "ComplexVec subtraction requires equal lengths");
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a - b).collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Sub for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn sub(self, other: &ComplexVec<T>) -> ComplexVec<T> {
        assert_eq!(self.vector.len(), other.vector.len(), "ComplexVec subtraction requires equal lengths");
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a - b).collect();
        ComplexVec::from_vec(result)
    }
}

// Element-wise multiplication: ComplexVec * ComplexVec (owned)
impl<T> Mul for ComplexVec<T>
where
    T: Float,
{
    type Output = Self;

    fn mul(self, other: ComplexVec<T>) -> Self {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec element-wise multiplication requires equal lengths"
        );
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a * b).collect();
        ComplexVec::from_vec(result)
    }
}

// Element-wise multiplication: &ComplexVec * &ComplexVec (borrowed)
impl<T> Mul for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, other: &ComplexVec<T>) -> ComplexVec<T> {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec element-wise multiplication requires equal lengths"
        );
        let result = self.vector.iter().zip(other.vector.iter()).map(|(a, b)| a * b).collect();
        ComplexVec::from_vec(result)
    }
}

// Scalar multiplication: ComplexVec * Complex<T>
impl<T> Mul<Complex<T>> for ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, scalar: Complex<T>) -> ComplexVec<T> {
        let result = self.vector.iter().map(|v| v * scalar).collect();
        ComplexVec::from_vec(result)
    }
}

// Scalar multiplication: &ComplexVec * Complex<T>
impl<T> Mul<Complex<T>> for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, scalar: Complex<T>) -> ComplexVec<T> {
        let result = self.vector.iter().map(|v| v * scalar).collect();
        ComplexVec::from_vec(result)
    }
}

// Scalar multiplication: Complex<T> * ComplexVec (commutative)
impl<T> Mul<ComplexVec<T>> for Complex<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, vec: ComplexVec<T>) -> ComplexVec<T> {
        let result = vec.vector.iter().map(|v| self * v).collect();
        ComplexVec::from_vec(result)
    }
}

// Scalar multiplication: Complex<T> * &ComplexVec (commutative)
impl<T> Mul<&ComplexVec<T>> for Complex<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, vec: &ComplexVec<T>) -> ComplexVec<T> {
        let result = vec.vector.iter().map(|v| self * v).collect();
        ComplexVec::from_vec(result)
    }
}

// Real scalar multiplication: ComplexVec * T
impl<T> Mul<T> for ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, scalar: T) -> ComplexVec<T> {
        let result = self.vector.iter().map(|v| v * scalar).collect();
        ComplexVec::from_vec(result)
    }
}

// Real scalar multiplication: &ComplexVec * T
impl<T> Mul<T> for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn mul(self, scalar: T) -> ComplexVec<T> {
        let result = self.vector.iter().map(|v| v * scalar).collect();
        ComplexVec::from_vec(result)
    }
}

// Real scalar multiplication: T * ComplexVec (commutative)
impl Mul<ComplexVec<f32>> for f32 {
    type Output = ComplexVec<f32>;

    fn mul(self, vec: ComplexVec<f32>) -> ComplexVec<f32> {
        let result = vec.vector.iter().map(|v| v * self).collect();
        ComplexVec::from_vec(result)
    }
}

impl Mul<ComplexVec<f64>> for f64 {
    type Output = ComplexVec<f64>;

    fn mul(self, vec: ComplexVec<f64>) -> ComplexVec<f64> {
        let result = vec.vector.iter().map(|v| v * self).collect();
        ComplexVec::from_vec(result)
    }
}

// Real scalar multiplication: T * &ComplexVec (commutative)
impl Mul<&ComplexVec<f32>> for f32 {
    type Output = ComplexVec<f32>;

    fn mul(self, vec: &ComplexVec<f32>) -> ComplexVec<f32> {
        let result = vec.vector.iter().map(|v| v * self).collect();
        ComplexVec::from_vec(result)
    }
}

impl Mul<&ComplexVec<f64>> for f64 {
    type Output = ComplexVec<f64>;

    fn mul(self, vec: &ComplexVec<f64>) -> ComplexVec<f64> {
        let result = vec.vector.iter().map(|v| v * self).collect();
        ComplexVec::from_vec(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_new() {
        let cv = ComplexVec::<f64>::new();
        assert_eq!(cv.len(), 0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let cv = ComplexVec::from_vec(data);
        assert_eq!(cv.len(), 2);
    }

    #[test]
    fn test_replace_vec() {
        let mut cv = ComplexVec::new();
        let data = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        cv.replace_vec(data);
        assert_eq!(cv.len(), 2);
    }

    #[test]
    fn test_len() {
        let data = vec![Complex::new(1.0, 2.0); 5];
        let cv = ComplexVec::from_vec(data);
        assert_eq!(cv.len(), 5);
    }

    #[test]
    fn test_abs() {
        let data = vec![
            Complex::new(3.0, 4.0), // mag = 5.0
            Complex::new(0.0, 1.0), // mag = 1.0
        ];
        let mut cv = ComplexVec::from_vec(data);
        let magnitudes = cv.abs();

        assert!((magnitudes[0] - 5.0).abs() < 1e-10);
        assert!((magnitudes[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![Complex::new(3.0, 4.0), Complex::new(5.0, 12.0)];
        let mut cv = ComplexVec::from_vec(data);
        let normalized = cv.normalize();
        let mags = normalized.vector;

        assert!((mags[0].norm() - 1.0).abs() < 1e-10);
        assert!((mags[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_inplace() {
        let data = vec![Complex::new(3.0, 4.0), Complex::new(5.0, 12.0)];
        let mut cv = ComplexVec::from_vec(data);
        cv.normalize_inplace();

        assert!((cv.vector[0].norm() - 1.0).abs() < 1e-10);
        assert!((cv.vector[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_magnitude() {
        let data = vec![Complex::new(0.0, 0.0), Complex::new(3.0, 4.0)];
        let mut cv = ComplexVec::from_vec(data);
        cv.normalize_inplace();

        assert_eq!(cv.vector[0], Complex::new(0.0, 0.0));
        assert!((cv.vector[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve_valid() {
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
        ];
        let kernel = vec![Complex::new(2.0, 0.0), Complex::new(1.0, 0.0)];

        let sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        let result = sig.convolve(&ker, ConvMode::Valid);

        assert_eq!(result.len(), 4); // 5 - 2 + 1 = 4
        assert_eq!(result.vector[0], Complex::new(4.0, 0.0));
        assert_eq!(result.vector[1], Complex::new(7.0, 0.0));
        assert_eq!(result.vector[2], Complex::new(10.0, 0.0));
        assert_eq!(result.vector[3], Complex::new(13.0, 0.0));
    }

    #[test]
    fn test_convolve_full() {
        let signal = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)];
        let kernel = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

        let sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        let result = sig.convolve(&ker, ConvMode::Full);

        assert_eq!(result.len(), 4); // 3 + 2 - 1 = 4
        assert_eq!(result.vector[0], Complex::new(1.0, 0.0));
        assert_eq!(result.vector[1], Complex::new(3.0, 0.0));
        assert_eq!(result.vector[2], Complex::new(5.0, 0.0));
        assert_eq!(result.vector[3], Complex::new(3.0, 0.0));
    }

    #[test]
    fn test_convolve_same() {
        let signal = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)];
        let kernel = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

        let sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        let result = sig.convolve(&ker, ConvMode::Same);

        assert_eq!(result.len(), 3); // Same as input
        // In same mode with kernel size 2, output is shifted by (2-1)/2 = 0
        assert_eq!(result.vector[0], Complex::new(1.0, 0.0));
        assert_eq!(result.vector[1], Complex::new(3.0, 0.0));
        assert_eq!(result.vector[2], Complex::new(5.0, 0.0));
    }

    #[test]
    fn test_convolve_inplace() {
        let signal = vec![Complex::new(1.0, 0.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0)];
        let kernel = vec![Complex::new(1.0, 0.0), Complex::new(1.0, 0.0)];

        let mut sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        sig.convolve_inplace(&ker, ConvMode::Valid);

        assert_eq!(sig.len(), 2);
        assert_eq!(sig.vector[0], Complex::new(3.0, 0.0));
        assert_eq!(sig.vector[1], Complex::new(5.0, 0.0));
    }

    #[test]
    fn test_convolve_impulse() {
        let signal = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let impulse = vec![Complex::new(1.0, 0.0)];

        let sig = ComplexVec::from_vec(signal.clone());
        let imp = ComplexVec::from_vec(impulse);
        let result = sig.convolve(&imp, ConvMode::Valid);

        for i in 0..signal.len() {
            println!("Signal: {}, Convolved Result: {}", signal[i], result[i]);
        }
        assert_eq!(result.len(), 2);
        assert_eq!(result.vector[0], signal[0]);
        assert_eq!(result.vector[1], signal[1]);
    }

    #[test]
    fn test_convolve_simd_matches_convolve_full() {
        assert_convolve_simd_matches_convolve(17, 5, ConvMode::Full);
        assert_convolve_simd_matches_convolve(64, 51, ConvMode::Full);
    }

    #[test]
    fn test_convolve_simd_matches_convolve_same() {
        assert_convolve_simd_matches_convolve(17, 5, ConvMode::Same);
        assert_convolve_simd_matches_convolve(64, 51, ConvMode::Same);
    }

    #[test]
    fn test_convolve_simd_matches_convolve_valid() {
        assert_convolve_simd_matches_convolve(17, 5, ConvMode::Valid);
        assert_convolve_simd_matches_convolve(64, 51, ConvMode::Valid);
    }

    #[test]
    fn test_convolve_simd_to_matches_convolve() {
        let signal = make_complex_vec_f32(23, 0.17);
        let kernel = make_complex_vec_f32(7, -0.29);
        let expected = signal.convolve(&kernel, ConvMode::Same);
        let mut actual = vec![Complex::new(0.0, 0.0); expected.len()];
        signal.convolve_simd_to(&kernel, ConvMode::Same, &mut actual);
        assert_complex_slices_close(&actual, &expected.vector, F32_EPSILON);
    }

    #[test]
    fn test_convolve_simd_impulse() {
        let signal = make_complex_vec_f32(13, 0.07);
        let impulse = ComplexVec::from_vec(vec![Complex::new(1.0, 0.0)]);
        let result = signal.convolve_simd(&impulse, ConvMode::Valid);
        assert_complex_slices_close(&result.vector, &signal.vector, F32_EPSILON);
    }

    #[test]
    #[should_panic(expected = "valid convolution requires input length to be at least kernel length")]
    fn test_convolve_simd_valid_panics_when_kernel_is_longer() {
        let signal = make_complex_vec_f32(4, 0.0);
        let kernel = make_complex_vec_f32(5, 0.0);
        signal.convolve_simd(&kernel, ConvMode::Valid);
    }

    #[test]
    fn test_indexing() {
        let data = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0), Complex::new(5.0, 6.0)];
        let mut cv = ComplexVec::from_vec(data);

        // Test read indexing
        assert_eq!(cv[0], Complex::new(1.0, 2.0));
        assert_eq!(cv[1], Complex::new(3.0, 4.0));
        assert_eq!(cv[2], Complex::new(5.0, 6.0));

        // Test write indexing
        cv[1] = Complex::new(7.0, 8.0);
        assert_eq!(cv[1], Complex::new(7.0, 8.0));
    }

    #[test]
    fn test_mul_element_wise() {
        let a = ComplexVec::from_vec(vec![Complex::new(2.0, 1.0), Complex::new(3.0, 2.0)]);
        let b = ComplexVec::from_vec(vec![Complex::new(1.0, 1.0), Complex::new(2.0, 0.0)]);

        // Test owned multiplication
        let c = a.clone() * b.clone();
        // (2+i)(1+i) = 2 + 2i + i + i^2 = 2 + 3i - 1 = 1 + 3i
        assert_eq!(c[0], Complex::new(1.0, 3.0));
        // (3+2i)(2+0i) = 6 + 4i
        assert_eq!(c[1], Complex::new(6.0, 4.0));

        // Test borrowed multiplication
        let d = &a * &b;
        assert_eq!(d[0], Complex::new(1.0, 3.0));
        assert_eq!(d[1], Complex::new(6.0, 4.0));
    }

    #[test]
    fn test_mul_complex_scalar() {
        let vec = ComplexVec::from_vec(vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)]);
        let scalar = Complex::new(2.0, 1.0);

        // Test vec * scalar (owned)
        let result1 = vec.clone() * scalar;
        // (1+2i)(2+i) = 2 + i + 4i + 2i^2 = 2 + 5i - 2 = 0 + 5i
        assert_eq!(result1[0], Complex::new(0.0, 5.0));
        // (3+4i)(2+i) = 6 + 3i + 8i + 4i^2 = 6 + 11i - 4 = 2 + 11i
        assert_eq!(result1[1], Complex::new(2.0, 11.0));

        // Test &vec * scalar (borrowed)
        let result2 = &vec * scalar;
        assert_eq!(result2[0], Complex::new(0.0, 5.0));
        assert_eq!(result2[1], Complex::new(2.0, 11.0));

        // Test scalar * vec (commutative, owned)
        let result3 = scalar * vec.clone();
        assert_eq!(result3[0], Complex::new(0.0, 5.0));
        assert_eq!(result3[1], Complex::new(2.0, 11.0));

        // Test scalar * &vec (commutative, borrowed)
        let result4 = scalar * &vec;
        assert_eq!(result4[0], Complex::new(0.0, 5.0));
        assert_eq!(result4[1], Complex::new(2.0, 11.0));
    }

    #[test]
    fn test_mul_real_scalar_f64() {
        let vec = ComplexVec::from_vec(vec![Complex::new(1.0_f64, 2.0_f64), Complex::new(3.0_f64, 4.0_f64)]);
        let scalar = 2.5_f64;

        // Test vec * scalar (owned)
        let result1 = vec.clone() * scalar;
        assert_eq!(result1[0], Complex::new(2.5, 5.0));
        assert_eq!(result1[1], Complex::new(7.5, 10.0));

        // Test &vec * scalar (borrowed)
        let result2 = &vec * scalar;
        assert_eq!(result2[0], Complex::new(2.5, 5.0));
        assert_eq!(result2[1], Complex::new(7.5, 10.0));

        // Test scalar * vec (commutative, owned)
        let result3 = scalar * vec.clone();
        assert_eq!(result3[0], Complex::new(2.5, 5.0));
        assert_eq!(result3[1], Complex::new(7.5, 10.0));

        // Test scalar * &vec (commutative, borrowed)
        let result4 = scalar * &vec;
        assert_eq!(result4[0], Complex::new(2.5, 5.0));
        assert_eq!(result4[1], Complex::new(7.5, 10.0));
    }

    #[test]
    fn test_mul_real_scalar_f32() {
        let vec = ComplexVec::from_vec(vec![Complex::new(1.0_f32, 2.0_f32), Complex::new(3.0_f32, 4.0_f32)]);
        let scalar = 2.5_f32;

        // Test vec * scalar (owned)
        let result1 = vec.clone() * scalar;
        assert_eq!(result1[0], Complex::new(2.5, 5.0));
        assert_eq!(result1[1], Complex::new(7.5, 10.0));

        // Test &vec * scalar (borrowed)
        let result2 = &vec * scalar;
        assert_eq!(result2[0], Complex::new(2.5, 5.0));
        assert_eq!(result2[1], Complex::new(7.5, 10.0));

        // Test scalar * vec (commutative, owned)
        let result3 = scalar * vec.clone();
        assert_eq!(result3[0], Complex::new(2.5, 5.0));
        assert_eq!(result3[1], Complex::new(7.5, 10.0));

        // Test scalar * &vec (commutative, borrowed)
        let result4 = scalar * &vec;
        assert_eq!(result4[0], Complex::new(2.5, 5.0));
        assert_eq!(result4[1], Complex::new(7.5, 10.0));
    }

    const F32_EPSILON: f32 = 1.0e-4;

    fn assert_convolve_simd_matches_convolve(input_len: usize, kernel_len: usize, mode: ConvMode) {
        let signal = make_complex_vec_f32(input_len, 0.13);
        let kernel = make_complex_vec_f32(kernel_len, -0.21);
        let actual = signal.convolve_simd(&kernel, mode);
        let expected = signal.convolve(&kernel, mode);
        assert_complex_slices_close(&actual.vector, &expected.vector, F32_EPSILON);
    }

    fn make_complex_vec_f32(len: usize, offset: f32) -> ComplexVec<f32> {
        ComplexVec::from_vec(
            (0..len)
                .map(|index| {
                    let value = index as f32 + offset;
                    Complex::new(value.sin() * 0.5, value.cos() * -0.25)
                })
                .collect(),
        )
    }

    fn assert_complex_slices_close(actual: &[Complex<f32>], expected: &[Complex<f32>], epsilon: f32) {
        assert_eq!(actual.len(), expected.len());
        for (actual_sample, expected_sample) in actual.iter().zip(expected.iter()) {
            assert!((actual_sample.re - expected_sample.re).abs() <= epsilon);
            assert!((actual_sample.im - expected_sample.im).abs() <= epsilon);
        }
    }
}
