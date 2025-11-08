#![allow(dead_code)]

use num_complex::{Complex};
use num_traits::Float;
use std::ops::{Index, IndexMut, Add, Sub, Deref};

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
            vector: Vec::<Complex<T>>::new()
        }
    }

    //Move vector into (ownership transfer)
    pub fn from_vec(vector: Vec<Complex<T>>) -> Self {
        ComplexVec {
            vector,
        }
    }

    pub fn replace_vec(&mut self, vector: Vec<Complex<T>>) {
        self.vector = vector;  // Ownership transferred, old vector dropped
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
        self.vector.iter().map(|x| x.norm() ).collect()
    }

    // Normalize to unit magnitude
    pub fn normalize(&mut self) -> ComplexVec<T> {
        ComplexVec::from_vec(
            self.vector.iter()
            .map(|c| {
                let mag = c.norm();
                if mag > T::zero() {
                    *c / mag
                } else {
                    *c
                }
            }).collect()
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
            },
            ConvMode::Valid => {
                // Only fully overlapped portion
                let valid_size = input_len - kernel_len + 1;
                full_result[kernel_len - 1..kernel_len - 1 + valid_size].to_vec()
            },
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
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec addition requires equal lengths"
        );
        let result = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Add for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn add(self, other: &ComplexVec<T>) -> ComplexVec<T> {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec addition requires equal lengths"
        );
        let result = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a + b)
            .collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Sub for ComplexVec<T>
where
    T: Float,
{
    type Output = Self;

    fn sub(self, other: ComplexVec<T>) -> Self {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec subtraction requires equal lengths"
        );
        let result = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a - b)
            .collect();
        ComplexVec::from_vec(result)
    }
}

impl<T> Sub for &ComplexVec<T>
where
    T: Float,
{
    type Output = ComplexVec<T>;

    fn sub(self, other: &ComplexVec<T>) -> ComplexVec<T> {
        assert_eq!(
            self.vector.len(),
            other.vector.len(),
            "ComplexVec subtraction requires equal lengths"
        );
        let result = self
            .vector
            .iter()
            .zip(other.vector.iter())
            .map(|(a, b)| a - b)
            .collect();
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
            Complex::new(3.0, 4.0),  // mag = 5.0
            Complex::new(0.0, 1.0),  // mag = 1.0
        ];
        let mut cv = ComplexVec::from_vec(data);
        let magnitudes = cv.abs();

        assert!((magnitudes[0] - 5.0).abs() < 1e-10);
        assert!((magnitudes[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 12.0),
        ];
        let mut cv = ComplexVec::from_vec(data);
        let normalized = cv.normalize();
        let mags = normalized.vector;

        assert!((mags[0].norm() - 1.0).abs() < 1e-10);
        assert!((mags[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_inplace() {
        let data = vec![
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 12.0),
        ];
        let mut cv = ComplexVec::from_vec(data);
        cv.normalize_inplace();

        assert!((cv.vector[0].norm() - 1.0).abs() < 1e-10);
        assert!((cv.vector[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_magnitude() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 4.0),
        ];
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
        let kernel = vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

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
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let kernel = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

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
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let kernel = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

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
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let kernel = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        sig.convolve_inplace(&ker, ConvMode::Valid);

        assert_eq!(sig.len(), 2);
        assert_eq!(sig.vector[0], Complex::new(3.0, 0.0));
        assert_eq!(sig.vector[1], Complex::new(5.0, 0.0));
    }

    #[test]
    fn test_convolve_impulse() {
        let signal = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ];
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
    fn test_indexing() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let mut cv = ComplexVec::from_vec(data);

        // Test read indexing
        assert_eq!(cv[0], Complex::new(1.0, 2.0));
        assert_eq!(cv[1], Complex::new(3.0, 4.0));
        assert_eq!(cv[2], Complex::new(5.0, 6.0));

        // Test write indexing
        cv[1] = Complex::new(7.0, 8.0);
        assert_eq!(cv[1], Complex::new(7.0, 8.0));
    }
}
