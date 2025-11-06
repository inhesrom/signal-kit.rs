#![allow(dead_code)]

use num_complex::Complex;
use num_traits::Float;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, StandardNormal};
use crate::complex_vec::ComplexVec;

/// Additive White Gaussian Noise (AWGN) generator
///
/// Generates complex Gaussian noise with configurable power.
/// For complex AWGN, each component (I and Q) is independently N(0, sigma)
/// where sigma = sqrt(noise_power/2) to achieve total power = noise_power
pub struct AWGN {
    sample_rate_hz: f64,
    block_size: usize,
    noise_power: f64,
    rng: StdRng,
}

impl AWGN {
    /// Create AWGN generator from a seed (reproducible)
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sample rate in Hz (for reference)
    /// * `block_size` - Number of samples to generate per block
    /// * `noise_power` - Total noise power (variance)
    /// * `seed` - RNG seed for reproducibility
    pub fn new_from_seed(
        sample_rate_hz: f64,
        block_size: usize,
        noise_power: f64,
        seed: u64,
    ) -> Self {
        AWGN {
            sample_rate_hz,
            block_size,
            noise_power,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Create AWGN generator from system entropy (non-reproducible)
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sample rate in Hz (for reference)
    /// * `block_size` - Number of samples to generate per block
    /// * `noise_power` - Total noise power (variance)
    pub fn new_from_entropy(
        sample_rate_hz: f64,
        block_size: usize,
        noise_power: f64,
    ) -> Self {
        AWGN {
            sample_rate_hz,
            block_size,
            noise_power,
            rng: StdRng::from_entropy(),
        }
    }

    /// Generate a block of complex Gaussian noise samples
    ///
    /// Returns ComplexVec<T> containing `block_size` samples of AWGN.
    /// Each component (I and Q) is independently drawn from N(0, sigma)
    /// where sigma = sqrt(noise_power/2).
    pub fn generate_block<T: Float>(&mut self) -> ComplexVec<T> {
        let mut samples = Vec::with_capacity(self.block_size);

        // Standard deviation per component for desired total power
        // Total power = I^2 + Q^2 = sigma^2 + sigma^2 = 2*sigma^2 = noise_power
        // Therefore: sigma = sqrt(noise_power/2)
        let std_dev = (self.noise_power / 2.0).sqrt();
        let std_dev_t = T::from(std_dev).unwrap();

        // Generate block_size complex samples
        for _ in 0..self.block_size {
            // Generate I and Q components from standard normal
            let i_std: f64 = StandardNormal.sample(&mut self.rng);
            let q_std: f64 = StandardNormal.sample(&mut self.rng);

            // Scale to desired standard deviation and convert to type T
            let i = T::from(i_std).unwrap() * std_dev_t;
            let q = T::from(q_std).unwrap() * std_dev_t;

            samples.push(Complex::new(i, q));
        }

        ComplexVec::from_vec(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awgn_from_seed() {
        let mut awgn = AWGN::new_from_seed(1e6, 1000, 1.0, 42);
        let block1 = awgn.generate_block::<f64>();

        assert_eq!(block1.len(), 1000);

        // Generate another block and verify it's different (stateful)
        let block2 = awgn.generate_block::<f64>();
        assert_eq!(block2.len(), 1000);

        // Blocks should be different due to RNG state progression
        assert_ne!(block1[0], block2[0]);
    }

    #[test]
    fn test_awgn_reproducibility() {
        // Same seed should produce identical sequences
        let mut awgn1 = AWGN::new_from_seed(1e6, 100, 1.0, 123);
        let mut awgn2 = AWGN::new_from_seed(1e6, 100, 1.0, 123);

        let block1 = awgn1.generate_block::<f64>();
        let block2 = awgn2.generate_block::<f64>();

        // Should be identical
        for i in 0..block1.len() {
            assert_eq!(block1[i], block2[i]);
        }
    }

    #[test]
    fn test_awgn_from_entropy() {
        let mut awgn = AWGN::new_from_entropy(1e6, 500, 2.0);
        let block = awgn.generate_block::<f32>();

        assert_eq!(block.len(), 500);

        // Basic sanity check: samples should not all be zero
        let mut has_nonzero = false;
        for i in 0..block.len() {
            if block[i].re != 0.0 || block[i].im != 0.0 {
                has_nonzero = true;
                break;
            }
        }
        assert!(has_nonzero);
    }

    #[test]
    fn test_awgn_f32_and_f64() {
        let mut awgn = AWGN::new_from_seed(1e6, 100, 1.0, 999);

        // Test with f32
        let block_f32 = awgn.generate_block::<f32>();
        assert_eq!(block_f32.len(), 100);

        // Reset RNG
        awgn = AWGN::new_from_seed(1e6, 100, 1.0, 999);

        // Test with f64
        let block_f64 = awgn.generate_block::<f64>();
        assert_eq!(block_f64.len(), 100);

        // Values should be approximately equal (within f32 precision)
        for i in 0..block_f32.len() {
            let diff_re = (block_f32[i].re as f64 - block_f64[i].re).abs();
            let diff_im = (block_f32[i].im as f64 - block_f64[i].im).abs();
            assert!(diff_re < 1e-6, "Real part differs too much at {}: {} vs {}", i, block_f32[i].re, block_f64[i].re);
            assert!(diff_im < 1e-6, "Imag part differs too much at {}: {} vs {}", i, block_f32[i].im, block_f64[i].im);
        }
    }

    #[test]
    fn test_awgn_statistical_properties() {
        // Generate many samples to check statistical properties
        let mut awgn = AWGN::new_from_seed(1e6, 10000, 1.0, 777);
        let block = awgn.generate_block::<f64>();

        // Calculate mean (should be close to 0)
        let mut mean_i = 0.0;
        let mut mean_q = 0.0;
        for i in 0..block.len() {
            mean_i += block[i].re;
            mean_q += block[i].im;
        }
        mean_i /= block.len() as f64;
        mean_q /= block.len() as f64;

        // Mean should be close to 0 (within 3*sigma/sqrt(N) with high probability)
        let expected_mean_error = 3.0 * (0.5_f64.sqrt()) / (block.len() as f64).sqrt();
        assert!(mean_i.abs() < expected_mean_error, "Mean I = {}, expected < {}", mean_i, expected_mean_error);
        assert!(mean_q.abs() < expected_mean_error, "Mean Q = {}, expected < {}", mean_q, expected_mean_error);

        // Calculate variance (should be close to noise_power)
        let mut var_i = 0.0;
        let mut var_q = 0.0;
        for i in 0..block.len() {
            var_i += (block[i].re - mean_i).powi(2);
            var_q += (block[i].im - mean_q).powi(2);
        }
        var_i /= (block.len() - 1) as f64;
        var_q /= (block.len() - 1) as f64;

        // Each component should have variance H noise_power/2 = 0.5
        let expected_component_var = 0.5;
        let var_tolerance = 0.1; // Allow 10% error
        assert!((var_i - expected_component_var).abs() < var_tolerance,
            "Variance I = {}, expected {}", var_i, expected_component_var);
        assert!((var_q - expected_component_var).abs() < var_tolerance,
            "Variance Q = {}, expected {}", var_q, expected_component_var);

        // Total power should be close to noise_power
        let total_power = var_i + var_q;
        assert!((total_power - 1.0).abs() < 2.0 * var_tolerance,
            "Total power = {}, expected 1.0", total_power);
    }
}
