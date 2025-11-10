#![allow(dead_code)]

use num_complex::Complex;
use num_traits::{Float, Signed};
use num_traits::cast::FromPrimitive;
use std::fmt::Debug;
use std::ops::{RemAssign, DivAssign};
use crate::complex_vec::ComplexVec;
use crate::fft::fft::{fft, ifft};

/// Resample a signal using FFT-based interpolation (similar to scipy.signal.resample)
///
/// # Arguments
/// * `signal` - Input signal to resample
/// * `factor` - Resampling factor (> 1.0 for upsampling, < 1.0 for downsampling, 1.0 for no change)
///
/// # Returns
/// Resampled signal with length = round(input_length * factor)
pub fn resample<T>(signal: &ComplexVec<T>, factor: f64) -> ComplexVec<T>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    let input_len = signal.len();
    let output_len = ((input_len as f64) * factor).round() as usize;

    // Handle edge cases
    if output_len == 0 {
        return ComplexVec::new();
    }
    if output_len == input_len {
        return ComplexVec::from_vec(signal.iter().cloned().collect());
    }

    if output_len > input_len {
        upsample(signal, output_len)
    } else {
        downsample(signal, output_len)
    }
}

/// Upsample signal by zero-padding in frequency domain
fn upsample<T>(signal: &ComplexVec<T>, output_len: usize) -> ComplexVec<T>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    let input_len = signal.len();

    // Clone input data for FFT processing
    let mut freq_data: Vec<Complex<T>> = signal.iter().cloned().collect();

    // Forward FFT (already scaled by 1/N)
    fft(&mut freq_data[..]);

    // Create zero-padded frequency domain data
    let mut padded_freq = vec![Complex::new(T::zero(), T::zero()); output_len];

    if input_len % 2 == 0 {
        // Even length: split Nyquist bin
        let half_input = input_len / 2;
        let nyquist_bin = freq_data[half_input];

        // Copy DC and positive frequencies
        for i in 0..=half_input {
            padded_freq[i] = freq_data[i];
        }

        // Split Nyquist bin (multiply by 0.5 and place at both ends)
        padded_freq[half_input] = nyquist_bin * T::from(0.5).unwrap();
        padded_freq[output_len - half_input] = nyquist_bin * T::from(0.5).unwrap();

        // Copy negative frequencies
        for i in (half_input + 1)..input_len {
            let offset = input_len - i;
            padded_freq[output_len - offset] = freq_data[i];
        }
    } else {
        // Odd length: no Nyquist bin
        let half_input = (input_len + 1) / 2;

        // Copy DC and positive frequencies
        for i in 0..half_input {
            padded_freq[i] = freq_data[i];
        }

        // Copy negative frequencies
        for i in half_input..input_len {
            let offset = input_len - i;
            padded_freq[output_len - offset] = freq_data[i];
        }
    }

    // Inverse FFT (no additional scaling needed - fft() already scaled by 1/N)
    ifft(&mut padded_freq[..]);

    ComplexVec::from_vec(padded_freq)
}

/// Downsample signal by truncating in frequency domain
fn downsample<T>(signal: &ComplexVec<T>, output_len: usize) -> ComplexVec<T>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    let input_len = signal.len();

    // Clone input data for FFT processing
    let mut freq_data: Vec<Complex<T>> = signal.iter().cloned().collect();

    // Forward FFT (already scaled by 1/N)
    fft(&mut freq_data[..]);

    // Create truncated frequency domain data
    let mut truncated_freq = vec![Complex::new(T::zero(), T::zero()); output_len];

    if output_len % 2 == 0 {
        // Even output length: include Nyquist bin
        let half_output = output_len / 2;

        // Copy DC and positive frequencies
        for i in 0..=half_output {
            truncated_freq[i] = freq_data[i];
        }

        // Copy negative frequencies
        for i in 1..half_output {
            truncated_freq[output_len - i] = freq_data[input_len - i];
        }
    } else {
        // Odd output length: no Nyquist bin
        let half_output = (output_len + 1) / 2;

        // Copy DC and positive frequencies
        for i in 0..half_output {
            truncated_freq[i] = freq_data[i];
        }

        // Copy negative frequencies
        for i in half_output..output_len {
            let offset = output_len - i;
            truncated_freq[i] = freq_data[input_len - offset];
        }
    }

    // Inverse FFT (no additional scaling needed - fft() already scaled by 1/N)
    ifft(&mut truncated_freq[..]);

    ComplexVec::from_vec(truncated_freq)
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_upsample_by_2() {
        // Create a simple DC signal
        let input: Vec<Complex<f64>> = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];
        let signal = ComplexVec::from_vec(input.clone());

        // Upsample by factor of 2
        let resampled = resample(&signal, 2.0);

        // Check output length
        assert_eq!(resampled.len(), 8);

        // For DC signal, all values should remain approximately 1.0
        for i in 0..resampled.len() {
            assert!((resampled[i].re - 1.0).abs() < 1e-10,
                "Sample {} has value {}, expected 1.0", i, resampled[i].re);
            assert!(resampled[i].im.abs() < 1e-10,
                "Sample {} has imaginary part {}, expected 0.0", i, resampled[i].im);
        }
    }

    #[test]
    fn test_downsample_by_2() {
        // Create a simple DC signal
        let input: Vec<Complex<f64>> = vec![
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(2.0, 0.0),
        ];
        let signal = ComplexVec::from_vec(input.clone());

        // Downsample by factor of 0.5
        let resampled = resample(&signal, 0.5);

        // Check output length
        assert_eq!(resampled.len(), 4);

        // For DC signal, all values should remain approximately 2.0
        for i in 0..resampled.len() {
            assert!((resampled[i].re - 2.0).abs() < 1e-10,
                "Sample {} has value {}, expected 2.0", i, resampled[i].re);
            assert!(resampled[i].im.abs() < 1e-10,
                "Sample {} has imaginary part {}, expected 0.0", i, resampled[i].im);
        }
    }

    #[test]
    fn test_bpsk_upsample_spectrum() {
        use std::env;
        use crate::generate::psk_carrier::PskCarrier;
        use crate::mod_type::ModType;
        use crate::fft::fft::{fft, fftshift, fftfreqs};
        use crate::vector_ops;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping BPSK upsample spectrum plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== BPSK Upsample Spectrum Test ===");

        // Create BPSK carrier parameters
        let sample_rate_hz = 10e6_f64;
        let symbol_rate_hz = 1e6_f64;
        let rolloff_factor = 0.35_f64;
        let block_size = 4096;
        let filter_taps = 51;
        let upsample_rate = 1.5;

        // Generate BPSK carrier signal
        let mut carrier = PskCarrier::new(
            sample_rate_hz,
            symbol_rate_hz,
            ModType::_BPSK,
            rolloff_factor,
            block_size,
            filter_taps,
            Some(42), // seed for reproducibility
        );

        let signal = carrier.generate_block();
        println!("Original signal length: {}", signal.len());

        let upsampled = resample(&signal, upsample_rate);
        println!("Upsampled signal length: {}", upsampled.len());

        // Compute FFT of upsampled signal
        let mut freq_data: Vec<Complex<f64>> = upsampled.iter().cloned().collect();
        fft(&mut freq_data[..]);
        let mut upsampled_fft = ComplexVec::from_vec(freq_data);
        let mut spectrum_db: Vec<f64> = vector_ops::to_db(&upsampled_fft.abs());

        // Apply fftshift and compute frequency axis
        fftshift::<f64>(&mut spectrum_db);
        let upsampled_sample_rate = sample_rate_hz * upsample_rate;
        let freqs: Vec<f64> = fftfreqs::<f64>(
            -upsampled_sample_rate / 2.0,
            upsampled_sample_rate / 2.0,
            spectrum_db.len(),
        );

        println!("FFT size: {}", spectrum_db.len());
        println!("Upsampled sample rate: {:.2} Hz", upsampled_sample_rate);

        // Plot spectrum
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &spectrum_db, "BPSK Signal Spectrum (Upsampled 4x)");
    }
}
