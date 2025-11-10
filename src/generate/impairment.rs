//! Channel impairments for realistic signal simulation
//!
//! This module provides functions to apply various channel impairments that occur
//! in real-world RF systems, including digitizer droop, frequency-dependent amplitude
//! variation, and other effects.
//!
//! # Digitizer Droop
//!
//! Real digitizers have anti-aliasing filters that create rolloff (droop) at the edges
//! of the captured bandwidth. This module provides functions to simulate this realistic
//! behavior using Butterworth filters.
//!
//! ## Example: Apply AD9361-style digitizer droop
//!
//! ```
//! use signal_kit::generate::impairment::apply_digitizer_droop_ad9361;
//! use num_complex::Complex;
//!
//! // Generate some IQ samples (e.g., from a carrier or AWGN)
//! let mut iq_samples = vec![Complex::new(1.0, 0.5); 10000];
//!
//! // Apply realistic digitizer droop (3rd order Butterworth, 45% of Nyquist)
//! apply_digitizer_droop_ad9361(&mut iq_samples);
//! // Now iq_samples has realistic band-edge rolloff like an AD9361 SDR
//! ```
//!
//! ## Example: Apply custom digitizer droop
//!
//! ```
//! use signal_kit::generate::impairment::apply_digitizer_droop;
//! use num_complex::Complex;
//!
//! let mut iq_samples = vec![Complex::new(1.0, 0.5); 10000];
//!
//! // Custom droop: 4th order filter, cutoff at 48% of Nyquist
//! // Works with any sample rate since frequency is normalized
//! apply_digitizer_droop(&mut iq_samples, 4, 0.48);
//! ```
//!
//! ## Example: Different digitizer profiles
//!
//! ```
//! use signal_kit::generate::{apply_digitizer_droop_ad9361, apply_digitizer_droop_traditional};
//! use num_complex::Complex;
//!
//! let mut sdr_signal = vec![Complex::new(1.0, 0.5); 10000];
//! apply_digitizer_droop_ad9361(&mut sdr_signal);  // Gentle rolloff
//!
//! let mut digitizer_signal = vec![Complex::new(1.0, 0.5); 10000];
//! apply_digitizer_droop_traditional(&mut digitizer_signal);  // Steeper rolloff
//! ```

#![allow(dead_code)]

use num_traits::Float;
use num_complex::Complex;
use std::f64::consts::PI;
use crate::filter::butterworth::apply_butterworth_filter;

pub fn frequency_dependent_amplitude_variation<T: Float>(num_samples: usize, amplitude_db: T, cycles: T, phase_offset: T,
) -> Vec<T> {
    let pi = T::from(PI).unwrap();
    (0..num_samples)
        .into_iter()
        .map(|idx| {
            amplitude_db
                * (T::from(2.0).unwrap() * pi * T::from(idx).unwrap()
                    / T::from(num_samples).unwrap()
                    * cycles
                    + phase_offset)
                    .sin()
        })
            .collect()
    }

/// Apply realistic digitizer channel droop using Butterworth anti-aliasing filter
///
/// Simulates the frequency response of a real digitizer's analog anti-aliasing filter.
/// Creates smooth rolloff at the band edges, just like hardware ADCs (e.g., AD9361, etc.)
///
/// # Arguments
/// * `signal` - Complex IQ signal to apply droop to (modified in-place)
/// * `order` - Filter order controlling rolloff steepness
///   - 3-4: Gentle rolloff (realistic for SDR frontends like AD9361)
///   - 5-6: Medium rolloff (typical digitizers)
///   - 7-8: Steep rolloff (high-end digitizers)
/// * `cutoff` - Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist)
///   - 0.40-0.45: Typical for SDR receivers
///   - 0.42-0.48: Typical for digitizers
///   - Lower values = more droop at band edges
///
/// # Example
/// ```ignore
/// use signal_kit::generate::impairment::apply_digitizer_droop;
/// 
/// let mut iq_samples = vec![Complex::new(1.0, 0.5); 10000];
/// 
/// // Apply AD9361-style droop (3rd order, 45% of Nyquist)
/// apply_digitizer_droop(&mut iq_samples, 3, 0.45);
/// 
/// // Apply traditional digitizer droop (6th order, 42% of Nyquist)
/// apply_digitizer_droop(&mut iq_samples, 6, 0.42);
/// ```
///
/// # Notes
/// - Works with any sample rate (uses normalized frequency)
/// - Applied in frequency domain via FFT for efficiency
/// - Creates smooth, realistic rolloff at band edges
/// - Typical digitizers have -3dB point around 0.4-0.48 × Nyquist frequency
pub fn apply_digitizer_droop(signal: &mut [Complex<f64>], order: i32, cutoff: f64) {
    apply_butterworth_filter(signal, order, cutoff);
}

/// Apply AD9361-style digitizer droop
///
/// Applies a 3rd order Butterworth filter with cutoff at 45% of Nyquist frequency,
/// mimicking the anti-aliasing characteristics of popular SDR receivers.
///
/// At 1 MHz sample rate, this creates -3dB rolloff at approximately 225 kHz.
///
/// # Arguments
/// * `signal` - Complex IQ signal to apply droop to (modified in-place)
pub fn apply_digitizer_droop_ad9361(signal: &mut [Complex<f64>]) {
    apply_digitizer_droop(signal, 3, 0.45);
}

/// Apply traditional digitizer droop
///
/// Applies a 6th order Butterworth filter with cutoff at 42% of Nyquist frequency,
/// mimicking typical high-quality digitizer anti-aliasing filters.
///
/// At 1 MHz sample rate, this creates -3dB rolloff at approximately 210 kHz.
///
/// # Arguments
/// * `signal` - Complex IQ signal to apply droop to (modified in-place)
pub fn apply_digitizer_droop_traditional(signal: &mut [Complex<f64>]) {
    apply_digitizer_droop(signal, 6, 0.42);
}

#[cfg(test)]
mod tests {
    use crate::{ComplexVec, generate::awgn::AWGN, generate::impairment::frequency_dependent_amplitude_variation, vector_ops::{add, to_linear}, fft::fft, spectrum::welch::welch, spectrum::window::WindowType, plot::plot_spectrum, vector_ops};
    use rand::{Rng, SeedableRng, rngs::StdRng};
    use num_complex::Complex;
    use super::{apply_digitizer_droop, apply_digitizer_droop_ad9361, apply_digitizer_droop_traditional};

    #[test]
    fn test_freq_ampl_variation() {
        use std::env;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping frequency-dependent amplitude variation plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6_f64;
        let num_samples = (2.0_f64).powf(20.0) as usize;  // Use power of 2 for better FFT performance
        // Generate three different frequency variation patterns
        let freq_var_1 = frequency_dependent_amplitude_variation(num_samples, 1.0, 2.0, 0.0);
        let freq_var_2 = frequency_dependent_amplitude_variation(num_samples, 0.6, 5.0, 0.0);
        let freq_var_3 = frequency_dependent_amplitude_variation(num_samples, 0.4, 1.0, 1.5);
        // Combine all variations
        let total_variation_db = add(&add(&freq_var_1, &freq_var_2), &freq_var_3);
        let total_variation_lin = to_linear(&total_variation_db);
        // Generate white noise
        let mut awgn = AWGN::new_from_seed(sample_rate, num_samples, 1.0, 0);
        let mut noise: ComplexVec<f32> = awgn.generate_block();
        // Apply frequency-dependent amplitude variation in frequency domain
        fft::fft(&mut noise);
        for (i, sample) in noise.iter_mut().enumerate() {
            *sample *= total_variation_lin[i] as f32;
        }
        fft::ifft(&mut noise);
        // Compute Welch PSD
        let (freqs, psd) = welch::<f32>(
            &noise,
            sample_rate as f32,
            2048,                    // 1024-point segments
            None,                    // 50% overlap (default)
            None,                    // No zero-padding (default)
            WindowType::Hann,        // Hann window (standard)
            None,                    // Mean averaging (default)
        );
        // Convert PSD to dB scale
        let psd_db: Vec<f32> = vector_ops::to_db(&psd);
        // Plot Welch PSD
        plot_spectrum(&freqs, &psd_db, "Frequency-Dependent Amplitude Variation Applied to AWGN");
    }

    #[test]
    fn test_random_freq_ampl_variation() {
        use std::env;
        use std::f64::consts::PI;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping frequency-dependent amplitude variation plot (set PLOT=true to enable)");
            return;
        }

        let seed = 2; // or any u64
        let mut rng = StdRng::seed_from_u64(seed);
        let sample_rate = 1e6_f64;
        let num_samples = (2.0_f64).powf(20.0) as usize;  // Use power of 2 for better FFT performance

        let mut awgn = AWGN::new_from_seed(sample_rate, num_samples, 1.0, 0);
        let mut noise: ComplexVec<f32> = awgn.generate_block();
        fft::fft(&mut noise);

        // Apply frequency-dependent amplitude variation in frequency domain
        let num_variations = rand::thread_rng().gen_range(1..=5);
        let mut freq_var_total: Vec<f32> = vec![0.0; num_samples];
        for _ in 0..num_variations {
            let amplitude_variation: f32 = rng.gen_range(0.0..=1.0);
            let cycles: f32 = rng.gen_range(1.0..=6.0);
            let phase_offset: f32 = rng.gen_range(0.0..=2.0 * PI as f32);
            
            let freq_variation = frequency_dependent_amplitude_variation::<f32>(num_samples, amplitude_variation, cycles, phase_offset);
            freq_var_total = add(&freq_var_total, &freq_variation);
        }
        let total_variation_lin = to_linear(&freq_var_total);

        // Apply frequency-dependent amplitude variation in frequency domain
        for (i, sample) in noise.iter_mut().enumerate() {
            *sample *= total_variation_lin[i] as f32;
        }
        fft::ifft(&mut noise);
        // Compute Welch PSD
        let (freqs, psd) = welch::<f32>(
            &noise,
            sample_rate as f32,
            2048,                    // 1024-point segments
            None,                    // 50% overlap (default)
            None,                    // No zero-padding (default)
            WindowType::Hann,        // Hann window (standard)
            None,                    // Mean averaging (default)
        );
        // Convert PSD to dB scale
        let psd_db: Vec<f32> = vector_ops::to_db(&psd);
        // Plot Welch PSD
        plot_spectrum(&freqs, &psd_db, "Frequency-Dependent Amplitude Variation Applied to AWGN");
    }

    #[test]
    fn test_digitizer_droop() {
        use std::env;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping digitizer droop plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Digitizer Channel Droop Impairment ===\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(20.0) as usize;

        // Generate white noise
        println!("Generating AWGN...");
        let mut awgn = AWGN::new_from_seed(sample_rate, n_samples, 1.0, 0);
        let noise: Vec<Complex<f64>> = awgn.generate_block::<f64>().to_vec();

        // Compute PSD of UNFILTERED noise
        let (freqs, psd_unfiltered) = welch(
            &noise,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_unfiltered_db: Vec<f64> = vector_ops::to_db(&psd_unfiltered);

        // Apply AD9361-style digitizer droop
        println!("Applying AD9361-style digitizer droop (3rd order, 45% Nyquist)...");
        let mut noise_ad9361 = noise.clone();
        apply_digitizer_droop_ad9361(&mut noise_ad9361);

        let (_, psd_ad9361) = welch(
            &noise_ad9361,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_ad9361_db: Vec<f64> = vector_ops::to_db(&psd_ad9361);

        // Apply traditional digitizer droop
        println!("Applying traditional digitizer droop (6th order, 42% Nyquist)...");
        let mut noise_traditional = noise.clone();
        apply_digitizer_droop_traditional(&mut noise_traditional);

        let (_, psd_traditional) = welch(
            &noise_traditional,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_traditional_db: Vec<f64> = vector_ops::to_db(&psd_traditional);

        // Apply custom digitizer droop (4th order, 48% Nyquist)
        println!("Applying custom digitizer droop (4th order, 48% Nyquist)...");
        let mut noise_custom = noise.clone();
        apply_digitizer_droop(&mut noise_custom, 4, 0.48);

        let (_, psd_custom) = welch(
            &noise_custom,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_custom_db: Vec<f64> = vector_ops::to_db(&psd_custom);

        // Plot all four
        plot_spectrum(&freqs, &psd_unfiltered_db, "UNFILTERED: White Noise (No Droop)");
        plot_spectrum(&freqs, &psd_ad9361_db, "DIGITIZER DROOP: AD9361 Style (3rd order, 45% Nyquist)");
        plot_spectrum(&freqs, &psd_traditional_db, "DIGITIZER DROOP: Traditional (6th order, 42% Nyquist)");
        plot_spectrum(&freqs, &psd_custom_db, "DIGITIZER DROOP: Custom (4th order, 48% Nyquist)");

        println!("\n✓ Generated 4 plots showing different digitizer droop profiles:");
        println!("  1. Unfiltered white noise (flat response)");
        println!("  2. AD9361 style: Gentle rolloff, -3dB @ 225 kHz");
        println!("  3. Traditional: Steeper rolloff, -3dB @ 210 kHz");
        println!("  4. Custom: Medium rolloff, -3dB @ 240 kHz");
        println!("\n✓ All droop profiles work with any sample rate (normalized frequency)");
        println!("✓ Creates realistic anti-aliasing filter behavior at band edges");
    }

    #[test]
    fn test_digitizer_droop_comparison() {
        use std::env;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping digitizer droop comparison plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Digitizer Droop: Order Comparison ===\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(20.0) as usize;

        // Generate white noise
        let mut awgn = AWGN::new_from_seed(sample_rate, n_samples, 1.0, 42);
        let noise: Vec<Complex<f64>> = awgn.generate_block::<f64>().to_vec();

        // Test different filter orders with same cutoff
        let cutoff = 0.45;
        let orders = vec![2, 3, 4, 6, 8];

        for order in orders {
            let mut filtered = noise.clone();
            apply_digitizer_droop(&mut filtered, order, cutoff);

            let (freqs, psd) = welch(
                &filtered,
                sample_rate,
                2048,
                None,
                None,
                WindowType::Hann,
                None,
            );
            let psd_db: Vec<f64> = vector_ops::to_db(&psd);

            plot_spectrum(
                &freqs,
                &psd_db,
                &format!("Digitizer Droop: Order {} (cutoff = {}×Nyquist)", order, cutoff)
            );

            println!("✓ Order {}: Rolloff steepness increases with order", order);
        }

        println!("\n✓ Higher order = steeper rolloff (more brick-wall like)");
        println!("✓ Lower order = gentler rolloff (more realistic for SDR hardware)");
    }

    #[test]
    fn test_digitizer_droop_on_carrier() {
        use std::env;
        use crate::generate::carrier::Carrier;
        use crate::ModType;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping digitizer droop on carrier plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Digitizer Droop Applied to QPSK Carrier ===\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(20.0) as usize;

        // Create a QPSK carrier
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,        // 10% bandwidth
            0.0,        // Centered
            15.0,       // 15 dB SNR
            0.35,       // RRC rolloff
            sample_rate,
            Some(42),
        );

        // Generate clean carrier signal
        println!("Generating QPSK carrier (clean)...");
        let clean_signal: Vec<Complex<f64>> = carrier.generate(n_samples).to_vec();

        // Compute PSD of clean signal
        let (freqs, psd_clean) = welch(
            &clean_signal,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_clean_db: Vec<f64> = vector_ops::to_db(&psd_clean);

        // Apply AD9361-style digitizer droop
        println!("Applying AD9361-style digitizer droop...");
        let mut drooped_signal = clean_signal.clone();
        apply_digitizer_droop_ad9361(&mut drooped_signal);

        let (_, psd_drooped) = welch(
            &drooped_signal,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_drooped_db: Vec<f64> = vector_ops::to_db(&psd_drooped);

        // Plot both
        plot_spectrum(&freqs, &psd_clean_db, "QPSK Carrier: Clean (No Digitizer Droop)");
        plot_spectrum(&freqs, &psd_drooped_db, "QPSK Carrier: With AD9361 Digitizer Droop");

        println!("\n✓ Generated 2 plots:");
        println!("  1. Clean QPSK carrier with RRC pulse shaping");
        println!("  2. Same carrier with realistic AD9361 digitizer droop");
        println!("\n✓ Notice the band-edge rolloff in the second plot");
        println!("✓ This simulates what you'd see from a real SDR or digitizer");
        println!("✓ Works with any sample rate - frequency is normalized");
    }
}
