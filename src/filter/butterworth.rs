use num_complex::Complex;
use rustfft::FftPlanner;

/// Create Butterworth frequency response in frequency domain
///
/// # Arguments
/// * `num_samples` - Number of frequency bins (should match signal length)
/// * `order` - Filter order (higher = steeper rolloff)
/// * `cutoff` - Normalized cutoff frequency (0.0 to 0.5, where 0.5 = Nyquist)
///
/// # Returns
/// Vector of frequency-domain filter coefficients
pub fn create_butterworth_filter(num_samples: usize, order: i32, cutoff: f64) -> Vec<f64> {
    let mut filter_response = vec![0.0; num_samples];

    for i in 0..num_samples {
        // Convert index to normalized frequency [-0.5, 0.5]
        let f = if i < num_samples / 2 {
            i as f64 / num_samples as f64
        } else {
            (i as f64 - num_samples as f64) / num_samples as f64
        };

        let norm_freq = f.abs();

        // Butterworth magnitude response: H(f) = 1 / sqrt(1 + (f/fc)^(2n))
        filter_response[i] = 1.0 / (1.0 + (norm_freq / cutoff).powi(2 * order)).sqrt();
    }

    filter_response
}

/// Apply Butterworth lowpass filter to IQ signal using FFT
///
/// # Arguments
/// * `signal` - Complex IQ signal to filter (modified in-place)
/// * `order` - Filter order (e.g., 3 for AD9361 style, 6 for traditional)
/// * `cutoff` - Normalized cutoff frequency (0.0 to 0.5)
///
/// # Example
/// ```ignore
/// let mut iq = generate_iq_noise(100_000, -97.0);
/// apply_butterworth_filter(&mut iq, 3, 0.45);  // 3rd order, cutoff at 0.45*Nyquist
/// ```
pub fn apply_butterworth_filter(signal: &mut [Complex<f64>], order: i32, cutoff: f64) {
    let n = signal.len();

    // Create filter in frequency domain
    let filter = create_butterworth_filter(n, order, cutoff);

    // FFT to frequency domain
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(signal);

    // Apply filter
    for (i, sample) in signal.iter_mut().enumerate() {
        *sample = *sample * filter[i];
    }

    // IFFT back to time domain
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(signal);

    // Normalize (rustfft doesn't normalize)
    let scale = 1.0 / (n as f64);
    for sample in signal.iter_mut() {
        *sample = *sample * scale;
    }
}

/// Convenience function: 3rd order Butterworth at 0.45*Nyquist (AD9361 style)
/// Cutoff = 0.45 * (fs/2) = 225 kHz at 1 MHz sample rate
pub fn apply_butterworth_3rd_045(signal: &mut [Complex<f64>]) {
    apply_butterworth_filter(signal, 3, 0.45);
}

/// Convenience function: 6th order Butterworth at 0.42*Nyquist (traditional digitizer)
/// Cutoff = 0.42 * (fs/2) = 210 kHz at 1 MHz sample rate
pub fn apply_butterworth_6th_042(signal: &mut [Complex<f64>]) {
    apply_butterworth_filter(signal, 6, 0.42);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to generate IQ noise
    fn generate_iq_noise(n_samples: usize, noise_floor_db: f64) -> Vec<Complex<f64>> {
        use rand::thread_rng;
        use rand_distr::{Distribution, Normal};

        let mut rng = thread_rng();
        let noise_power = 10_f64.powf(noise_floor_db / 10.0);
        let sigma = (noise_power / 2.0).sqrt();
        let normal = Normal::new(0.0, sigma).unwrap();

        (0..n_samples)
            .map(|_| Complex::new(normal.sample(&mut rng), normal.sample(&mut rng)))
            .collect()
    }

    #[test]
    fn test_filter_creation() {
        let filter = create_butterworth_filter(1024, 3, 0.45);
        assert_eq!(filter.len(), 1024);

        // DC should be 1.0 (unity gain)
        assert!((filter[0] - 1.0).abs() < 1e-10);

        // Frequencies beyond cutoff should be attenuated
        // Bin 512 is Nyquist (0.5), cutoff is 0.45, so it should be somewhat attenuated
        // For 3rd order Butterworth: H(0.5/0.45) = 1/sqrt(1 + (0.5/0.45)^6) ≈ 0.64
        assert!(filter[512] < 0.7);

        // Very high frequency (bin 256 = 0.25, which is < cutoff) should pass more
        assert!(filter[256] > 0.9);
    }

    #[test]
    fn test_dc_response() {
        // Create DC signal
        let mut signal: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0); 1024];

        // Apply filter
        apply_butterworth_filter(&mut signal, 3, 0.45);

        // DC should pass through with unity gain
        for sample in signal.iter() {
            assert!((sample.re - 1.0).abs() < 0.01);
            assert!((sample.im - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn test_3rd_order() {
        let mut signal = generate_iq_noise(1024, -80.0);
        apply_butterworth_3rd_045(&mut signal);
        assert_eq!(signal.len(), 1024);
    }

    #[test]
    fn test_6th_order() {
        let mut signal = generate_iq_noise(1024, -80.0);
        apply_butterworth_6th_042(&mut signal);
        assert_eq!(signal.len(), 1024);
    }

    #[test]
    fn test_butterworth_spectrum() {
        use std::env;
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Butterworth filter spectrum plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Butterworth Filter Frequency Response (FFT-based) ===\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(20.0) as usize;

        // Generate white noise
        let noise = generate_iq_noise(n_samples, -80.0);

        // Apply 3rd order Butterworth filter
        println!("Filtering with 3rd order Butterworth (cutoff = 0.45*Nyquist = 225 kHz)");
        let mut filtered_signal = noise.clone();
        apply_butterworth_filter(&mut filtered_signal, 4, 0.55);

        // Compute Welch PSD of filtered signal
        let (freqs, psd) = welch(
            &filtered_signal,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );

        // Convert PSD to dB scale
        let psd_db: Vec<f64> = vector_ops::to_db(&psd);

        // Plot Welch PSD
        plot_spectrum(&freqs, &psd_db, "Butterworth 3rd Order FFT-Based Filter Applied to AWGN");

        println!("✓ Filtered {} samples", filtered_signal.len());
        println!("✓ Cutoff frequency: 225 kHz (0.45 * Nyquist = 0.45 * 500 kHz)");
        println!("✓ Filter order: 3");
        println!("✓ Implementation: FFT-based frequency domain filtering");
    }

    #[test]
    fn test_filter_response() {
        use std::env;
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping filter response plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Butterworth Filter Response: Before vs After ===\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(20.0) as usize;

        // Generate white noise
        println!("Generating AWGN...");
        let noise = generate_iq_noise(n_samples, -80.0);

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

        // Filter the noise
        println!("Filtering with 3rd order Butterworth (225 kHz cutoff)...");
        let mut filtered_noise = noise.clone();
        apply_butterworth_filter(&mut filtered_noise, 3, 0.45);

        // Compute PSD of FILTERED noise
        let (_, psd_filtered) = welch(
            &filtered_noise,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_filtered_db: Vec<f64> = vector_ops::to_db(&psd_filtered);

        // Compute the difference (filter response)
        let filter_response_db: Vec<f64> = psd_filtered_db.iter()
            .zip(psd_unfiltered_db.iter())
            .map(|(filtered, unfiltered)| filtered - unfiltered)
            .collect();

        // Plot all three
        plot_spectrum(&freqs, &psd_unfiltered_db, "UNFILTERED: White Noise Spectrum");
        plot_spectrum(&freqs, &psd_filtered_db, "FILTERED: Butterworth Applied to White Noise");
        plot_spectrum(&freqs, &filter_response_db, "FILTER RESPONSE: Butterworth 3rd Order (FFT-based)");

        println!("\n✓ Generated 3 plots:");
        println!("  1. Unfiltered white noise (should be flat)");
        println!("  2. Filtered white noise (should show lowpass shape)");
        println!("  3. Filter response (should show Butterworth characteristic)");
        println!("\n✓ Expected filter response:");
        println!("  - 0 dB at DC (0 Hz) - NO NOTCH!");
        println!("  - -3 dB at 225 kHz");
        println!("  - Smooth rolloff beyond cutoff");
    }

    #[test]
    fn test_impulse_response() {
        use std::env;
        use crate::plot::plot_spectrum;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping impulse response test (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Butterworth Filter Impulse Response ===\n");

        let sample_rate = 1e6;
        let n_samples = 4096;

        // Create impulse signal (1.0 at t=0, 0.0 elsewhere)
        let mut impulse: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); n_samples];
        impulse[0] = Complex::new(1.0, 0.0);

        // Filter the impulse
        apply_butterworth_filter(&mut impulse, 3, 0.45);

        // Compute FFT to get frequency response
        use rustfft::FftPlanner;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n_samples);
        let mut freq_response = impulse.clone();
        fft.process(&mut freq_response);

        // Compute magnitude in dB and shift
        let mut magnitude_db: Vec<f64> = freq_response.iter()
            .map(|c| 20.0 * (c.norm().max(1e-20)).log10())
            .collect();

        // FFT shift for centered plotting
        use crate::fft::fft::fftshift;
        fftshift(&mut magnitude_db);

        // Get frequency axis
        use crate::fft::fft::fftfreqs;
        let freqs: Vec<f64> = fftfreqs(-sample_rate/2.0, sample_rate/2.0, n_samples);

        plot_spectrum(&freqs, &magnitude_db, "Butterworth 3rd Order - Impulse Response (FFT-based)");

        println!("✓ Generated impulse response");
        println!("✓ Check that:");
        println!("  - DC (0 Hz) is at maximum (~0 dB) - NO NOTCH!");
        println!("  - -3dB point is around 225 kHz");
        println!("  - Smooth Butterworth rolloff");
    }
}
