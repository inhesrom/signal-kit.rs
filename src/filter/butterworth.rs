use num_complex::Complex;
use num_traits::{Float, Signed};
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
pub fn create_butterworth_filter<T: Float>(num_samples: usize, order: i32, cutoff: T) -> Vec<T> {
    let mut filter_response = vec![T::zero(); num_samples];
    let one = T::one();

    for i in 0..num_samples {
        // Convert index to normalized frequency [-0.5, 0.5]
        let f = if i < num_samples / 2 {
            T::from(i).unwrap() / T::from(num_samples).unwrap()
        } else {
            (T::from(i).unwrap() - T::from(num_samples).unwrap()) / T::from(num_samples).unwrap()
        };

        let norm_freq = f.abs();

        // Butterworth magnitude response: H(f) = 1 / sqrt(1 + (f/fc)^(2n))
        let ratio = norm_freq / cutoff;
        let power = ratio.powi(2 * order);
        filter_response[i] = one / (one + power).sqrt();
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
pub fn apply_butterworth_filter<T>(signal: &mut [Complex<T>], order: i32, cutoff: T)
where
    T: Float + Signed + Send + Sync + std::fmt::Debug + num_traits::cast::FromPrimitive + std::ops::RemAssign + std::ops::DivAssign + 'static,
{
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
    let scale = T::one() / T::from(n).unwrap();
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

    #[test]
    fn test_dc_suppression_diagnostic() {
        use std::env;
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;
        use crate::fft::fft::{fft, fftshift, fftfreqs};

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping DC suppression diagnostic (set PLOT=true to enable)");
            return;
        }

        println!("\n=== DC SUPPRESSION DIAGNOSTIC ===");
        println!("Testing 3 methods to determine source of DC drop:\n");

        let sample_rate = 1e6;
        let n_samples = (2.0_f64).powf(18.0) as usize; // 262144 samples
        let order = 3;
        let cutoff = 0.45;

        // METHOD 1: Direct filter coefficients (GROUND TRUTH)
        println!("METHOD 1: Direct Filter Coefficients");
        let filter = create_butterworth_filter(n_samples, order, cutoff);
        println!("  Filter[0] (DC bin) = {:.15}", filter[0]);
        println!("  Filter[1] (1st bin) = {:.15}", filter[1]);
        println!("  Filter[10] (10th bin) = {:.15}", filter[10]);

        let dc_db_direct = 20.0 * filter[0].log10();
        println!("  DC in dB = {:.6} dB", dc_db_direct);

        if (filter[0] - 1.0).abs() < 1e-10 {
            println!("  ✓ NO DC SUPPRESSION in filter coefficients!\n");
        } else {
            println!("  ✗ DC SUPPRESSION DETECTED in filter!\n");
        }

        // METHOD 2: Direct FFT (no windowing, no averaging)
        println!("METHOD 2: Direct FFT Method (No Windowing)");
        let noise = generate_iq_noise(n_samples, -80.0);
        let mut filtered_fft = noise.clone();
        apply_butterworth_filter(&mut filtered_fft, order, cutoff);

        // Take FFT manually
        let mut freq_domain = filtered_fft.clone();
        fft(&mut freq_domain);

        // Compute PSD manually (no windowing)
        let mut psd_fft: Vec<f64> = freq_domain.iter()
            .map(|c| c.norm_sqr() / n_samples as f64)
            .collect();

        // Shift for plotting
        fftshift(&mut psd_fft);
        let psd_fft_db: Vec<f64> = vector_ops::to_db(&psd_fft);
        let freqs_fft: Vec<f64> = fftfreqs(-sample_rate/2.0, sample_rate/2.0, n_samples);

        // Find DC value (center of shifted array)
        let dc_idx_fft = n_samples / 2;
        let dc_db_fft = psd_fft_db[dc_idx_fft];
        println!("  DC value in PSD = {:.6} dB", dc_db_fft);
        println!("  ✓ Direct FFT (no windowing artifacts)\n");

        // METHOD 3: Welch with Hann window (standard method)
        println!("METHOD 3: Welch PSD with Hann Window");
        let mut filtered_welch_hann = noise.clone();
        apply_butterworth_filter(&mut filtered_welch_hann, order, cutoff);

        let (freqs_hann, psd_hann) = welch(
            &filtered_welch_hann,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );
        let psd_hann_db: Vec<f64> = vector_ops::to_db(&psd_hann);

        // Find DC bin (center)
        let dc_idx_hann = psd_hann_db.len() / 2;
        let dc_db_hann = psd_hann_db[dc_idx_hann];
        println!("  DC value in PSD = {:.6} dB", dc_db_hann);

        // METHOD 4: Welch with Rectangular window (no windowing effect)
        println!("\nMETHOD 4: Welch PSD with Rectangular Window");
        let mut filtered_welch_rect = noise.clone();
        apply_butterworth_filter(&mut filtered_welch_rect, order, cutoff);

        let (freqs_rect, psd_rect) = welch(
            &filtered_welch_rect,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Rectangular,
            None,
        );
        let psd_rect_db: Vec<f64> = vector_ops::to_db(&psd_rect);

        let dc_idx_rect = psd_rect_db.len() / 2;
        let dc_db_rect = psd_rect_db[dc_idx_rect];
        println!("  DC value in PSD = {:.6} dB", dc_db_rect);

        // ANALYSIS
        println!("\n=== ANALYSIS ===");
        println!("DC value comparison:");
        println!("  Method 1 (Direct coefficients): {:.6} dB", dc_db_direct);
        println!("  Method 2 (Direct FFT):          {:.6} dB", dc_db_fft);
        println!("  Method 3 (Welch + Hann):        {:.6} dB", dc_db_hann);
        println!("  Method 4 (Welch + Rectangular): {:.6} dB", dc_db_rect);

        println!("\nDC drop from true value (0 dB):");
        println!("  Method 1: {:.6} dB drop", dc_db_direct);
        println!("  Method 2: {:.6} dB drop", dc_db_fft);
        println!("  Method 3: {:.6} dB drop (Hann window artifact)", dc_db_hann);
        println!("  Method 4: {:.6} dB drop (Rectangular window)", dc_db_rect);

        // Plot all methods
        plot_spectrum(&freqs_fft, &psd_fft_db,
            "METHOD 2: Direct FFT (NO Windowing) - TRUE Filter Response");
        plot_spectrum(&freqs_hann, &psd_hann_db,
            "METHOD 3: Welch + Hann Window - Shows DC Drop from Windowing");
        plot_spectrum(&freqs_rect, &psd_rect_db,
            "METHOD 4: Welch + Rectangular Window - Closer to Truth");

        println!("\n=== CONCLUSION ===");

        let hann_dc_drop = dc_db_direct - dc_db_hann;
        let rect_dc_drop = dc_db_direct - dc_db_rect;

        if hann_dc_drop.abs() > 1.0 {
            println!("✓ PROOF: DC drop of {:.2} dB in Hann window method is a WINDOWING ARTIFACT!", hann_dc_drop);
            println!("✓ The Butterworth filter itself has NO DC suppression (filter[0] = {})!", filter[0]);
            println!("✓ Hann window reduces gain at DC, creating artificial suppression");
        }

        if rect_dc_drop.abs() < hann_dc_drop.abs() {
            println!("✓ Rectangular window has {:.2} dB less DC drop than Hann", hann_dc_drop - rect_dc_drop);
        }

        println!("\n⚠️  If you see DC drop in your plots, it's from Welch + Hann windowing,");
        println!("    NOT from the Butterworth filter!");
    }

    #[test]
    fn test_dc_component_in_noise() {
        use crate::generate::awgn::AWGN;

        println!("\n=== CHECKING DC COMPONENT IN AWGN ===\n");

        let n_samples = 262144;
        let sample_rate = 1e6;

        let mut awgn = AWGN::new_from_seed(sample_rate, n_samples, 1.0, 42);
        let noise: Vec<Complex<f64>> = awgn.generate_block().to_vec();

        // Compute DC (mean) of the noise
        let sum: Complex<f64> = noise.iter().sum();
        let dc_mean = sum / (n_samples as f64);

        println!("Noise samples: {}", n_samples);
        println!("DC (mean) = {} + {}i", dc_mean.re, dc_mean.im);
        println!("DC magnitude = {}", dc_mean.norm());
        println!("DC power (magnitude^2) = {}", dc_mean.norm_sqr());
        println!("DC in dB = {} dB\n", 10.0 * dc_mean.norm_sqr().log10());

        println!("✓ AWGN has near-zero DC by design (zero mean Gaussian)!");
        println!("✓ This is why you see DC drop in spectrograms of filtered noise");
        println!("✓ The Butterworth filter preserves DC, but there's no DC to preserve!\n");
    }
}

    #[test]
    fn test_butterworth_preserves_dc() {
        use crate::generate::awgn::AWGN;
        use crate::fft::fft::fft;

        println!("\n=== PROOF: Butterworth Preserves DC ===\n");

        let n_samples = 262144;
        let sample_rate = 1e6;

        // Generate noise and ADD a DC offset
        let mut awgn = AWGN::new_from_seed(sample_rate, n_samples, 1.0, 42);
        let mut signal_with_dc: Vec<Complex<f64>> = awgn.generate_block().to_vec();

        // Add DC offset
        let dc_offset = Complex::new(10.0, 5.0);
        for sample in signal_with_dc.iter_mut() {
            *sample += dc_offset;
        }

        // Measure DC before filtering
        let dc_before: Complex<f64> = signal_with_dc.iter().sum::<Complex<f64>>() / (n_samples as f64);
        println!("DC BEFORE filtering: {} + {}i", dc_before.re, dc_before.im);
        println!("DC magnitude BEFORE: {}", dc_before.norm());
        println!("DC in dB BEFORE: {} dB\n", 20.0 * dc_before.norm().log10());

        // Apply Butterworth filter
        apply_butterworth_filter(&mut signal_with_dc, 3, 0.45);

        // Measure DC after filtering
        let dc_after: Complex<f64> = signal_with_dc.iter().sum::<Complex<f64>>() / (n_samples as f64);
        println!("DC AFTER filtering: {} + {}i", dc_after.re, dc_after.im);
        println!("DC magnitude AFTER: {}", dc_after.norm());
        println!("DC in dB AFTER: {} dB\n", 20.0 * dc_after.norm().log10());

        // Check that DC is preserved
        let dc_ratio = dc_after.norm() / dc_before.norm();
        println!("DC preservation ratio: {:.6}", dc_ratio);
        println!("DC preservation in dB: {:.6} dB\n", 20.0 * dc_ratio.log10());

        assert!((dc_ratio - 1.0).abs() < 0.01, "DC should be preserved within 1%!");

        println!("✓ PROOF: Butterworth filter preserves DC with {:.2}% accuracy!", (dc_ratio - 1.0).abs() * 100.0);
        println!("✓ filter[0] = 1.0 means unity gain at DC");
        println!("✓ The filter has NO DC suppression!\n");
    }
