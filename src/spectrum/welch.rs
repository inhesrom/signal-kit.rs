#![allow(dead_code)]

use num_complex::Complex;
use num_traits::Float;
use crate::spectrum::window::{WindowType, generate_window, window_energy};
use crate::fft::fft::{fft, fftfreqs, fftshift};

/// Averaging method for combining periodograms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingMethod {
    /// Mean averaging (standard Welch method)
    Mean,
    /// Median averaging (robust to outliers)
    Median,
    /// Maximum across segments (peak power analysis)
    Max,
    /// Minimum across segments (noise floor estimation)
    Min,
}

/// Welch's method for power spectral density estimation
///
/// Estimates the power spectral density by dividing the signal into overlapping
/// segments, computing modified periodograms, and averaging them.
///
/// # Arguments
/// * `signal` - Input signal (complex samples)
/// * `sample_rate` - Sampling frequency
/// * `nperseg` - Segment length for FFT
/// * `noverlap` - Number of samples to overlap between segments (default: nperseg/2)
/// * `nfft` - FFT length for zero-padding (default: nperseg)
/// * `window` - Window function type (default: Hann)
/// * `averaging` - Averaging method (default: Mean)
///
/// # Returns
/// Tuple of (frequencies, psd) where frequencies are in Hz and psd is in power/Hz.
/// Returns two-sided spectrum from -fs/2 to +fs/2 for complex input signals.
pub fn welch<T>(
    signal: &[Complex<T>],
    sample_rate: T,
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
    window: WindowType,
    averaging: Option<AveragingMethod>,
) -> (Vec<T>, Vec<T>)
where
    T: Float + std::fmt::Debug + std::ops::RemAssign + std::ops::DivAssign + Send + Sync + num_traits::FromPrimitive + num_traits::Signed + 'static,
{
    // Compute defaults
    let (overlap, fft_len) = compute_defaults(nperseg, noverlap, nfft);
    let avg_method = averaging.unwrap_or(AveragingMethod::Mean);

    // Generate window
    let window_vec = generate_window(window, nperseg);

    // Extract overlapping segments
    let segments = extract_segments(signal, nperseg, overlap);

    if segments.is_empty() {
        // Return empty result if no segments
        return (vec![], vec![]);
    }

    // Compute PSD for each segment
    let psds: Vec<Vec<T>> = segments
        .iter()
        .map(|seg| compute_segment_psd(seg, &window_vec, fft_len))
        .collect();

    // Average the PSDs
    let averaged_psd = average_spectra(psds, avg_method);

    // Normalize for proper PSD scaling
    // Note: normalize_psd needs nperseg (segment length), not fft_len
    let mut normalized_psd = normalize_psd(averaged_psd, &window_vec, sample_rate, fft_len);

    // Shift PSD to center DC component
    fftshift(&mut normalized_psd);

    // Generate two-sided frequency axis from -fs/2 to +fs/2
    let two = T::from(2.0).unwrap();
    let freqs = fftfreqs(-sample_rate / two, sample_rate / two, fft_len);

    (freqs, normalized_psd)
}

/// Compute default values for noverlap and nfft
fn compute_defaults(
    nperseg: usize,
    noverlap: Option<usize>,
    nfft: Option<usize>,
) -> (usize, usize) {
    let overlap = noverlap.unwrap_or(nperseg / 2);
    let fft_len = nfft.unwrap_or(nperseg);
    (overlap, fft_len)
}

/// Extract overlapping segments from the signal
fn extract_segments<T: Float>(
    signal: &[Complex<T>],
    nperseg: usize,
    noverlap: usize,
) -> Vec<Vec<Complex<T>>> {
    let step = nperseg - noverlap;
    let mut segments = Vec::new();

    let mut start = 0;
    while start + nperseg <= signal.len() {
        let segment = signal[start..start + nperseg].to_vec();
        segments.push(segment);
        start += step;
    }

    segments
}

/// Apply window to a segment and compute its power spectral density
fn compute_segment_psd<T>(
    segment: &[Complex<T>],
    window: &[T],
    nfft: usize,
) -> Vec<T>
where
    T: Float + std::fmt::Debug + std::ops::RemAssign + std::ops::DivAssign + Send + Sync + num_traits::FromPrimitive + num_traits::Signed + 'static,
{
    // Apply window
    let windowed = apply_window(segment, window);

    // Pad or truncate to nfft length
    let mut fft_input = prepare_fft_input(windowed, nfft);

    // Compute FFT
    fft(&mut fft_input);

    // Compute power: |FFT|^2
    fft_input.iter().map(|c| c.norm_sqr()).collect()
}

/// Apply window function to a segment
fn apply_window<T: Float>(segment: &[Complex<T>], window: &[T]) -> Vec<Complex<T>> {
    segment
        .iter()
        .zip(window.iter())
        .map(|(s, &w)| Complex::new(s.re * w, s.im * w))
        .collect()
}

/// Prepare FFT input by padding or truncating
fn prepare_fft_input<T: Float>(windowed: Vec<Complex<T>>, nfft: usize) -> Vec<Complex<T>> {
    let mut fft_input = windowed;

    if nfft > fft_input.len() {
        // Zero-pad
        fft_input.resize(nfft, Complex::new(T::zero(), T::zero()));
    } else if nfft < fft_input.len() {
        // Truncate
        fft_input.truncate(nfft);
    }

    fft_input
}

/// Average power spectra using the specified method
fn average_spectra<T: Float>(spectra: Vec<Vec<T>>, method: AveragingMethod) -> Vec<T> {
    match method {
        AveragingMethod::Mean => compute_mean(&spectra),
        AveragingMethod::Median => compute_median(&spectra),
        AveragingMethod::Max => compute_max(&spectra),
        AveragingMethod::Min => compute_min(&spectra),
    }
}

/// Compute mean across all spectra
fn compute_mean<T: Float>(spectra: &[Vec<T>]) -> Vec<T> {
    if spectra.is_empty() {
        return vec![];
    }

    let n_bins = spectra[0].len();
    let n_spectra = T::from(spectra.len()).unwrap();

    (0..n_bins)
        .map(|i| {
            let sum = spectra.iter().fold(T::zero(), |acc, spec| acc + spec[i]);
            sum / n_spectra
        })
        .collect()
}

/// Compute median across all spectra
fn compute_median<T: Float>(spectra: &[Vec<T>]) -> Vec<T> {
    if spectra.is_empty() {
        return vec![];
    }

    let n_bins = spectra[0].len();

    (0..n_bins)
        .map(|i| {
            let mut values: Vec<T> = spectra.iter().map(|spec| spec[i]).collect();
            median_of_slice(&mut values)
        })
        .collect()
}

/// Compute maximum across all spectra
fn compute_max<T: Float>(spectra: &[Vec<T>]) -> Vec<T> {
    if spectra.is_empty() {
        return vec![];
    }

    let n_bins = spectra[0].len();

    (0..n_bins)
        .map(|i| {
            spectra
                .iter()
                .map(|spec| spec[i])
                .fold(T::neg_infinity(), |acc, val| {
                    if val > acc { val } else { acc }
                })
        })
        .collect()
}

/// Compute minimum across all spectra
fn compute_min<T: Float>(spectra: &[Vec<T>]) -> Vec<T> {
    if spectra.is_empty() {
        return vec![];
    }

    let n_bins = spectra[0].len();

    (0..n_bins)
        .map(|i| {
            spectra
                .iter()
                .map(|spec| spec[i])
                .fold(T::infinity(), |acc, val| {
                    if val < acc { val } else { acc }
                })
        })
        .collect()
}

/// Compute median of a slice (sorts in place)
fn median_of_slice<T: Float>(values: &mut [T]) -> T {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = values.len();

    if len % 2 == 0 {
        // Even length: average of middle two
        let mid = len / 2;
        (values[mid - 1] + values[mid]) / T::from(2.0).unwrap()
    } else {
        // Odd length: middle value
        values[len / 2]
    }
}

/// Normalize PSD for proper scaling
fn normalize_psd<T: Float>(
    mut psd: Vec<T>,
    window: &[T],
    sample_rate: T,
    nfft: usize,
) -> Vec<T> {
    // Scaling factor: nfft^2 / (fs * sum(window^2))
    // The nfft^2 accounts for FFT magnitude scaling
    let window_power = window_energy(window);
    let nfft_float = T::from(nfft).unwrap();
    let scale = (nfft_float * nfft_float) / (sample_rate * window_power);

    // Apply scaling
    for val in psd.iter_mut() {
        *val = *val * scale;
    }

    psd
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::generate::cw::CW;
    use crate::generate::awgn::AWGN;
    use crate::generate::fsk_carrier::FskCarrier;
    use crate::generate::psk_carrier::PskCarrier;
    use crate::mod_type::ModType;

    #[test]
    fn test_welch_basic() {
        // Test with a simple CW tone
        let sample_rate = 1e6;
        let freq = 1e5; // 100 kHz tone
        let block_size = 4096;

        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block();

        let nperseg = 512;
        let (freqs, psd) = welch(
            &signal.iter().cloned().collect::<Vec<_>>(),
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        // Should have positive frequencies
        assert!(freqs.len() > 0);
        assert!(psd.len() == freqs.len());

        // Find peak
        let (peak_idx, peak_power) = psd
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let peak_freq = freqs[peak_idx];

        // Peak should be near the CW frequency
        assert!(
            (peak_freq - freq).abs() < 5e3,
            "Peak at {} Hz, expected near {} Hz",
            peak_freq,
            freq
        );

        // Peak power should be significant
        assert!(*peak_power > 0.0);
    }

    #[test]
    fn test_welch_cw_tone() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch CW tone plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: CW Tone Test ===");

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64; // 200 kHz
        let block_size = 8192;

        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block();

        let nperseg = 1024;
        let (freqs, psd) = welch(
            &signal.iter().cloned().collect::<Vec<_>>(),
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("Sample rate: {} Hz", sample_rate);
        println!("CW frequency: {} Hz", freq);
        println!("Segment length: {}", nperseg);
        println!("Number of frequency bins: {}", freqs.len());

        // Convert to dB
        use crate::vector_ops;
        let psd_db = vector_ops::to_db(&psd);

        // Plot
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &psd_db, "Welch PSD: CW Tone at 200 kHz");
    }

    #[test]
    fn test_welch_two_tones() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch two tones plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: Two Tones Test ===");

        let sample_rate = 1e6_f64;
        let freq1 = 1.5e5_f64; // 150 kHz
        let freq2 = 3e5_f64;   // 300 kHz
        let block_size = 8192;

        // Generate two CW tones
        let mut cw1 = CW::new(freq1, sample_rate, block_size);
        let mut cw2 = CW::new(freq2, sample_rate, block_size);
        let signal1 = cw1.generate_block::<f64>();
        let signal2 = cw2.generate_block::<f64>();

        // Add them together
        let mut combined: Vec<Complex<f64>> = Vec::with_capacity(block_size);
        for i in 0..block_size {
            combined.push(signal1[i] + signal2[i]);
        }

        let nperseg = 1024;
        let (freqs, psd) = welch(
            &combined,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("Sample rate: {} Hz", sample_rate);
        println!("Tone 1 frequency: {} Hz", freq1);
        println!("Tone 2 frequency: {} Hz", freq2);
        println!("Segment length: {}", nperseg);

        // Convert to dB
        use crate::vector_ops;
        let psd_db = vector_ops::to_db(&psd);

        // Plot
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &psd_db, "Welch PSD: Two Tones (150 kHz + 300 kHz)");
    }

    #[test]
    fn test_welch_awgn() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch AWGN plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: AWGN Test ===");

        let sample_rate = 1e6_f64;
        let block_size = 16384;
        let noise_power = 1.0;

        let mut awgn = AWGN::new_from_seed(sample_rate, block_size, noise_power, 42);
        let signal = awgn.generate_block();

        let nperseg = 2048;
        let (freqs, psd) = welch(
            &signal.iter().cloned().collect::<Vec<_>>(),
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("Sample rate: {} Hz", sample_rate);
        println!("Noise power: {}", noise_power);
        println!("Segment length: {}", nperseg);

        // Convert to dB
        use crate::vector_ops;
        let psd_db = vector_ops::to_db(&psd);

        // Plot
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &psd_db, "Welch PSD: White Gaussian Noise");
    }

    #[test]
    fn test_welch_fsk_signal() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch FSK signal plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: FSK Signal Test ===");

        let sample_rate = 1e6_f64;
        let symbol_rate = 1e5_f64;
        let carrier_freq = 2.5e5_f64;
        let deviation = 5e4_f64;
        let block_size = 16384;

        // Generate FSK signal
        let mut fsk: FskCarrier<f64> = FskCarrier::new(
            sample_rate,
            symbol_rate,
            carrier_freq,
            deviation,
            block_size,
            Some(42),
        );
        let signal = fsk.generate_block();

        // Add some noise
        let snr_db = 20.0;
        let signal_power = 1.0; // FSK has unit magnitude
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;

        let mut awgn = AWGN::new_from_seed(sample_rate, block_size, noise_power, 99);
        let noise = awgn.generate_block();

        let mut noisy_signal: Vec<Complex<f64>> = Vec::with_capacity(block_size);
        for i in 0..block_size {
            noisy_signal.push(signal[i] + noise[i]);
        }

        let nperseg = 2048;
        let (freqs, psd) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("Sample rate: {} Hz", sample_rate);
        println!("FSK frequencies: {} Hz and {} Hz",
                 carrier_freq - deviation/2.0,
                 carrier_freq + deviation/2.0);
        println!("SNR: {} dB", snr_db);
        println!("Segment length: {}", nperseg);

        // Convert to dB
        use crate::vector_ops;
        let psd_db = vector_ops::to_db(&psd);

        // Plot
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &psd_db, "Welch PSD: FSK Signal with AWGN (SNR=20dB)");
    }

    #[test]
    fn test_welch_psk_signal() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch PSK signal plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: QPSK Signal Test ===");

        let sample_rate = 10e6_f64;
        let symbol_rate = 1e6_f64;
        let rolloff = 0.35;
        let block_size = 16384;
        let filter_taps = 51;

        let mut psk = PskCarrier::new(
            sample_rate,
            symbol_rate,
            ModType::_QPSK,
            rolloff,
            block_size,
            filter_taps,
            Some(42),
        );
        let signal = psk.generate_block();

        let nperseg = 2048;
        let (freqs, psd) = welch(
            &signal.iter().cloned().collect::<Vec<_>>(),
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("Sample rate: {} Hz", sample_rate);
        println!("Symbol rate: {} Hz", symbol_rate);
        println!("Rolloff: {}", rolloff);
        println!("Segment length: {}", nperseg);

        // Convert to dB
        use crate::vector_ops;
        let psd_db = vector_ops::to_db(&psd);

        // Plot
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &psd_db, "Welch PSD: QPSK Signal (RRC Shaped)");
    }

    #[test]
    fn test_welch_window_comparison() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch window comparison plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: Window Comparison Test ===");

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64;
        let block_size = 8192;

        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block();
        let signal_vec: Vec<Complex<f64>> = signal.iter().cloned().collect();

        let nperseg = 1024;

        // Compute PSD with different windows
        let (freqs_hann, psd_hann) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        let (_, psd_hamming) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hamming,
            None,
        );

        let (_, psd_blackman) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Blackman,
            None,
        );

        let (_, psd_rect) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Rectangular,
            None,
        );

        println!("CW frequency: {} Hz", freq);
        println!("Comparing Hann, Hamming, Blackman, and Rectangular windows");

        // Convert to dB
        use crate::vector_ops;
        let psd_hann_db = vector_ops::to_db(&psd_hann);
        let psd_hamming_db = vector_ops::to_db(&psd_hamming);
        let psd_blackman_db = vector_ops::to_db(&psd_blackman);
        let psd_rect_db = vector_ops::to_db(&psd_rect);

        // Plot
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_hann = Scatter::new(freqs_hann.clone(), psd_hann_db)
            .mode(Mode::Lines)
            .name("Hann");
        let trace_hamming = Scatter::new(freqs_hann.clone(), psd_hamming_db)
            .mode(Mode::Lines)
            .name("Hamming");
        let trace_blackman = Scatter::new(freqs_hann.clone(), psd_blackman_db)
            .mode(Mode::Lines)
            .name("Blackman");
        let trace_rect = Scatter::new(freqs_hann, psd_rect_db)
            .mode(Mode::Lines)
            .name("Rectangular");

        let layout = Layout::new()
            .title("Welch PSD: Window Function Comparison")
            .x_axis(Axis::new().title("Frequency (Hz)"))
            .y_axis(Axis::new().title("PSD (dB)"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_hann);
        plot.add_trace(trace_hamming);
        plot.add_trace(trace_blackman);
        plot.add_trace(trace_rect);
        plot.set_layout(layout);

        plot.show();
    }

    #[test]
    fn test_welch_averaging_comparison() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch averaging comparison plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: Averaging Method Comparison Test ===");

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64;
        let block_size = 16384;

        // Generate CW with noise
        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block::<f64>();

        let snr_db = 15.0;
        let signal_power = 1.0;
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;

        let mut awgn = AWGN::new_from_seed(sample_rate, block_size, noise_power, 77);
        let noise = awgn.generate_block();

        let mut noisy_signal: Vec<Complex<f64>> = Vec::with_capacity(block_size);
        for i in 0..block_size {
            noisy_signal.push(signal[i] + noise[i]);
        }

        let nperseg = 1024;

        // Compute PSD with different averaging methods
        let (freqs_mean, psd_mean) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            Some(AveragingMethod::Mean),
        );

        let (_, psd_median) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            Some(AveragingMethod::Median),
        );

        let (_, psd_max) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            Some(AveragingMethod::Max),
        );

        let (_, psd_min) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            Some(AveragingMethod::Min),
        );

        println!("CW frequency: {} Hz", freq);
        println!("SNR: {} dB", snr_db);
        println!("Comparing Mean, Median, Max, and Min averaging methods");

        // Convert to dB
        use crate::vector_ops;
        let psd_mean_db = vector_ops::to_db(&psd_mean);
        let psd_median_db = vector_ops::to_db(&psd_median);
        let psd_max_db = vector_ops::to_db(&psd_max);
        let psd_min_db = vector_ops::to_db(&psd_min);

        // Plot
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_mean = Scatter::new(freqs_mean.clone(), psd_mean_db)
            .mode(Mode::Lines)
            .name("Mean (standard)");
        let trace_median = Scatter::new(freqs_mean.clone(), psd_median_db)
            .mode(Mode::Lines)
            .name("Median");
        let trace_max = Scatter::new(freqs_mean.clone(), psd_max_db)
            .mode(Mode::Lines)
            .name("Max");
        let trace_min = Scatter::new(freqs_mean, psd_min_db)
            .mode(Mode::Lines)
            .name("Min");

        let layout = Layout::new()
            .title("Welch PSD: Averaging Method Comparison")
            .x_axis(Axis::new().title("Frequency (Hz)"))
            .y_axis(Axis::new().title("PSD (dB)"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_mean);
        plot.add_trace(trace_median);
        plot.add_trace(trace_max);
        plot.add_trace(trace_min);
        plot.set_layout(layout);

        plot.show();
    }

    #[test]
    fn test_welch_parameter_effects() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch parameter effects plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: Parameter Effects Test ===");

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64;
        let block_size = 16384;

        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block();
        let signal_vec: Vec<Complex<f64>> = signal.iter().cloned().collect();

        // Test different segment lengths
        let (freqs_256, psd_256) = welch(
            &signal_vec,
            sample_rate,
            256,
            None,
            None,
            WindowType::Hann,
            None,
        );

        let (freqs_512, psd_512) = welch(
            &signal_vec,
            sample_rate,
            512,
            None,
            None,
            WindowType::Hann,
            None,
        );

        let (freqs_1024, psd_1024) = welch(
            &signal_vec,
            sample_rate,
            1024,
            None,
            None,
            WindowType::Hann,
            None,
        );

        let (freqs_2048, psd_2048) = welch(
            &signal_vec,
            sample_rate,
            2048,
            None,
            None,
            WindowType::Hann,
            None,
        );

        println!("CW frequency: {} Hz", freq);
        println!("Comparing nperseg values: 256, 512, 1024, 2048");

        // Convert to dB
        use crate::vector_ops;
        let psd_256_db = vector_ops::to_db(&psd_256);
        let psd_512_db = vector_ops::to_db(&psd_512);
        let psd_1024_db = vector_ops::to_db(&psd_1024);
        let psd_2048_db = vector_ops::to_db(&psd_2048);

        // Plot
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_256 = Scatter::new(freqs_256, psd_256_db)
            .mode(Mode::Lines)
            .name("nperseg=256");
        let trace_512 = Scatter::new(freqs_512, psd_512_db)
            .mode(Mode::Lines)
            .name("nperseg=512");
        let trace_1024 = Scatter::new(freqs_1024, psd_1024_db)
            .mode(Mode::Lines)
            .name("nperseg=1024");
        let trace_2048 = Scatter::new(freqs_2048, psd_2048_db)
            .mode(Mode::Lines)
            .name("nperseg=2048");

        let layout = Layout::new()
            .title("Welch PSD: Segment Length Effects")
            .x_axis(Axis::new().title("Frequency (Hz)"))
            .y_axis(Axis::new().title("PSD (dB)"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_256);
        plot.add_trace(trace_512);
        plot.add_trace(trace_1024);
        plot.add_trace(trace_2048);
        plot.set_layout(layout);

        plot.show();
    }

    #[test]
    fn test_welch_overlap_effects() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping Welch overlap effects plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== Welch PSD: Overlap Effects Test ===");

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64;
        let block_size = 16384;

        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block();
        let signal_vec: Vec<Complex<f64>> = signal.iter().cloned().collect();

        let nperseg = 1024;

        // Test different overlap amounts
        let (freqs_0, psd_0) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            Some(0), // 0% overlap
            None,
            WindowType::Hann,
            None,
        );

        let (freqs_50, psd_50) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            Some(nperseg / 2), // 50% overlap
            None,
            WindowType::Hann,
            None,
        );

        let (freqs_75, psd_75) = welch(
            &signal_vec,
            sample_rate,
            nperseg,
            Some(nperseg * 3 / 4), // 75% overlap
            None,
            WindowType::Hann,
            None,
        );

        println!("CW frequency: {} Hz", freq);
        println!("Comparing overlap: 0%, 50%, 75%");

        // Convert to dB
        use crate::vector_ops;
        let psd_0_db = vector_ops::to_db(&psd_0);
        let psd_50_db = vector_ops::to_db(&psd_50);
        let psd_75_db = vector_ops::to_db(&psd_75);

        // Plot
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_0 = Scatter::new(freqs_0, psd_0_db)
            .mode(Mode::Lines)
            .name("0% overlap");
        let trace_50 = Scatter::new(freqs_50, psd_50_db)
            .mode(Mode::Lines)
            .name("50% overlap");
        let trace_75 = Scatter::new(freqs_75, psd_75_db)
            .mode(Mode::Lines)
            .name("75% overlap");

        let layout = Layout::new()
            .title("Welch PSD: Overlap Effects")
            .x_axis(Axis::new().title("Frequency (Hz)"))
            .y_axis(Axis::new().title("PSD (dB)"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_0);
        plot.add_trace(trace_50);
        plot.add_trace(trace_75);
        plot.set_layout(layout);

        plot.show();
    }

    #[test]
    fn test_parseval_theorem_conservation() {
        // Test that total power is conserved between time and frequency domains
        // This validates the PSD normalization is correct per Parseval's theorem:
        // sum(|x[n]|^2) / N = integral(PSD(f) df) from 0 to fs

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64; // 200 kHz tone
        let block_size = 8192;

        // Generate a simple CW tone
        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block::<f64>();

        // Calculate time-domain power (average power)
        let time_power: f64 = signal
            .iter()
            .map(|s| s.norm_sqr())
            .sum::<f64>() / block_size as f64;

        // Calculate frequency-domain power by integrating PSD
        let nperseg = 1024;
        let (freqs, psd) = welch(
            &signal.iter().cloned().collect::<Vec<_>>(),
            sample_rate,
            nperseg,
            None,
            None,
            WindowType::Hann,
            None,
        );

        // Frequency resolution
        let df = freqs[1] - freqs[0]; // Assuming uniform spacing

        // Integrate PSD over all frequency bins to get total power
        let freq_power: f64 = psd.iter().sum::<f64>() * df;

        println!("\n=== Parseval's Theorem Test (CW Tone) ===");
        println!("Time-domain power: {:.6e}", time_power);
        println!("Frequency-domain power: {:.6e}", freq_power);
        println!("Power ratio (freq/time): {:.6}", freq_power / time_power);
        println!("Error: {:.2}%", ((freq_power - time_power).abs() / time_power) * 100.0);

        // Allow 5% tolerance due to spectral leakage and windowing effects
        let relative_error = (freq_power - time_power).abs() / time_power;
        assert!(
            relative_error < 0.05,
            "Parseval's theorem violated: relative error = {:.2}% (expected < 5%)",
            relative_error * 100.0
        );
    }

    #[test]
    fn test_parseval_theorem_with_noise() {
        // Test Parseval's theorem with a signal containing both tone and AWGN
        // This is more realistic and tests the PSD normalization under typical conditions

        let sample_rate = 1e6_f64;
        let freq = 2e5_f64; // 200 kHz tone
        let block_size = 16384;

        // Generate CW tone
        let mut cw = CW::new(freq, sample_rate, block_size);
        let signal = cw.generate_block::<f64>();

        // Add AWGN with known power
        let snr_db = 10.0;
        let signal_power = 1.0;
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;

        let mut awgn = AWGN::new_from_seed(sample_rate, block_size, noise_power, 42);
        let noise = awgn.generate_block::<f64>();

        let mut noisy_signal: Vec<Complex<f64>> = Vec::with_capacity(block_size);
        for i in 0..block_size {
            noisy_signal.push(signal[i] + noise[i]);
        }

        // Calculate time-domain power
        let time_power: f64 = noisy_signal
            .iter()
            .map(|s| s.norm_sqr())
            .sum::<f64>() / block_size as f64;

        // Calculate frequency-domain power
        let nperseg = 1024;
        let (freqs, psd) = welch(
            &noisy_signal,
            sample_rate,
            nperseg,
            Some((nperseg / 2) as usize), // 50% overlap
            None,
            WindowType::Hann,
            Some(AveragingMethod::Mean),
        );

        let df = freqs[1] - freqs[0];
        let freq_power: f64 = psd.iter().sum::<f64>() * df;

        println!("\n=== Parseval's Theorem Test (CW + Noise) ===");
        println!("Time-domain power: {:.6e}", time_power);
        println!("Frequency-domain power: {:.6e}", freq_power);
        println!("Power ratio (freq/time): {:.6}", freq_power / time_power);
        println!("Error: {:.2}%", ((freq_power - time_power).abs() / time_power) * 100.0);

        // More lenient tolerance (10%) with noise and overlapping segments
        let relative_error = (freq_power - time_power).abs() / time_power;
        assert!(
            relative_error < 0.10,
            "Parseval's theorem violated with noise: relative error = {:.2}% (expected < 10%)",
            relative_error * 100.0
        );
    }
}
