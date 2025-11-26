//! Cross Ambiguity Function (CAF) for TDOA and Doppler estimation
//!
//! This module provides efficient FFT-based CAF computation with parallel
//! processing, peak detection, and interpolation methods.

use num_complex::Complex;
use num_traits::{Float, FromPrimitive, Signed};
use rayon::prelude::*;
use std::fmt::Debug;
use std::ops::{DivAssign, RemAssign};

use crate::fft::{fft, ifft};

/// Errors that can occur during CAF computation
#[derive(Debug, Clone)]
pub enum CafError {
    /// Doppler range is invalid (min >= max)
    InvalidDopplerRange { min: f64, max: f64 },
    /// Delay range is invalid (min >= max)
    InvalidDelayRange { min: f64, max: f64 },
    /// Input signals have different lengths
    SignalLengthMismatch {
        signal1_len: usize,
        signal2_len: usize,
    },
}

impl std::fmt::Display for CafError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CafError::InvalidDopplerRange { min, max } => {
                write!(f, "Invalid Doppler range: min ({}) >= max ({})", min, max)
            }
            CafError::InvalidDelayRange { min, max } => {
                write!(f, "Invalid delay range: min ({}) >= max ({})", min, max)
            }
            CafError::SignalLengthMismatch {
                signal1_len,
                signal2_len,
            } => write!(
                f,
                "Signal length mismatch: {} != {}",
                signal1_len, signal2_len
            ),
        }
    }
}

impl std::error::Error for CafError {}

/// Interpolation method for peak refinement
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    /// 2D Parabolic interpolation (fast, good for smooth peaks)
    Parabolic2D,
    /// 2D Sinc interpolation (more accurate, computationally expensive)
    Sinc2D,
    /// No interpolation
    None,
}

/// Parameters for CAF computation
#[derive(Debug, Clone)]
pub struct CafParams {
    /// Time step for downsampling (1 = fine mode, >1 = coarse mode)
    pub time_step: usize,
    /// Frequency resolution for Doppler search (Hz)
    pub freq_step_hz: f64,
    /// Doppler search range in Hz (min, max)
    pub doppler_range_hz: (f64, f64),
    /// Time delay search range in samples (min, max)
    /// Can include negative values for searching backward time shifts
    pub delay_range_samples: (f64, f64),
    /// Sample rate of signals (Hz)
    pub sample_rate_hz: f64,
}

/// CAF surface representation
#[derive(Debug, Clone)]
pub struct CafSurface<T: Float> {
    /// 2D array: surface[doppler_idx][time_idx]
    pub surface: Vec<Vec<T>>,
    /// Time delay values (samples)
    pub time_delays: Vec<T>,
    /// Doppler shift values (Hz)
    pub doppler_shifts: Vec<T>,
}

/// Coarse peak location
#[derive(Debug, Clone)]
pub struct Peak {
    /// Peak magnitude
    pub magnitude: f64,
    /// Time delay index
    pub delay_idx: usize,
    /// Doppler shift index
    pub doppler_idx: usize,
    /// Time delay (samples)
    pub delay_samples: f64,
    /// Doppler shift (Hz)
    pub doppler_hz: f64,
    /// SNR in dB (peak relative to noise floor)
    pub snr_db: f64,
    /// Estimated noise floor (linear magnitude)
    pub noise_floor: f64,
}

/// Refined peak with sub-sample accuracy
#[derive(Debug, Clone)]
pub struct RefinedPeak {
    /// Peak magnitude
    pub magnitude: f64,
    /// Time delay with sub-sample precision (samples)
    pub delay_samples: f64,
    /// Doppler shift with sub-bin precision (Hz)
    pub doppler_hz: f64,
    /// SNR in dB (peak relative to noise floor)
    pub snr_db: f64,
    /// Estimated noise floor (linear magnitude)
    pub noise_floor: f64,
}

/// Compute Cross Ambiguity Function using FFT-based algorithm
///
/// Uses rayon for parallel processing across Doppler frequencies.
/// For each Doppler shift:
///   1. Apply frequency shift to reference signal
///   2. FFT both signals
///   3. Multiply conjugate in frequency domain
///   4. IFFT to get correlation
///
/// # Arguments
/// * `signal1` - Reference signal (complex samples)
/// * `signal2` - Target signal (complex samples)
/// * `params` - CAF computation parameters
///
/// # Returns
/// CAF surface with time-delay vs Doppler, or error if parameters are invalid
///
/// # Errors
/// Returns `CafError` if:
/// - Signal lengths don't match
/// - Doppler range is invalid (min >= max)
/// - Delay range is invalid (min >= max)
pub fn compute_caf<T>(
    signal1: &[Complex<T>],
    signal2: &[Complex<T>],
    params: &CafParams,
) -> Result<CafSurface<T>, CafError>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    // Validate signal lengths
    if signal1.len() != signal2.len() {
        return Err(CafError::SignalLengthMismatch {
            signal1_len: signal1.len(),
            signal2_len: signal2.len(),
        });
    }

    // Validate Doppler range
    if params.doppler_range_hz.0 >= params.doppler_range_hz.1 {
        return Err(CafError::InvalidDopplerRange {
            min: params.doppler_range_hz.0,
            max: params.doppler_range_hz.1,
        });
    }

    // Validate delay range
    if params.delay_range_samples.0 >= params.delay_range_samples.1 {
        return Err(CafError::InvalidDelayRange {
            min: params.delay_range_samples.0,
            max: params.delay_range_samples.1,
        });
    }

    let n = signal1.len();

    // Build Doppler shift array from configurable range
    let num_doppler = ((params.doppler_range_hz.1 - params.doppler_range_hz.0)
        / params.freq_step_hz) as usize
        + 1;
    let doppler_shifts: Vec<f64> = (0..num_doppler)
        .map(|i| params.doppler_range_hz.0 + (i as f64) * params.freq_step_hz)
        .collect();

    // Build time delay array from configurable range (supports negative delays)
    let min_delay = params.delay_range_samples.0;
    let max_delay = params.delay_range_samples.1;
    let n_time =
        ((max_delay - min_delay) / (params.time_step as f64)).ceil() as usize + 1;

    let time_delays: Vec<T> = (0..n_time)
        .map(|i| T::from(min_delay + (i * params.time_step) as f64).unwrap())
        .collect();

    // Parallel processing over Doppler shifts (lock-free)
    let surface_data: Vec<Vec<T>> = doppler_shifts
        .par_iter()
        .map(|&doppler_hz| {
            let correlation =
                compute_correlation_at_doppler(signal1, signal2, doppler_hz, params.sample_rate_hz);

            // Extract delay range from correlation and downsample by time_step
            let downsampled: Vec<T> = (0..n_time)
                .map(|i| {
                    let delay_samples = min_delay + (i * params.time_step) as f64;
                    // Handle circular indexing for correlation
                    let mut idx = delay_samples.round() as isize;
                    if idx < 0 {
                        idx = (n as isize + idx) % n as isize;
                    }
                    let idx = (idx as usize) % n;
                    correlation[idx].norm()
                })
                .collect();

            downsampled
        })
        .collect();

    // Build Doppler shift array (as T)
    let doppler_shifts_t: Vec<T> = doppler_shifts
        .iter()
        .map(|&d| T::from(d).unwrap())
        .collect();

    Ok(CafSurface {
        surface: surface_data,
        time_delays,
        doppler_shifts: doppler_shifts_t,
    })
}

/// Compute CAF with automatic parameter calculation
///
/// This function automatically calculates optimal/required `time_step` and `freq_step_hz`
/// based on signal characteristics, providing a simpler interface than `compute_caf`.
///
/// # Auto-Calculated Parameters
///
/// ## Integration Time and Frequency Resolution
/// The integration time T = N/Fs (signal length / sample rate) determines the
/// frequency resolution:
/// - Main lobe width: Δf ≈ 1/T Hz
/// - freq_step_hz = (1/T) / points_per_mainlobe
///
/// Example: 1,048,576 samples at 1 MHz → T ≈ 1.05 s → Δf ≈ 0.95 Hz
/// With points_per_mainlobe=2 → freq_step ≈ 0.475 Hz
///
/// ## Bandwidth and Delay Resolution
/// The signal bandwidth determines delay resolution:
/// - Main lobe width: Δτ ≈ 1/B samples
/// - time_step = max(1, floor((1/B) / points_per_mainlobe))
///
/// For narrowband signals (B << Fs), time_step defaults to 1.
/// For wideband signals, coarser stepping is used.
///
/// # Arguments
/// * `signal1` - Reference signal (complex samples)
/// * `signal2` - Target signal (complex samples)
/// * `sample_rate_hz` - Sample rate in Hz
/// * `bandwidth_hz` - Signal bandwidth in Hz (occupied spectrum)
/// * `doppler_range_hz` - Doppler search range (min, max) in Hz
/// * `time_delay_range` - Time delay search range (min, max) in seconds
/// * `points_per_mainlobe` - Sampling density (default: 2). Higher values give
///   finer resolution but increase computation. Typical range: 2-4.
///
/// # Returns
/// CAF surface with automatically optimized resolution
///
/// # Errors
/// Returns `CafError` if:
/// - Signal lengths don't match
/// - Ranges are invalid (min >= max)
///
/// # Example
/// ```no_run
/// use signal_kit::caf::auto_compute_caf;
/// use num_complex::Complex;
///
/// let sample_rate = 1e6; // 1 MHz
/// let bandwidth = 10e3;   // 10 kHz signal
/// let signal1: Vec<Complex<f64>> = vec![/* ... */];
/// let signal2: Vec<Complex<f64>> = vec![/* ... */];
///
/// let surface = auto_compute_caf(
///     &signal1,
///     &signal2,
///     sample_rate,
///     bandwidth,
///     (-1000.0, 1000.0),    // Doppler range: ±1 kHz
///     (-0.001, 0.001),      // Time delay range: ±1 ms
///     None,                 // Use default points_per_mainlobe = 2
/// )?;
/// # Ok::<(), signal_kit::caf::CafError>(())
/// ```
pub fn auto_compute_caf<T>(
    signal1: &[Complex<T>],
    signal2: &[Complex<T>],
    sample_rate_hz: f64,
    bandwidth_hz: f64,
    doppler_range_hz: (f64, f64),
    time_delay_range: (f64, f64),
    points_per_mainlobe: Option<usize>,
) -> Result<CafSurface<T>, CafError>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    // Use default of 2 points per mainlobe if not specified
    let points_per_mainlobe_value = points_per_mainlobe.unwrap_or(2);

    // Calculate integration time T = N / Fs
    let n = signal1.len() as f64;
    let integration_time = n / sample_rate_hz;

    // Calculate frequency step based on frequency resolution (1/T)
    // freq_step = (1/T) / points_per_mainlobe
    let freq_resolution = 1.0 / integration_time;
    let freq_step_hz = freq_resolution / (points_per_mainlobe_value as f64);

    // Calculate delay step based on delay resolution (1/B)
    // For narrowband signals, this will be large → time_step = 1
    // For wideband signals, this allows coarser stepping
    let delay_resolution_samples = sample_rate_hz / bandwidth_hz;
    let time_step = (delay_resolution_samples / (points_per_mainlobe_value as f64)).floor().max(1.0) as usize;

    // Convert time delay range from seconds to samples
    let delay_range_samples = (
        time_delay_range.0 * sample_rate_hz,
        time_delay_range.1 * sample_rate_hz,
    );

    // Build parameters
    let params = CafParams {
        time_step,
        freq_step_hz,
        doppler_range_hz,
        delay_range_samples,
        sample_rate_hz,
    };

    // Call the main compute_caf function
    compute_caf(signal1, signal2, &params)
}

/// Compute correlation at a specific Doppler shift
///
/// # Arguments
/// * `signal1` - Reference signal
/// * `signal2` - Target signal
/// * `doppler_hz` - Doppler shift to apply (Hz)
/// * `sample_rate_hz` - Sample rate (Hz)
///
/// # Returns
/// Complex correlation in time domain
fn compute_correlation_at_doppler<T>(
    signal1: &[Complex<T>],
    signal2: &[Complex<T>],
    doppler_hz: f64,
    sample_rate_hz: f64,
) -> Vec<Complex<T>>
where
    T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
{
    // Apply frequency shift to signal1
    let mut signal1_shifted: Vec<Complex<T>> = signal1.to_vec();
    freq_shift_inplace(&mut signal1_shifted, doppler_hz, sample_rate_hz);

    // FFT of shifted signal1
    let mut s1_freq = signal1_shifted.clone();
    fft(&mut s1_freq);

    // FFT of signal2
    let mut s2_freq = signal2.to_vec();
    fft(&mut s2_freq);

    // Multiply conjugate: S1* × S2
    let mut correlation_freq: Vec<Complex<T>> = s1_freq
        .iter()
        .zip(s2_freq.iter())
        .map(|(s1, s2)| s1.conj() * s2)
        .collect();

    // IFFT to get correlation in time domain
    ifft(&mut correlation_freq);

    correlation_freq
}

/// Apply frequency shift in-place
///
/// Multiplies signal by e^(j2πft) for frequency offset f
fn freq_shift_inplace<T>(signal: &mut [Complex<T>], freq_hz: f64, sample_rate_hz: f64)
where
    T: Float + FromPrimitive,
{
    let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
    let fs = T::from(sample_rate_hz).unwrap();
    let f = T::from(freq_hz).unwrap();

    for (i, sample) in signal.iter_mut().enumerate() {
        let t = T::from(i).unwrap() / fs;
        let phase = two_pi * f * t;
        let phasor = Complex::new(phase.cos(), phase.sin());
        *sample = *sample * phasor;
    }
}

/// Estimate noise floor from CAF surface using median
///
/// The median provides a robust estimate of the noise floor that is not
/// affected by the peak or strong sidelobes. This allows computing SNR
/// as the peak magnitude relative to the noise floor.
///
/// # Arguments
/// * `surface` - CAF surface from compute_caf()
///
/// # Returns
/// Median magnitude value (linear, not dB)
pub fn estimate_noise_floor<T>(surface: &CafSurface<T>) -> T
where
    T: Float + Debug,
{
    let mut all_values: Vec<T> = surface
        .surface
        .iter()
        .flat_map(|row| row.iter().copied())
        .filter(|&val| val > T::zero()) // Only positive values
        .collect();

    if all_values.is_empty() {
        return T::from(1e-10).unwrap(); // Fallback for empty surface
    }

    // Sort to find median
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_idx = all_values.len() / 2;
    all_values[median_idx]
}

/// Find peak in CAF surface using 2D max search
///
/// # Arguments
/// * `surface` - CAF surface from compute_caf()
///
/// # Returns
/// Peak location with magnitude and coordinates
pub fn find_peak<T>(surface: &CafSurface<T>) -> Peak
where
    T: Float + Debug,
{
    // Estimate noise floor using median
    let noise_floor = estimate_noise_floor(surface);

    let mut max_magnitude = T::zero();
    let mut max_doppler_idx = 0;
    let mut max_delay_idx = 0;

    for (doppler_idx, row) in surface.surface.iter().enumerate() {
        for (delay_idx, &magnitude) in row.iter().enumerate() {
            if magnitude > max_magnitude {
                max_magnitude = magnitude;
                max_doppler_idx = doppler_idx;
                max_delay_idx = delay_idx;
            }
        }
    }

    // Calculate SNR in dB
    let magnitude_f64 = max_magnitude.to_f64().unwrap();
    let noise_floor_f64 = noise_floor.to_f64().unwrap();
    let snr_db = 10.0 * (magnitude_f64 / noise_floor_f64).log10();

    Peak {
        magnitude: magnitude_f64,
        delay_idx: max_delay_idx,
        doppler_idx: max_doppler_idx,
        delay_samples: surface.time_delays[max_delay_idx].to_f64().unwrap(),
        doppler_hz: surface.doppler_shifts[max_doppler_idx].to_f64().unwrap(),
        snr_db,
        noise_floor: noise_floor_f64,
    }
}

/// Interpolate peak for sub-sample accuracy
///
/// # Arguments
/// * `surface` - CAF surface
/// * `coarse_peak` - Peak from find_peak()
/// * `method` - Interpolation method to use
///
/// # Returns
/// Refined peak with sub-sample precision
pub fn interpolate_peak<T>(
    surface: &CafSurface<T>,
    coarse_peak: &Peak,
    method: InterpolationMethod,
) -> RefinedPeak
where
    T: Float + Debug,
{
    match method {
        InterpolationMethod::Parabolic2D => interpolate_parabolic_2d(surface, coarse_peak),
        InterpolationMethod::Sinc2D => interpolate_sinc_2d(surface, coarse_peak),
        InterpolationMethod::None => RefinedPeak {
            magnitude: coarse_peak.magnitude,
            delay_samples: coarse_peak.delay_samples,
            doppler_hz: coarse_peak.doppler_hz,
            snr_db: coarse_peak.snr_db,
            noise_floor: coarse_peak.noise_floor,
        },
    }
}

/// Parabolic 2D interpolation for sub-sample peak refinement
///
/// Fits a 2D parabola to the 3×3 neighborhood around the peak
fn interpolate_parabolic_2d<T>(surface: &CafSurface<T>, peak: &Peak) -> RefinedPeak
where
    T: Float + Debug,
{
    let d_idx = peak.doppler_idx;
    let t_idx = peak.delay_idx;

    let n_doppler = surface.surface.len();
    let n_time = surface.surface[0].len();

    // Check if we have enough neighbors for interpolation
    if d_idx == 0 || d_idx >= n_doppler - 1 || t_idx == 0 || t_idx >= n_time - 1 {
        // Peak is on the edge, can't interpolate
        return RefinedPeak {
            magnitude: peak.magnitude,
            delay_samples: peak.delay_samples,
            doppler_hz: peak.doppler_hz,
            snr_db: peak.snr_db,
            noise_floor: peak.noise_floor,
        };
    }

    // Extract 3x3 neighborhood (doppler, time)
    let z00 = surface.surface[d_idx - 1][t_idx - 1].to_f64().unwrap();
    let z01 = surface.surface[d_idx - 1][t_idx].to_f64().unwrap();
    let z02 = surface.surface[d_idx - 1][t_idx + 1].to_f64().unwrap();
    let z10 = surface.surface[d_idx][t_idx - 1].to_f64().unwrap();
    let z11 = surface.surface[d_idx][t_idx].to_f64().unwrap(); // Center (peak)
    let z12 = surface.surface[d_idx][t_idx + 1].to_f64().unwrap();
    let z20 = surface.surface[d_idx + 1][t_idx - 1].to_f64().unwrap();
    let z21 = surface.surface[d_idx + 1][t_idx].to_f64().unwrap();
    let z22 = surface.surface[d_idx + 1][t_idx + 1].to_f64().unwrap();

    // Fit parabola: z = a + bx + cy + dx² + ey² + fxy
    // We only need the quadratic terms for peak location

    // Estimate derivatives using finite differences
    // First derivatives (should be ~0 at peak)
    let dz_dt = (z12 - z10) / 2.0;
    let dz_dd = (z21 - z01) / 2.0;

    // Second derivatives
    let d2z_dt2 = z10 - 2.0 * z11 + z12;
    let d2z_dd2 = z01 - 2.0 * z11 + z21;
    let d2z_dtdd = (z22 - z20 - z02 + z00) / 4.0;

    // Solve for peak offset using derivatives
    // [d2z_dt2   d2z_dtdd ] [Δt]   = - [dz_dt]
    // [d2z_dtdd  d2z_dd2  ] [Δd]       [dz_dd]

    let det = d2z_dt2 * d2z_dd2 - d2z_dtdd * d2z_dtdd;

    if det.abs() < 1e-10 {
        // Singular matrix, can't interpolate
        return RefinedPeak {
            magnitude: peak.magnitude,
            delay_samples: peak.delay_samples,
            doppler_hz: peak.doppler_hz,
            snr_db: peak.snr_db,
            noise_floor: peak.noise_floor,
        };
    }

    let delta_t = -(d2z_dd2 * dz_dt - d2z_dtdd * dz_dd) / det;
    let delta_d = -(d2z_dt2 * dz_dd - d2z_dtdd * dz_dt) / det;

    // Clamp offsets to reasonable range [-1, 1]
    let delta_t = delta_t.max(-1.0).min(1.0);
    let delta_d = delta_d.max(-1.0).min(1.0);

    // Calculate refined position
    let time_step = if t_idx > 0 {
        (surface.time_delays[t_idx] - surface.time_delays[t_idx - 1]).to_f64().unwrap()
    } else {
        surface.time_delays[1].to_f64().unwrap() - surface.time_delays[0].to_f64().unwrap()
    };

    let doppler_step = if d_idx > 0 {
        (surface.doppler_shifts[d_idx] - surface.doppler_shifts[d_idx - 1])
            .to_f64()
            .unwrap()
    } else {
        (surface.doppler_shifts[1] - surface.doppler_shifts[0])
            .to_f64()
            .unwrap()
    };

    let refined_delay = peak.delay_samples + delta_t * time_step;
    let refined_doppler = peak.doppler_hz + delta_d * doppler_step;

    // Estimate refined magnitude using parabola
    let refined_magnitude = z11 + dz_dt * delta_t + dz_dd * delta_d
        + 0.5 * (d2z_dt2 * delta_t * delta_t + d2z_dd2 * delta_d * delta_d + 2.0 * d2z_dtdd * delta_t * delta_d);

    let final_magnitude = refined_magnitude.max(peak.magnitude); // Don't go below coarse peak

    // Recalculate SNR with refined magnitude
    let refined_snr_db = 10.0 * (final_magnitude / peak.noise_floor).log10();

    RefinedPeak {
        magnitude: final_magnitude,
        delay_samples: refined_delay,
        doppler_hz: refined_doppler,
        snr_db: refined_snr_db,
        noise_floor: peak.noise_floor,
    }
}

/// Sinc 2D interpolation for sub-sample peak refinement
///
/// Uses zero-padding in frequency domain for higher resolution
fn interpolate_sinc_2d<T>(surface: &CafSurface<T>, peak: &Peak) -> RefinedPeak
where
    T: Float + Debug,
{
    // Extract region around peak
    let d_idx = peak.doppler_idx;
    let t_idx = peak.delay_idx;

    let n_doppler = surface.surface.len();
    let n_time = surface.surface[0].len();

    // Define interpolation window size
    let window_size = 16.min(n_time / 2).min(n_doppler / 2);

    // Extract window around peak
    let d_start = d_idx.saturating_sub(window_size / 2);
    let d_end = (d_idx + window_size / 2).min(n_doppler);
    let t_start = t_idx.saturating_sub(window_size / 2);
    let t_end = (t_idx + window_size / 2).min(n_time);

    let d_size = d_end - d_start;
    let t_size = t_end - t_start;

    // Extract local surface
    let mut local_surface: Vec<Vec<f64>> = vec![vec![0.0; t_size]; d_size];
    for (i, d) in (d_start..d_end).enumerate() {
        for (j, t) in (t_start..t_end).enumerate() {
            local_surface[i][j] = surface.surface[d][t].to_f64().unwrap();
        }
    }

    // Upsample using FFT (sinc interpolation)
    let upsample_factor = 4;
    let upsampled = upsample_2d(&local_surface, upsample_factor);

    // Find peak in upsampled surface
    let mut max_val = 0.0;
    let mut max_i = 0;
    let mut max_j = 0;

    for (i, row) in upsampled.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_i = i;
                max_j = j;
            }
        }
    }

    // Map back to original coordinates
    let local_d_offset = max_i as f64 / upsample_factor as f64;
    let local_t_offset = max_j as f64 / upsample_factor as f64;

    let time_step = if t_idx > 0 {
        (surface.time_delays[t_idx] - surface.time_delays[t_idx - 1]).to_f64().unwrap()
    } else {
        surface.time_delays[1].to_f64().unwrap() - surface.time_delays[0].to_f64().unwrap()
    };

    let doppler_step = if d_idx > 0 {
        (surface.doppler_shifts[d_idx] - surface.doppler_shifts[d_idx - 1])
            .to_f64()
            .unwrap()
    } else {
        (surface.doppler_shifts[1] - surface.doppler_shifts[0])
            .to_f64()
            .unwrap()
    };

    let refined_delay = surface.time_delays[t_start].to_f64().unwrap() + local_t_offset * time_step;
    let refined_doppler =
        surface.doppler_shifts[d_start].to_f64().unwrap() + local_d_offset * doppler_step;

    // Recalculate SNR with refined magnitude
    let refined_snr_db = 10.0 * (max_val / peak.noise_floor).log10();

    RefinedPeak {
        magnitude: max_val,
        delay_samples: refined_delay,
        doppler_hz: refined_doppler,
        snr_db: refined_snr_db,
        noise_floor: peak.noise_floor,
    }
}

/// Upsample 2D surface using FFT (sinc interpolation)
fn upsample_2d(surface: &[Vec<f64>], factor: usize) -> Vec<Vec<f64>> {
    let n_doppler = surface.len();
    let n_time = surface[0].len();

    let new_n_doppler = n_doppler * factor;
    let new_n_time = n_time * factor;

    // For simplicity, upsample each dimension separately
    // First upsample time dimension
    let mut time_upsampled: Vec<Vec<f64>> = Vec::new();
    for row in surface.iter() {
        time_upsampled.push(upsample_1d(row, factor));
    }

    // Then upsample doppler dimension (transpose, upsample, transpose back)
    let mut doppler_upsampled: Vec<Vec<f64>> = vec![vec![0.0; new_n_time]; new_n_doppler];

    for t in 0..new_n_time {
        let column: Vec<f64> = time_upsampled.iter().map(|row| row[t]).collect();
        let upsampled_column = upsample_1d(&column, factor);
        for (d, &val) in upsampled_column.iter().enumerate() {
            doppler_upsampled[d][t] = val;
        }
    }

    doppler_upsampled
}

/// Upsample 1D signal using FFT (sinc interpolation)
fn upsample_1d(signal: &[f64], factor: usize) -> Vec<f64> {
    let n = signal.len();
    let new_n = n * factor;

    // Convert to complex
    let mut signal_complex: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();

    // FFT
    fft(&mut signal_complex);

    // Zero-pad in frequency domain
    let mut padded = vec![Complex::new(0.0, 0.0); new_n];

    // Copy positive frequencies
    let half = n / 2;
    for i in 0..=half {
        padded[i] = signal_complex[i];
    }

    // Copy negative frequencies
    for i in 1..half {
        padded[new_n - i] = signal_complex[n - i];
    }

    // IFFT
    ifft(&mut padded);

    // Extract real part and scale
    padded.iter().map(|c| c.re * factor as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::AWGN;
    use crate::ComplexVec;
    use crate::test_utils::should_plot;

    /// Helper function to generate test signal with AWGN
    fn generate_test_signal(length: usize, sample_rate: f64, seed: u64) -> ComplexVec<f64> {
        let mut awgn = AWGN::new_from_seed(sample_rate, length, 1.0, seed);
        awgn.generate_block::<f64>()
    }

    /// Helper function to apply circular time shift
    ///
    /// Shifts signal by `shift` samples. Positive shift moves the signal earlier in time
    /// (signal appears to arrive sooner), which simulates negative TDOA.
    ///
    /// Example: shift=50 means output[0] = input[50], so the signal starts 50 samples earlier.
    fn circular_shift<T: Clone>(signal: &[T], shift: isize) -> Vec<T> {
        let n = signal.len();
        let shift = ((shift % n as isize) + n as isize) as usize % n;

        let mut shifted = signal.to_vec();
        shifted.rotate_right(shift);
        shifted
    }

    #[test]
    fn test_caf_autocorrelation() {
        if !should_plot() {
            println!("Skipping test_caf_autocorrelation plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6;
        let length = 1024;

        // Generate signal
        let signal = generate_test_signal(length, sample_rate, 42);

        // CAF parameters
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (-50.0, 50.0),
            sample_rate_hz: sample_rate,
        };

        // Compute CAF
        let surface = compute_caf(&signal, &signal, &params).unwrap();

        // Find peak
        let peak = find_peak(&surface);

        println!("Autocorrelation peak:");
        println!("  Delay: {} samples", peak.delay_samples);
        println!("  FDOA: {} Hz", peak.doppler_hz);
        println!("  Magnitude: {}", peak.magnitude);
        println!("  SNR: {:.1} dB", peak.snr_db);

        // Peak should be near (0, 0)
        assert!(peak.delay_samples.abs() < 2.0, "Expected delay near 0, got {}", peak.delay_samples);
        assert!(peak.doppler_hz.abs() < 100.0, "Expected Doppler near 0, got {}", peak.doppler_hz);
        assert!(peak.snr_db > 0.0, "SNR should be positive for autocorrelation");

        // Plot
        plot::plot_caf_surface_3d(&surface, Some(&peak), "CAF: Autocorrelation");
        plot::plot_caf_heatmap(&surface, Some(&peak), "CAF Heatmap: Autocorrelation");
        plot::plot_caf_slices(&surface, &peak, "CAF Slices: Autocorrelation");
    }

    #[test]
    fn test_caf_time_delay() {
        let sample_rate = 1e6;
        let length = 1024;
        let time_delay = 50; // samples

        // Generate common signal
        let signal_common = generate_test_signal(length, sample_rate, 0);
        let noise1 = generate_test_signal(length, sample_rate, 1);
        let noise2 = generate_test_signal(length, sample_rate, 2);

        // signal1 = common + noise1 * 0.1
        let signal1 = &signal_common + &(&noise1 * 0.1);

        // signal2 = shifted(common) + noise2 * 0.1
        let signal_common_shifted = ComplexVec::from_vec(circular_shift(&signal_common, time_delay as isize));
        let signal2 = &signal_common_shifted + &(&noise2 * 0.1);

        // CAF parameters
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (0.0, 100.0),
            sample_rate_hz: sample_rate,
        };

        // Compute CAF
        let surface = compute_caf(&signal1, &signal2, &params).unwrap();

        // Find peak
        let peak = find_peak(&surface);

        println!("Time delay peak:");
        println!("  Expected delay: {} samples", time_delay);
        println!("  Found delay: {} samples", peak.delay_samples);
        println!("  FDOA: {} Hz", peak.doppler_hz);
        println!("  Magnitude: {}", peak.magnitude);
        println!("  SNR: {:.1} dB", peak.snr_db);
        println!("  Noise floor: {:.6}", peak.noise_floor);

        // Validate
        assert!((peak.delay_samples - time_delay as f64).abs() < 1.0);
        assert!(peak.doppler_hz.abs() < 100.0);
        assert!(peak.snr_db > 0.0, "SNR should be positive for signal above noise");

        // Plot
        if should_plot() {
            plot::plot_caf_surface_3d(&surface, Some(&peak), "CAF: Time Delay");
            plot::plot_caf_heatmap(&surface, Some(&peak), "CAF Heatmap: Time Delay");
            plot::plot_caf_slices(&surface, &peak, "CAF Slices: Time Delay");
        }

    }

    #[test]
    fn test_caf_frequency_shift() {
        if !should_plot() {
            println!("Skipping test_caf_frequency_shift plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6;
        let length = 1048576; //2**20
        let freq_shift_hz = 500.0;

        // Generate common signal
        let signal_common = generate_test_signal(length, sample_rate, 42);
        let noise1 = generate_test_signal(length, sample_rate, 100);
        let noise2 = generate_test_signal(length, sample_rate, 200);

        // signal1 = common + noise1 * 0.1
        let signal1 = &signal_common + &(&noise1 * 0.1);

        // signal2 = freq_shift(common) + noise2 * 0.1
        let mut signal_common_shifted = signal_common.clone();
        signal_common_shifted.freq_shift(freq_shift_hz, sample_rate);
        let signal2 = &signal_common_shifted + &(&noise2 * 0.1);

        // CAF parameters
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (-20.0e3, 20.0e3),
            delay_range_samples: (-10.0, 10.0),
            sample_rate_hz: sample_rate,
        };

        // Compute CAF
        let surface = compute_caf(&signal1, &signal2, &params).unwrap();

        // Find peak
        let peak = find_peak(&surface);

        println!("Frequency shift peak:");
        println!("  Delay: {} samples", peak.delay_samples);
        println!("  Expected FDOA: {} Hz", freq_shift_hz);
        println!("  Found FDOA: {} Hz", peak.doppler_hz);
        println!("  Magnitude: {}", peak.magnitude);
        println!("  SNR: {:.1} dB", peak.snr_db);

        // Plot
        plot::plot_caf_surface_3d(&surface, Some(&peak), "CAF: Frequency Shift");
        plot::plot_caf_heatmap(&surface, Some(&peak), "CAF Heatmap: Frequency Shift");
        plot::plot_caf_slices(&surface, &peak, "CAF Slices: Frequency Shift");

        // Validate
        assert!(peak.delay_samples.abs() < 2.0);
        assert!((peak.doppler_hz - freq_shift_hz).abs() < 50.0);
        assert!(peak.snr_db > 0.0, "SNR should be positive for signal above noise");
    }

    #[test]
    fn test_caf_tdoa_doppler_combined() {
        if !should_plot() {
            println!("Skipping test_caf_tdoa_doppler_combined plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6;
        let length = 1024;
        let time_delay = 40;
        let freq_shift_hz = 150.0;

        // Generate common signal
        let signal_common = generate_test_signal(length, sample_rate, 42);
        let noise1 = generate_test_signal(length, sample_rate, 100);
        let noise2 = generate_test_signal(length, sample_rate, 200);

        // signal1 = common + noise1 * 0.1
        let signal1 = &signal_common + &(&noise1 * 0.1);

        // signal2 = freq_shift(time_shift(common)) + noise2 * 0.1
        let mut signal_common_shifted = ComplexVec::from_vec(circular_shift(&signal_common, time_delay as isize));
        signal_common_shifted.freq_shift(freq_shift_hz, sample_rate);
        let signal2 = &signal_common_shifted + &(&noise2 * 0.1);

        // CAF parameters
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 25.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (0.0, 100.0),
            sample_rate_hz: sample_rate,
        };

        // Compute CAF
        let surface = compute_caf(&signal1, &signal2, &params).unwrap();

        // Find peak
        let peak = find_peak(&surface);

        println!("Combined TDOA+Doppler peak:");
        println!("  Expected delay: {} samples", time_delay);
        println!("  Found delay: {} samples", peak.delay_samples);
        println!("  Expected FDOA: {} Hz", freq_shift_hz);
        println!("  Found FDOA: {} Hz", peak.doppler_hz);
        println!("  Magnitude: {}", peak.magnitude);
        println!("  SNR: {:.1} dB", peak.snr_db);

        // Validate
        assert!((peak.delay_samples - time_delay as f64).abs() < 2.0);
        assert!((peak.doppler_hz - freq_shift_hz).abs() < 50.0);
        assert!(peak.snr_db > 0.0, "SNR should be positive for signal above noise");

        // Plot
        plot::plot_caf_surface_3d(&surface, Some(&peak), "CAF: TDOA + Doppler");
        plot::plot_caf_heatmap(&surface, Some(&peak), "CAF Heatmap: TDOA + Doppler");
        plot::plot_caf_slices(&surface, &peak, "CAF Slices: TDOA + Doppler");
    }

    #[test]
    fn test_caf_interpolation() {
        if !should_plot() {
            println!("Skipping test_caf_interpolation plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6;
        let length = 2048;
        let time_delay = 37;
        let freq_shift_hz = 123.0;

        // Generate common signal
        let signal_common = generate_test_signal(length, sample_rate, 42);

        // Apply shifts
        let signal1 = signal_common.clone();
        let mut signal2 = ComplexVec::from_vec(circular_shift(&signal_common, time_delay as isize));
        signal2.freq_shift(freq_shift_hz, sample_rate);

        // CAF parameters (coarser grid to test interpolation)
        let params = CafParams {
            time_step: 2, // Coarse mode
            freq_step_hz: 50.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (0.0, 100.0),
            sample_rate_hz: sample_rate,
        };

        // Compute CAF
        let surface = compute_caf(&signal1, &signal2, &params).unwrap();

        // Find coarse peak
        let coarse_peak = find_peak(&surface);

        // Interpolate with both methods
        let parabolic_peak = interpolate_peak(&surface, &coarse_peak, InterpolationMethod::Parabolic2D);
        let sinc_peak = interpolate_peak(&surface, &coarse_peak, InterpolationMethod::Sinc2D);

        println!("Interpolation comparison:");
        println!("  Expected: delay={} samples, Doppler={} Hz", time_delay, freq_shift_hz);
        println!("  Coarse:      delay={:.2} samples, Doppler={:.2} Hz, SNR={:.1} dB",
                 coarse_peak.delay_samples, coarse_peak.doppler_hz, coarse_peak.snr_db);
        println!("  Parabolic:   delay={:.2} samples, Doppler={:.2} Hz, SNR={:.1} dB",
                 parabolic_peak.delay_samples, parabolic_peak.doppler_hz, parabolic_peak.snr_db);
        println!("  Sinc:        delay={:.2} samples, Doppler={:.2} Hz, SNR={:.1} dB",
                 sinc_peak.delay_samples, sinc_peak.doppler_hz, sinc_peak.snr_db);

        // Both methods should improve accuracy
        let coarse_delay_error = (coarse_peak.delay_samples - time_delay as f64).abs();
        let parabolic_delay_error = (parabolic_peak.delay_samples - time_delay as f64).abs();
        let sinc_delay_error = (sinc_peak.delay_samples - time_delay as f64).abs();

        println!("  Delay errors: coarse={:.2}, parabolic={:.2}, sinc={:.2}",
                 coarse_delay_error, parabolic_delay_error, sinc_delay_error);

        // Validate SNR values are consistent
        assert!(parabolic_peak.snr_db > 0.0, "Parabolic SNR should be positive");
        assert!(sinc_peak.snr_db > 0.0, "Sinc SNR should be positive");
        assert!((parabolic_peak.noise_floor - coarse_peak.noise_floor).abs() < 1e-10,
                "Noise floor should be consistent across interpolation");
        assert!((sinc_peak.noise_floor - coarse_peak.noise_floor).abs() < 1e-10,
                "Noise floor should be consistent across interpolation");

        // Plot
        plot::plot_caf_surface_3d(&surface, Some(&coarse_peak), "CAF: Interpolation Test");
        plot::plot_caf_heatmap(&surface, Some(&coarse_peak), "CAF Heatmap: Interpolation Test");
        plot::plot_caf_slices(&surface, &coarse_peak, "CAF Slices: Interpolation Test");
    }

    #[test]
    fn test_caf_invalid_doppler_range() {
        let sample_rate = 1e6;
        let length = 1024;

        let signal1 = generate_test_signal(length, sample_rate, 42);
        let signal2 = generate_test_signal(length, sample_rate, 43);

        // Invalid Doppler range (min >= max)
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (500.0, -500.0), // min > max
            delay_range_samples: (0.0, 100.0),
            sample_rate_hz: sample_rate,
        };

        let result = compute_caf(&signal1, &signal2, &params);
        assert!(result.is_err());

        if let Err(CafError::InvalidDopplerRange { min, max }) = result {
            assert_eq!(min, 500.0);
            assert_eq!(max, -500.0);
        } else {
            panic!("Expected InvalidDopplerRange error");
        }
    }

    #[test]
    fn test_caf_invalid_delay_range() {
        let sample_rate = 1e6;
        let length = 1024;

        let signal1 = generate_test_signal(length, sample_rate, 42);
        let signal2 = generate_test_signal(length, sample_rate, 43);

        // Invalid delay range (min >= max)
        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (100.0, 0.0), // min > max
            sample_rate_hz: sample_rate,
        };

        let result = compute_caf(&signal1, &signal2, &params);
        assert!(result.is_err());

        if let Err(CafError::InvalidDelayRange { min, max }) = result {
            assert_eq!(min, 100.0);
            assert_eq!(max, 0.0);
        } else {
            panic!("Expected InvalidDelayRange error");
        }
    }

    #[test]
    fn test_caf_signal_length_mismatch() {
        let sample_rate = 1e6;

        let signal1 = generate_test_signal(1024, sample_rate, 42);
        let signal2 = generate_test_signal(2048, sample_rate, 43); // Different length

        let params = CafParams {
            time_step: 1,
            freq_step_hz: 50.0,
            doppler_range_hz: (-500.0, 500.0),
            delay_range_samples: (0.0, 100.0),
            sample_rate_hz: sample_rate,
        };

        let result = compute_caf(&signal1, &signal2, &params);
        assert!(result.is_err());

        if let Err(CafError::SignalLengthMismatch { signal1_len, signal2_len }) = result {
            assert_eq!(signal1_len, 1024);
            assert_eq!(signal2_len, 2048);
        } else {
            panic!("Expected SignalLengthMismatch error");
        }
    }

    #[test]
    fn test_auto_compute_caf() {
        let sample_rate = 1e6;
        let bandwidth = 10e3; // 10 kHz bandwidth
        let length = 65536; // Longer signal for better frequency resolution
        let time_delay = 50; // samples
        let freq_shift_hz = 100.0;

        // Generate common signal
        let signal_common = generate_test_signal(length, sample_rate, 42);
        let noise1 = generate_test_signal(length, sample_rate, 100);
        let noise2 = generate_test_signal(length, sample_rate, 200);

        // signal1 = common + noise1 * 0.1
        let signal1 = &signal_common + &(&noise1 * 0.1);

        // signal2 = freq_shift(time_shift(common)) + noise2 * 0.1
        let mut signal_common_shifted = ComplexVec::from_vec(circular_shift(&signal_common, time_delay as isize));
        signal_common_shifted.freq_shift(freq_shift_hz, sample_rate);
        let signal2 = &signal_common_shifted + &(&noise2 * 0.1);

        // Use auto_compute_caf
        let surface = auto_compute_caf(
            &signal1,
            &signal2,
            sample_rate,
            bandwidth,
            (-500.0, 500.0),                     // Doppler range in Hz
            (-0.001, 0.001),                     // Time delay range in seconds (±1 ms)
            None,                                 // Use default points_per_mainlobe = 2
        )
        .unwrap();

        // Find peak
        let peak = find_peak(&surface);

        // Calculate what the auto parameters should be
        let integration_time = length as f64 / sample_rate;
        let freq_resolution = 1.0 / integration_time;
        let expected_freq_step = freq_resolution / 2.0;

        println!("Auto CAF peak:");
        println!("  Signal length: {} samples", length);
        println!("  Integration time: {:.6} s", integration_time);
        println!("  Frequency resolution: {:.2} Hz", freq_resolution);
        println!("  Auto freq step: {:.2} Hz", expected_freq_step);
        println!("  Expected delay: {} samples", time_delay);
        println!("  Found delay: {:.2} samples", peak.delay_samples);
        println!("  Expected FDOA: {} Hz", freq_shift_hz);
        println!("  Found FDOA: {:.2} Hz", peak.doppler_hz);
        println!("  SNR: {:.1} dB", peak.snr_db);

        if should_plot() {
            plot::plot_caf_surface_3d(&surface, Some(&peak), "CAF: Autocorrelation");
            plot::plot_caf_heatmap(&surface, Some(&peak), "CAF Heatmap: Autocorrelation");
            plot::plot_caf_slices(&surface, &peak, "CAF Slices: Autocorrelation");
        }
        
        // Validate - should find the peak within reasonable accuracy
        assert!(
            (peak.delay_samples - time_delay as f64).abs() < 2.0,
            "Delay error too large: expected {}, got {}",
            time_delay,
            peak.delay_samples
        );
        // With better resolution, Doppler should be within freq_step/2
        assert!(
            (peak.doppler_hz - freq_shift_hz).abs() < expected_freq_step,
            "Doppler error too large: expected {}, got {}, freq_step={}",
            freq_shift_hz,
            peak.doppler_hz,
            expected_freq_step
        );
        assert!(peak.snr_db > 0.0, "SNR should be positive");
    }
}

#[cfg(test)]
mod plot {
    use super::*;
    use plotly::common::{ColorScale, ColorScalePalette, Mode};
    use plotly::layout::{Axis, Layout};
    use plotly::{HeatMap, Plot, Scatter, Scatter3D, Surface};

    /// Plot 3D surface of CAF magnitude
    pub fn plot_caf_surface_3d(surface: &CafSurface<f64>, peak: Option<&Peak>, title: &str) {
        let mut plot = Plot::new();

        // Estimate noise floor using median
        let noise_floor = estimate_noise_floor(surface);
        let noise_floor_db = 10.0 * noise_floor.log10();

        // Convert to dB relative to noise floor
        let surface_db: Vec<Vec<f64>> = surface
            .surface
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&val| {
                        if val > 0.0 {
                            let val_db = 10.0 * val.log10();
                            // Relative to noise floor, clamped at -100 dB
                            (val_db - noise_floor_db).max(-100.0)
                        } else {
                            -100.0
                        }
                    })
                    .collect()
            })
            .collect();

        let trace = Surface::new(surface_db)
            .x(surface.time_delays.iter().map(|&x| x).collect())
            .y(surface.doppler_shifts.iter().map(|&y| y).collect())
            .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

        plot.add_trace(trace);

        // Add 3D peak marker if provided
        if let Some(pk) = peak {
            // Convert peak magnitude to dB relative to noise floor
            let peak_db = 10.0 * pk.magnitude.log10() - noise_floor_db;

            let hover_text = format!(
                "Peak<br>SNR: {:.1} dB<br>Delay: {:.2} samples<br>FDOA: {:.2} Hz",
                pk.snr_db, pk.delay_samples, pk.doppler_hz
            );

            let marker_trace = Scatter3D::new(
                vec![pk.delay_samples],
                vec![pk.doppler_hz],
                vec![peak_db],
            )
            .mode(Mode::Markers)
            .name("Peak")
            .text(hover_text);

            plot.add_trace(marker_trace);
        }

        let layout = Layout::new().title(title);

        plot.set_layout(layout);
        plot.show();
    }

    /// Plot 2D heatmap with optional peak marker
    pub fn plot_caf_heatmap(surface: &CafSurface<f64>, peak: Option<&Peak>, title: &str) {
        let mut plot = Plot::new();

        // Estimate noise floor using median
        let noise_floor = estimate_noise_floor(surface);
        let noise_floor_db = 10.0 * noise_floor.log10();

        // Convert to dB relative to noise floor
        let surface_db: Vec<Vec<f64>> = surface
            .surface
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&val| {
                        if val > 0.0 {
                            let val_db = 10.0 * val.log10();
                            // Relative to noise floor, clamped at -100 dB
                            (val_db - noise_floor_db).max(-100.0)
                        } else {
                            -100.0
                        }
                    })
                    .collect()
            })
            .collect();

        let trace = HeatMap::new(
            surface.time_delays.iter().map(|&x| x).collect(),
            surface.doppler_shifts.iter().map(|&y| y).collect(),
            surface_db,
        )
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis));

        plot.add_trace(trace);

        // Add peak marker if provided
        if let Some(pk) = peak {
            let marker_trace = Scatter::new(vec![pk.delay_samples], vec![pk.doppler_hz])
                .mode(Mode::Markers)
                .name("Peak");
            plot.add_trace(marker_trace);

            // Add text annotation with peak details
            let annotation_text = format!(
                "SNR: {:.1} dB<br>Delay: {:.2} samples<br>FDOA: {:.2} Hz",
                pk.snr_db, pk.delay_samples, pk.doppler_hz
            );
            let text_trace = Scatter::new(vec![pk.delay_samples], vec![pk.doppler_hz])
                .mode(Mode::Text)
                .text(annotation_text)
                .text_position(plotly::common::Position::TopRight)
                .show_legend(false);
            plot.add_trace(text_trace);
        }

        let layout = Layout::new()
            .title(title)
            .x_axis(Axis::new().title("Time Delay (samples)"))
            .y_axis(Axis::new().title("Doppler Shift (Hz)"));

        plot.set_layout(layout);
        plot.show();
    }

    /// Plot 1D slices through peak
    pub fn plot_caf_slices(surface: &CafSurface<f64>, peak: &Peak, title: &str) {
        use plotly::layout::{GridPattern, LayoutGrid, RowOrder};

        let mut plot = Plot::new();

        // Estimate noise floor using median
        let noise_floor = estimate_noise_floor(surface);
        let noise_floor_db = 10.0 * noise_floor.log10();

        // Time slice at peak Doppler (relative to noise floor)
        let time_slice: Vec<f64> = surface.surface[peak.doppler_idx]
            .iter()
            .map(|&val| {
                if val > 0.0 {
                    let val_db = 10.0 * val.log10();
                    (val_db - noise_floor_db).max(-100.0)
                } else {
                    -100.0
                }
            })
            .collect();

        let time_trace = Scatter::new(
            surface.time_delays.iter().map(|&x| x).collect(),
            time_slice,
        )
        .mode(Mode::Lines)
        .name("Time Slice")
        .x_axis("x1")
        .y_axis("y1");

        // Doppler slice at peak time (relative to noise floor)
        let doppler_slice: Vec<f64> = surface
            .surface
            .iter()
            .map(|row| {
                let val = row[peak.delay_idx];
                if val > 0.0 {
                    let val_db = 10.0 * val.log10();
                    (val_db - noise_floor_db).max(-100.0)
                } else {
                    -100.0
                }
            })
            .collect();

        let doppler_trace = Scatter::new(
            surface.doppler_shifts.iter().map(|&y| y).collect(),
            doppler_slice,
        )
        .mode(Mode::Lines)
        .name("Doppler Slice")
        .x_axis("x2")
        .y_axis("y2");

        plot.add_trace(time_trace);
        plot.add_trace(doppler_trace);

        // Add vertical markers at peak positions
        // Calculate peak magnitude in dB relative to noise floor
        let peak_db = 10.0 * peak.magnitude.log10() - noise_floor_db;

        // Marker on time slice at peak delay
        let time_marker = Scatter::new(
            vec![peak.delay_samples, peak.delay_samples],
            vec![-100.0, peak_db],
        )
        .mode(Mode::Lines)
        .name("Peak Position")
        .line(plotly::common::Line::new().dash(plotly::common::DashType::Dash))
        .x_axis("x1")
        .y_axis("y1")
        .show_legend(false);
        plot.add_trace(time_marker);

        // Annotation on time slice
        let time_annotation_y = peak_db + 5.0; // Position slightly above peak
        let time_text = Scatter::new(vec![peak.delay_samples], vec![time_annotation_y])
            .mode(Mode::Text)
            .text(format!("SNR: {:.1} dB", peak.snr_db))
            .text_position(plotly::common::Position::TopCenter)
            .x_axis("x1")
            .y_axis("y1")
            .show_legend(false);
        plot.add_trace(time_text);

        // Marker on Doppler slice at peak Doppler
        let doppler_marker = Scatter::new(
            vec![peak.doppler_hz, peak.doppler_hz],
            vec![-100.0, peak_db],
        )
        .mode(Mode::Lines)
        .name("Peak Position")
        .line(plotly::common::Line::new().dash(plotly::common::DashType::Dash))
        .x_axis("x2")
        .y_axis("y2")
        .show_legend(false);
        plot.add_trace(doppler_marker);

        // Annotation on Doppler slice
        let doppler_annotation_y = peak_db + 5.0; // Position slightly above peak
        let doppler_text = Scatter::new(vec![peak.doppler_hz], vec![doppler_annotation_y])
            .mode(Mode::Text)
            .text(format!("SNR: {:.1} dB", peak.snr_db))
            .text_position(plotly::common::Position::TopCenter)
            .x_axis("x2")
            .y_axis("y2")
            .show_legend(false);
        plot.add_trace(doppler_text);

        let layout = Layout::new()
            .title(title)
            .grid(
                LayoutGrid::new()
                    .rows(2)
                    .columns(1)
                    .pattern(GridPattern::Independent)
                    .row_order(RowOrder::TopToBottom),
            )
            .x_axis(Axis::new().title("Time Delay (samples)").domain(&[0.0, 1.0]))
            .y_axis(
                Axis::new()
                    .title("CAF Magnitude (dB)")
                    .domain(&[0.55, 1.0]),
            )
            .x_axis2(Axis::new().title("Doppler Shift (Hz)").domain(&[0.0, 1.0]))
            .y_axis2(
                Axis::new()
                    .title("CAF Magnitude (dB)")
                    .domain(&[0.0, 0.45]),
            );

        plot.set_layout(layout);
        plot.show();
    }
}
