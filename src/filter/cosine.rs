use num_complex::Complex;
use rustfft::FftPlanner;

/// Create a cosine-tapered lowpass filter (alternative to Butterworth)
/// 
/// This creates a smooth rolloff using a raised cosine taper, which is commonly
/// used in real digitizers. It has **NO DC suppression** and a gentle, smooth rolloff.
///
/// The filter has three regions:
/// - **Passband** (0 to passband_end): Unity gain (1.0), including DC
/// - **Transition** (passband_end to stopband_start): Raised cosine taper
/// - **Stopband** (stopband_start to Nyquist): Zero gain (0.0)
///
/// # Arguments
/// * `num_samples` - Number of frequency bins (must match signal length)
/// * `passband_end` - End of flat passband (0.0 to 0.5, normalized frequency)
/// * `stopband_start` - Start of stopband (0.0 to 0.5, must be > passband_end)
///
/// # Returns
/// Vector of frequency-domain filter coefficients
///
/// # Example
/// ```ignore
/// // Passband: 0-40% of Nyquist (flat, unity gain)
/// // Transition: 40-48% of Nyquist (smooth cosine rolloff)
/// // Stopband: 48-50% of Nyquist (zero gain)
/// let filter = create_cosine_taper_filter(1024, 0.40, 0.48);
/// assert_eq!(filter[0], 1.0);  // DC has unity gain
/// ```
pub fn create_cosine_taper_filter(num_samples: usize, passband_end: f64, stopband_start: f64) -> Vec<f64> {
    let mut filter_response = vec![1.0; num_samples];
    
    for i in 0..num_samples {
        // Convert index to normalized frequency [-0.5, 0.5]
        let f = if i < num_samples / 2 {
            i as f64 / num_samples as f64
        } else {
            (i as f64 - num_samples as f64) / num_samples as f64
        };
        
        let norm_freq = f.abs();
        
        if norm_freq <= passband_end {
            // Passband: unity gain (including DC at i=0)
            filter_response[i] = 1.0;
        } else if norm_freq >= stopband_start {
            // Stopband: zero gain
            filter_response[i] = 0.0;
        } else {
            // Transition: raised cosine taper
            // Smoothly transitions from 1.0 to 0.0 using cosine
            let transition_width = stopband_start - passband_end;
            let position = (norm_freq - passband_end) / transition_width;
            filter_response[i] = 0.5 * (1.0 + (std::f64::consts::PI * position).cos());
        }
    }
    
    filter_response
}

/// Apply cosine-tapered lowpass filter (alternative to Butterworth)
///
/// This filter has:
/// - **Unity gain at DC** (NO DC suppression!)
/// - Flat passband up to passband_end
/// - Smooth raised cosine rolloff in transition band
/// - Zero gain in stopband
///
/// Works with any sample rate (uses normalized frequency).
///
/// # Arguments
/// * `signal` - Complex IQ signal to filter (modified in-place)
/// * `passband_end` - End of passband (0.0 to 0.5, normalized)
/// * `stopband_start` - Start of stopband (0.0 to 0.5, must be > passband_end)
///
/// # Example
/// ```ignore
/// use signal_kit::filter::cosine::apply_cosine_taper_filter;
/// use num_complex::Complex;
///
/// let mut iq = vec![Complex::new(1.0, 0.5); 10000];
///
/// // Passband: 0 to 40% of Nyquist (flat)
/// // Transition: 40% to 48% of Nyquist (smooth rolloff)  
/// // Stopband: 48% to 50% (Nyquist)
/// apply_cosine_taper_filter(&mut iq, 0.40, 0.48);
/// ```
///
/// # Notes
/// - At 1 MHz sample rate with passband_end=0.40: flat up to 200 kHz
/// - At 10 MHz sample rate with passband_end=0.40: flat up to 2 MHz
/// - Transition width controls rolloff steepness
pub fn apply_cosine_taper_filter(signal: &mut [Complex<f64>], passband_end: f64, stopband_start: f64) {
    let n = signal.len();
    let filter = create_cosine_taper_filter(n, passband_end, stopband_start);
    
    // FFT to frequency domain
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(signal);
    
    // Apply filter in frequency domain
    for (i, sample) in signal.iter_mut().enumerate() {
        *sample = *sample * filter[i];
    }
    
    // IFFT back to time domain
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(signal);
    
    // Normalize (rustfft doesn't normalize IFFT)
    let scale = 1.0 / (n as f64);
    for sample in signal.iter_mut() {
        *sample = *sample * scale;
    }
}

/// Convenience: Cosine taper digitizer rolloff (gentle, realistic)
///
/// Passband: 0-42% of Nyquist (flat, including DC)
/// Transition: 42-48% of Nyquist (smooth rolloff)
/// Stopband: 48-50% of Nyquist (attenuated)
///
/// This mimics realistic digitizer anti-aliasing behavior without DC suppression.
///
/// # Example
/// ```ignore
/// use signal_kit::filter::cosine::apply_cosine_taper_digitizer;
/// 
/// let mut iq_samples = vec![Complex::new(1.0, 0.0); 10000];
/// apply_cosine_taper_digitizer(&mut iq_samples);
/// ```
pub fn apply_cosine_taper_digitizer(signal: &mut [Complex<f64>]) {
    apply_cosine_taper_filter(signal, 0.42, 0.48);
}

/// Convenience: Aggressive cosine taper rolloff
///
/// Passband: 0-38% of Nyquist  
/// Transition: 38-45% of Nyquist
/// Stopband: 45-50% of Nyquist
///
/// Steeper rolloff than standard digitizer, but still no DC suppression.
pub fn apply_cosine_taper_aggressive(signal: &mut [Complex<f64>]) {
    apply_cosine_taper_filter(signal, 0.38, 0.45);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_creation() {
        let filter = create_cosine_taper_filter(1024, 0.40, 0.48);
        assert_eq!(filter.len(), 1024);

        // DC should be 1.0 (unity gain, NO suppression)
        assert_eq!(filter[0], 1.0);

        // Low frequencies in passband should be 1.0
        for i in 1..100 {
            let f = i as f64 / 1024.0;
            if f < 0.40 {
                assert_eq!(filter[i], 1.0, "Passband bin {} should be 1.0", i);
            }
        }

        // High frequencies in stopband should be 0.0
        let stopband_bin = (0.48 * 1024.0) as usize;
        assert!(filter[stopband_bin] < 0.01, "Stopband should be near 0");
    }

    #[test]
    fn test_dc_preserved() {
        // Create DC signal
        let mut signal: Vec<Complex<f64>> = vec![Complex::new(1.0, 1.0); 1024];

        // Apply filter
        apply_cosine_taper_filter(&mut signal, 0.40, 0.48);

        // DC should pass through with unity gain
        for sample in signal.iter() {
            assert!((sample.re - 1.0).abs() < 0.01, "Real part should be preserved");
            assert!((sample.im - 1.0).abs() < 0.01, "Imag part should be preserved");
        }
    }

    #[test]
    fn test_cosine_taper_digitizer() {
        let mut signal = vec![Complex::new(1.0, 0.5); 1024];
        apply_cosine_taper_digitizer(&mut signal);
        assert_eq!(signal.len(), 1024);
    }

    #[test]
    fn test_transition_band() {
        let filter = create_cosine_taper_filter(10000, 0.40, 0.48);
        
        // Check that transition is smooth (monotonic decreasing)
        let start_bin = (0.40 * 10000.0) as usize;
        let end_bin = (0.48 * 10000.0) as usize;
        
        for i in start_bin..(end_bin - 1) {
            assert!(filter[i] >= filter[i + 1], 
                "Transition should be monotonic: bin {} = {}, bin {} = {}", 
                i, filter[i], i+1, filter[i+1]);
        }
    }
}
