#![allow(dead_code)]

use crate::mod_type::ModType;
use crate::psk_carrier::PskCarrier;
use crate::cw::CW;
use crate::awgn::AWGN;
use crate::complex_vec::ComplexVec;
use num_traits::Float;

/// A high-level carrier generator that combines modulation, frequency shifting, and noise
///
/// Allows easy creation of carriers with specified bandwidth, center frequency, SNR, and modulation type.
/// Supports PSK/QAM modulation, FSK, and CW (continuous wave) unmodulated signals.
///
/// # Example
/// ```ignore
/// let mut carrier = Carrier::new(
///     ModType::_QPSK,      // Modulation type
///     0.1,                  // Normalized bandwidth (0.0-1.0)
///     0.1,                  // Normalized center frequency (-0.5 to 0.5)
///     10.0,                 // SNR in dB
///     0.35,                 // RRC rolloff factor (0.0-1.0)
///     1e6,                  // Sample rate in Hz
///     Some(42),             // Optional seed for reproducibility
/// );
/// let iq_samples = carrier.generate(1000);
/// ```
pub struct Carrier {
    modulation: ModType,
    bandwidth: f64,              // Normalized 0.0-1.0 (occupied_bw / sample_rate)
    center_freq: f64,            // Normalized -0.5 to 0.5
    snr_db: f64,                 // Target SNR in dB
    rolloff: f64,                // RRC rolloff factor (0.0-1.0)
    sample_rate_hz: f64,         // Sample rate in Hz
    seed: Option<u64>,           // Optional seed for reproducibility
}

impl Carrier {
    /// Create a new carrier with specified parameters
    ///
    /// # Arguments
    /// * `modulation` - ModType enum (BPSK, QPSK, 8PSK, FSK, CW, etc.)
    /// * `bandwidth` - Normalized occupied bandwidth (0.0-1.0)
    ///   - Calculated as: (symbol_rate * (1 + rolloff)) / sample_rate
    /// * `center_freq` - Normalized center frequency (-0.5 to 0.5)
    /// * `snr_db` - Target signal-to-noise ratio in dB
    /// * `rolloff` - RRC rolloff factor (0.0-1.0), typically 0.35
    /// * `sample_rate_hz` - Sample rate in Hz
    /// * `seed` - Optional seed for reproducible bit generation (None uses entropy)
    pub fn new(
        modulation: ModType,
        bandwidth: f64,
        center_freq: f64,
        snr_db: f64,
        rolloff: f64,
        sample_rate_hz: f64,
        seed: Option<u64>,
    ) -> Self {
        assert!(
            bandwidth > 0.0 && bandwidth <= 1.0,
            "bandwidth must be in range (0.0, 1.0]"
        );
        assert!(
            center_freq >= -0.5 && center_freq <= 0.5,
            "center_freq must be in range [-0.5, 0.5]"
        );
        assert!(
            rolloff >= 0.0 && rolloff <= 1.0,
            "rolloff must be in range [0.0, 1.0]"
        );
        assert!(sample_rate_hz > 0.0, "sample_rate_hz must be positive");

        Carrier {
            modulation,
            bandwidth,
            center_freq,
            snr_db,
            rolloff,
            sample_rate_hz,
            seed,
        }
    }

    /// Generate a specified number of samples from this carrier
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    /// ComplexVec<T> containing the generated samples
    pub fn generate<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        // Calculate symbol rate from bandwidth
        // bandwidth = (symbol_rate * (1 + rolloff)) / sample_rate
        // symbol_rate = (bandwidth * sample_rate) / (1 + rolloff)
        let symbol_rate = (self.bandwidth * self.sample_rate_hz) / (1.0 + self.rolloff);

        // Generate baseband signal using appropriate carrier type
        let mut iq_samples = match self.modulation {
            ModType::_BPSK | ModType::_QPSK | ModType::_8PSK | ModType::_16APSK
            | ModType::_16QAM | ModType::_32QAM | ModType::_64QAM => {
                self.generate_psk_based(symbol_rate, num_samples)
            }
            // FSK handling would go here if we support it
            ModType::_CW => {
                self.generate_cw(num_samples)
            }
        };

        // Frequency shift to center frequency (if not baseband)
        if self.center_freq.abs() > 1e-9 {
            let freq_offset_hz = self.center_freq * self.sample_rate_hz;
            iq_samples.freq_shift(freq_offset_hz, self.sample_rate_hz);
        }

        // Measure signal power
        let signal_power = self.measure_power(&iq_samples);

        // Add AWGN to achieve target SNR
        // SNR_dB = 10 * log10(signal_power / noise_power)
        // noise_power = signal_power / 10^(SNR_dB/10)
        let snr_linear = 10.0_f64.powf(self.snr_db / 10.0);
        let noise_power = signal_power / snr_linear;

        let mut awgn = match self.seed {
            Some(s) => AWGN::new_from_seed(self.sample_rate_hz, num_samples, noise_power, s),
            None => AWGN::new_from_entropy(self.sample_rate_hz, num_samples, noise_power),
        };

        let noise = awgn.generate_block::<T>();
        let result = iq_samples + noise;

        result
    }

    /// Generate PSK-based modulation (BPSK, QPSK, 8PSK, etc.)
    fn generate_psk_based<T: Float>(&self, symbol_rate: f64, num_samples: usize) -> ComplexVec<T> {
        // Estimate block size (will be refined by PSK carrier)
        let block_size = (num_samples as f64 * symbol_rate / self.sample_rate_hz).ceil() as usize;

        let mut psk = PskCarrier::<T>::new(
            T::from(self.sample_rate_hz).unwrap(),
            T::from(symbol_rate).unwrap(),
            self.modulation,
            T::from(self.rolloff).unwrap(),
            block_size,
            128, // Default filter taps
            self.seed,
        );

        // Generate blocks until we have enough samples
        let mut result = ComplexVec::new();
        while result.len() < num_samples {
            let block = psk.generate_block();
            result.extend(block.iter().cloned());
        }

        // Trim to exact sample count
        let vec: Vec<_> = result
            .iter()
            .take(num_samples)
            .cloned()
            .collect();
        ComplexVec::from_vec(vec)
    }

    /// Generate CW (unmodulated continuous wave)
    fn generate_cw<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        // CW is just a constant magnitude signal at baseband (0 Hz)
        // Generate as if it's a CW at frequency 0
        let mut cw = CW::new(0.0, self.sample_rate_hz, num_samples);
        let samples = cw.generate_block::<T>();
        ComplexVec::from_vec(samples)
    }

    /// Measure the average power of a signal
    fn measure_power<T: Float>(&self, signal: &ComplexVec<T>) -> f64 {
        let mut sum_power = 0.0;
        for sample in signal.iter() {
            let power = (sample.re.to_f64().unwrap()).powi(2)
                + (sample.im.to_f64().unwrap()).powi(2);
            sum_power += power;
        }
        sum_power / signal.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_carrier_qpsk_creation() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        assert_eq!(carrier.modulation, ModType::_QPSK);
        assert_eq!(carrier.snr_db, 10.0);
    }

    #[test]
    fn test_carrier_generate_qpsk() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.0,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let iq = carrier.generate::<f64>(1000);
        assert_eq!(iq.len(), 1000);
    }

    #[test]
    fn test_carrier_generate_cw() {
        let carrier = Carrier::new(
            ModType::_CW,
            0.1,
            0.0,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let iq = carrier.generate::<f64>(1000);
        assert_eq!(iq.len(), 1000);
    }

    #[test]
    fn test_carrier_with_frequency_shift() {
        let carrier = Carrier::new(
            ModType::_CW,
            0.1,
            0.25, // Shift to 250 kHz
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let iq = carrier.generate::<f64>(100);
        assert_eq!(iq.len(), 100);
    }

    #[test]
    fn test_invalid_bandwidth() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Carrier::new(
                ModType::_QPSK,
                1.5, // Invalid: > 1.0
                0.0,
                10.0,
                0.35,
                1e6,
                None,
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_center_freq() {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Carrier::new(
                ModType::_QPSK,
                0.1,
                0.6, // Invalid: > 0.5
                10.0,
                0.35,
                1e6,
                None,
            );
        }));
        assert!(result.is_err());
    }

    #[test]
    fn test_carrier_combination_spectrum() {
        use std::env;
        use crate::welch::welch;
        use crate::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        // Check if plotting is enabled
        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping carrier combination spectrum plot (set TEST_PLOT=true to enable)");
            return;
        }

        println!("\n=== Two Carrier Combination PSD (Welch Method) ===");

        let sample_rate = 1e6;

        // Create two carriers with contrasting characteristics
        // Carrier 1: Strong, narrow bandwidth signal
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.05,     // 50 kHz bandwidth (narrow)
            0.15,     // 150 kHz center freq
            25.0,     // 25 dB SNR (high power)
            0.35,     // RRC rolloff
            sample_rate,
            Some(42),
        );

        // Carrier 2: Weak, wide bandwidth signal (interference/noise)
        let carrier2 = Carrier::new(
            ModType::_8PSK,
            0.20,     // 200 kHz bandwidth (wide)
            -0.25,    // -250 kHz center freq
            5.0,      // 5 dB SNR (low power)
            0.35,
            sample_rate,
            Some(43),
        );

        // Generate more samples for better Welch averaging
        let num_samples = (2.0).powf(20.0) as usize;
        let iq1 = carrier1.generate::<f64>(num_samples);
        let iq2 = carrier2.generate::<f64>(num_samples);

        // Combine carriers (simulating transponder/SDR channel)
        let combined = iq1 + iq2;

        assert_eq!(combined.len(), num_samples);

        // Convert ComplexVec to Vec<Complex<f64>> for Welch processing
        let signal: Vec<_> = (0..combined.len()).map(|i| combined[i]).collect();

        // Compute Welch PSD with appropriate parameters for good resolution
        let (freqs, psd) = welch(
            &signal,
            sample_rate,
            1024,                    // 1024-point segments (976 Hz resolution)
            None,                    // 50% overlap (default)
            None,                    // No zero-padding (default)
            WindowType::Hann,        // Hann window (standard)
            None,                    // Mean averaging (default)
        );

        // Convert PSD to dB scale
        let psd_db: Vec<f64> = vector_ops::to_db(&psd);

        // Plot Welch PSD
        plot_spectrum(&freqs, &psd_db, "Two Carrier PSD (Welch Method)");
    }
}
