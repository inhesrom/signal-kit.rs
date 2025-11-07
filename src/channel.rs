use crate::carrier::Carrier;
use crate::awgn::AWGN;
use crate::complex_vec::ComplexVec;
use num_traits::Float;

/// A multi-carrier channel simulator with shared AWGN
///
/// This struct manages multiple carriers that are combined into a single channel with AWGN.
/// Noise is added once to the combined signal (not per-carrier), modeling a realistic
/// transponder or channel scenario.
///
/// # SNR Calculation
///
/// For a multi-carrier channel with a specified noise floor:
/// - Each carrier's SNR = P_i / N₀
/// - Where P_i is the power of carrier i and N₀ is the noise floor
///
/// The user specifies the noise floor (N₀) which is equivalent to specifying the noise
/// power spectral density (in dB or linear units).
///
/// # Example
/// ```ignore
/// let carrier1 = Carrier::new(ModType::_QPSK, 0.1, 0.1, 10.0, 0.35, 1e6, Some(42));
/// let carrier2 = Carrier::new(ModType::_QPSK, 0.1, -0.1, 10.0, 0.35, 1e6, Some(43));
///
/// let mut channel = Channel::new(vec![carrier1, carrier2]);
/// channel.set_noise_floor_db(-100.0); // Set noise floor in dB
///
/// let combined_iq = channel.generate::<f64>(10000);
/// ```
pub struct Channel {
    carriers: Vec<Carrier>,
    noise_floor_db: Option<f64>,  // Noise power (linear units, converted to dB internally)
    seed: Option<u64>,             // Seed for AWGN generator
}

impl Channel {
    /// Create a new channel with multiple carriers
    ///
    /// # Arguments
    /// * `carriers` - Vector of Carrier objects to be combined in this channel
    pub fn new(carriers: Vec<Carrier>) -> Self {
        Channel {
            carriers,
            noise_floor_db: None,
            seed: None,
        }
    }

    /// Set the noise floor in dB
    ///
    /// The noise floor represents the power of the AWGN that will be added to the
    /// combined signal. This is equivalent to the noise power spectral density.
    ///
    /// # Arguments
    /// * `noise_floor_db` - Noise floor in dB (e.g., -100.0 dB)
    pub fn set_noise_floor_db(&mut self, noise_floor_db: f64) {
        self.noise_floor_db = Some(noise_floor_db);
    }

    /// Set the noise floor in linear units
    ///
    /// # Arguments
    /// * `noise_floor_linear` - Noise floor as a linear power value
    pub fn set_noise_floor_linear(&mut self, noise_floor_linear: f64) {
        let noise_floor_db = 10.0 * noise_floor_linear.log10();
        self.noise_floor_db = Some(noise_floor_db);
    }

    /// Set the seed for reproducible AWGN generation
    ///
    /// # Arguments
    /// * `seed` - Seed value for the AWGN random number generator
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = Some(seed);
    }

    /// Calculate the required noise power based on the noise floor
    ///
    /// Converts the dB noise floor to linear power units.
    fn noise_floor_to_power(&self) -> f64 {
        match self.noise_floor_db {
            Some(db) => 10.0_f64.powf(db / 10.0),
            None => panic!("Noise floor must be set before generating channel. Use set_noise_floor_db() or set_noise_floor_linear()"),
        }
    }

    /// Generate combined carrier signal with shared AWGN
    ///
    /// # Process
    /// 1. Generate clean (noise-free) signal for each carrier
    /// 2. Combine all carriers by summing samples
    /// 3. Add AWGN with power equal to the set noise floor
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    /// ComplexVec<T> containing the combined signal with AWGN
    pub fn generate<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        if self.carriers.is_empty() {
            panic!("Channel must have at least one carrier");
        }

        // Generate clean signals from all carriers
        let mut combined = ComplexVec::new();
        for carrier in &self.carriers {
            let clean_signal = carrier.generate_clean::<T>(num_samples);
            if combined.len() == 0 {
                combined = clean_signal;
            } else {
                combined = combined + clean_signal;
            }
        }

        // Get noise power from the noise floor
        let noise_power = self.noise_floor_to_power();

        // Generate AWGN with the specified noise power
        // Use a fixed seed if provided, otherwise use entropy
        let sample_rate_hz = self.carriers[0].get_sample_rate();
        let mut awgn = match self.seed {
            Some(s) => AWGN::new_from_seed(sample_rate_hz, num_samples, noise_power, s),
            None => AWGN::new_from_entropy(sample_rate_hz, num_samples, noise_power),
        };

        let noise = awgn.generate_block::<T>();
        let result = combined + noise;

        result
    }

    /// Generate combined carrier signal without noise
    ///
    /// Useful for analysis or when you want to add noise separately.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    /// ComplexVec<T> containing the combined clean signal
    pub fn generate_clean<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        if self.carriers.is_empty() {
            panic!("Channel must have at least one carrier");
        }

        // Generate and combine clean signals from all carriers
        let mut combined = ComplexVec::new();
        for carrier in &self.carriers {
            let clean_signal = carrier.generate_clean::<T>(num_samples);
            if combined.len() == 0 {
                combined = clean_signal;
            } else {
                combined = combined + clean_signal;
            }
        }

        combined
    }

    /// Get a reference to the carriers in this channel
    pub fn carriers(&self) -> &[Carrier] {
        &self.carriers
    }

    /// Get the number of carriers in this channel
    pub fn num_carriers(&self) -> usize {
        self.carriers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mod_type::ModType;

    #[test]
    fn test_channel_creation() {
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let channel = Channel::new(vec![carrier1]);
        assert_eq!(channel.num_carriers(), 1);
    }

    #[test]
    fn test_channel_single_carrier() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let mut channel = Channel::new(vec![carrier]);
        channel.set_noise_floor_db(-100.0);

        let result = channel.generate::<f64>(1000);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_channel_two_carriers() {
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let carrier2 = Carrier::new(
            ModType::_QPSK,
            0.1,
            -0.1,
            15.0,
            0.35,
            1e6,
            Some(43),
        );

        let mut channel = Channel::new(vec![carrier1, carrier2]);
        channel.set_noise_floor_db(-100.0);

        let result = channel.generate::<f64>(1000);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_channel_generate_clean() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let channel = Channel::new(vec![carrier]);

        let clean = channel.generate_clean::<f64>(1000);
        assert_eq!(clean.len(), 1000);
    }

    #[test]
    fn test_channel_noise_floor_linear() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let mut channel = Channel::new(vec![carrier]);

        // Set noise floor as linear value
        channel.set_noise_floor_linear(0.01);

        let result = channel.generate::<f64>(1000);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_channel_with_seed() {
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let carrier2 = Carrier::new(
            ModType::_QPSK,
            0.1,
            -0.1,
            10.0,
            0.35,
            1e6,
            Some(43),
        );

        // Generate with same seed - should get same result
        let mut channel1 = Channel::new(vec![carrier1.clone(), carrier2.clone()]);
        channel1.set_noise_floor_db(-100.0);
        channel1.set_seed(999);

        let mut channel2 = Channel::new(vec![carrier1, carrier2]);
        channel2.set_noise_floor_db(-100.0);
        channel2.set_seed(999);

        let result1 = channel1.generate::<f64>(1000);
        let result2 = channel2.generate::<f64>(1000);

        // Results should be identical with same seed
        for (s1, s2) in result1.iter().zip(result2.iter()) {
            assert!((s1.re - s2.re).abs() < 1e-10);
            assert!((s1.im - s2.im).abs() < 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "Noise floor must be set")]
    fn test_channel_no_noise_floor() {
        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            1e6,
            Some(42),
        );
        let channel = Channel::new(vec![carrier]);

        // Should panic because noise floor was never set
        let _ = channel.generate::<f64>(1000);
    }

    #[test]
    #[should_panic(expected = "at least one carrier")]
    fn test_channel_empty() {
        let channel = Channel::new(vec![]);
        let _ = channel.generate::<f64>(1000);
    }

    #[test]
    fn test_channel_spectrum_with_snr_verification() {
        use std::env;
        use crate::welch::welch;
        use crate::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        // Check if plotting is enabled
        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());

        let sample_rate = 1e6;
        let num_samples = (2.0).powf(20.0) as usize;

        // Create two carriers with different target SNRs
        // To get different SNRs with a shared noise floor, they need different POWERS
        // We achieve this by using different modulation types with different constellation sizes:
        // - QPSK (4 symbols): lower power
        // - 16QAM (16 symbols): higher power due to larger constellation
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.05,     // 50 kHz bandwidth (narrow)
            0.15,     // 150 kHz center freq
            10.0,     // Target 10 dB SNR (lower power)
            0.35,     // RRC rolloff
            sample_rate,
            Some(42),
        );

        let carrier2 = Carrier::new(
            ModType::_16QAM,  // 16 symbols = higher average power than QPSK
            0.10,     // 100 kHz bandwidth (wider)
            -0.20,    // -200 kHz center freq
            5.0,      // Target 5 dB SNR (higher power, lower SNR need)
            0.35,
            sample_rate,
            Some(43),
        );

        // --- Approach: Noise floor first, then calculate required signal powers ---
        // This models a real transponder: the noise floor is a channel property,
        // and we design carriers to achieve specific SNRs by adjusting transmit power.

        // Step 1: Define target SNRs and set a baseline noise floor
        let target_snr1_db = 10.0;  // dB
        let target_snr2_db = 5.0;   // dB
        let noise_floor_db = -85.0; // dB (arbitrary baseline for the channel)
        let noise_floor = 10.0_f64.powf(noise_floor_db / 10.0);

        println!("\n=== Channel Spectrum: Two Carriers with Different SNRs ===");
        println!("Target SNR1: {:.1} dB", target_snr1_db);
        println!("Target SNR2: {:.1} dB", target_snr2_db);
        println!("Noise floor: {:.6} dB ({:.2e} linear)", noise_floor_db, noise_floor);

        // Step 2: Calculate required signal powers to achieve target SNRs
        // SNR = P_signal / P_noise  =>  P_signal = SNR * P_noise
        let snr1_linear = 10.0_f64.powf(target_snr1_db / 10.0);
        let snr2_linear = 10.0_f64.powf(target_snr2_db / 10.0);
        let required_power1 = snr1_linear * noise_floor;
        let required_power2 = snr2_linear * noise_floor;

        println!("Required power for SNR1: {:.6} dB", 10.0 * required_power1.log10());
        println!("Required power for SNR2: {:.6} dB", 10.0 * required_power2.log10());

        // Step 3: Generate clean carriers and measure their unscaled powers
        let clean1_unscaled = carrier1.generate_clean::<f64>(num_samples);
        let clean2_unscaled = carrier2.generate_clean::<f64>(num_samples);
        let power1_unscaled = measure_signal_power(&clean1_unscaled);
        let power2_unscaled = measure_signal_power(&clean2_unscaled);

        println!("Carrier 1 unscaled power: {:.6} dB", 10.0 * power1_unscaled.log10());
        println!("Carrier 2 unscaled power: {:.6} dB", 10.0 * power2_unscaled.log10());

        // Step 4: Calculate scaling factors to achieve required powers
        let scale1 = (required_power1 / power1_unscaled).sqrt(); // sqrt because power = amplitude^2
        let scale2 = (required_power2 / power2_unscaled).sqrt();

        // Step 5: Apply scaling to both carriers
        let mut clean1 = clean1_unscaled.clone();
        let mut clean2 = clean2_unscaled.clone();
        for sample in clean1.iter_mut() {
            *sample = *sample * scale1;
        }
        for sample in clean2.iter_mut() {
            *sample = *sample * scale2;
        }

        // Step 6: Verify actual powers after scaling
        let power1 = measure_signal_power(&clean1);
        let power2 = measure_signal_power(&clean2);

        println!("\nActual scaled carrier powers:");
        println!("Carrier 1 power: {:.6} dB", 10.0 * power1.log10());
        println!("Carrier 2 power: {:.6} dB", 10.0 * power2.log10());
        println!("Power ratio (C1/C2): {:.2} dB", 10.0 * (power1 / power2).log10());

        // Step 7: Verify SNRs will be achieved
        let snr1_actual = 10.0 * (power1 / noise_floor).log10();
        let snr2_actual = 10.0 * (power2 / noise_floor).log10();
        println!("\nExpected SNRs with noise floor:");
        println!("Carrier 1 SNR: {:.2} dB (target {:.1} dB)", snr1_actual, target_snr1_db);
        println!("Carrier 2 SNR: {:.2} dB (target {:.1} dB)", snr2_actual, target_snr2_db);
        println!("SNR difference (C1 - C2): {:.2} dB", snr1_actual - snr2_actual);

        // Combine the scaled clean signals manually
        // (bypassing Channel to use the pre-scaled carriers)
        let combined_clean = clean1 + clean2;
        assert_eq!(combined_clean.len(), num_samples);

        // Add shared AWGN to the combined signal
        use crate::awgn::AWGN;

        // Use the noise floor directly as the AWGN power
        let noise_power = noise_floor;

        println!("\nNoise specification:");
        println!("Noise floor: {:.6} dB ({:.2e} linear)", noise_floor_db, noise_floor);

        let mut awgn = AWGN::new_from_seed(sample_rate, num_samples, noise_power, 999);
        let noise = awgn.generate_block::<f64>();
        let actual_noise_power = measure_signal_power(&noise);
        let combined = combined_clean + noise;

        // Verify actual SNR using measured noise power
        let actual_snr1 = 10.0 * (power1 / actual_noise_power).log10();
        let actual_snr2 = 10.0 * (power2 / actual_noise_power).log10();
        println!("\nVerification (measured noise power):");
        println!("Actual noise power: {:.6} dB ({:.2e} linear)", 10.0 * actual_noise_power.log10(), actual_noise_power);
        println!("Carrier 1 actual SNR: {:.2} dB (target 10.0 dB)", actual_snr1);
        println!("Carrier 2 actual SNR: {:.2} dB (target 5.0 dB)", actual_snr2);

        // Only plot if enabled
        if plot.to_lowercase() == "true" {
            // Convert ComplexVec to Vec<Complex<f64>> for Welch processing
            let signal: Vec<_> = (0..combined.len()).map(|i| combined[i]).collect();

            // Compute Welch PSD with appropriate parameters
            let (freqs, psd) = welch(
                &signal,
                sample_rate,
                1024,                    // 1024-point segments
                None,                    // 50% overlap (default)
                None,                    // No zero-padding (default)
                WindowType::Hann,        // Hann window
                None,                    // Mean averaging (default)
            );

            // Convert PSD to dB scale
            let psd_db: Vec<f64> = vector_ops::to_db(&psd);

            // Plot Welch PSD
            plot_spectrum(&freqs, &psd_db, "Channel Spectrum: Two Carriers (10dB and 5dB SNR) with Shared AWGN");
        } else {
            println!("Skipping spectrum plot (set TEST_PLOT=true to enable)");
        }
    }

    /// Helper function to measure signal power
    fn measure_signal_power<T: num_traits::Float>(signal: &crate::complex_vec::ComplexVec<T>) -> f64 {
        let mut sum_power = 0.0;
        for sample in signal.iter() {
            let power = (sample.re.to_f64().unwrap()).powi(2)
                + (sample.im.to_f64().unwrap()).powi(2);
            sum_power += power;
        }
        sum_power / signal.len() as f64
    }
}
