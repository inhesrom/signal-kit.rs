use crate::generate::carrier::Carrier;
use crate::generate::awgn::AWGN;
use crate::generate::impairment::Impairment;
use crate::complex_vec::ComplexVec;
use num_traits::Float;

/// A multi-carrier channel simulator with shared AWGN and configurable impairments
///
/// This struct manages multiple carriers that are combined into a single channel with AWGN.
/// Noise is added once to the combined signal (not per-carrier), modeling a realistic
/// transponder or channel scenario. Channel impairments (digitizer droop, frequency-dependent
/// amplitude variations, etc.) can be applied to the complete combined signal.
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
/// # Example with Impairments
/// ```ignore
/// use signal_kit::{Carrier, Channel, ModType, generate::Impairment};
///
/// let carrier1 = Carrier::new(ModType::_QPSK, 0.1, 0.1, 10.0, 0.35, 1e6, Some(42));
/// let carrier2 = Carrier::new(ModType::_QPSK, 0.1, -0.1, 10.0, 0.35, 1e6, Some(43));
///
/// let mut channel = Channel::new(vec![carrier1, carrier2]);
/// channel.set_noise_floor_db(-100.0);
/// channel.add_impairment(Impairment::DigitizerDroopAD9361);
/// channel.add_impairment(Impairment::FrequencyVariation {
///     amplitude_db: 1.0,
///     cycles: 3.0,
///     phase_offset: 0.0,
/// });
///
/// let combined_iq = channel.generate::<f64>(10000);
/// ```
pub struct Channel {
    carriers: Vec<Carrier>,
    noise_floor_db: Option<f64>,  // Noise power (linear units, converted to dB internally)
    seed: Option<u64>,             // Seed for AWGN generator
    impairments: Vec<Impairment>,  // Channel impairments to apply after noise
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
            impairments: Vec::new(),
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

    /// Add a single impairment to the channel
    ///
    /// Impairments are applied after noise addition, to the complete combined signal.
    /// Multiple impairments are applied in the order they were added.
    ///
    /// # Arguments
    /// * `impairment` - Channel impairment to apply
    pub fn add_impairment(&mut self, impairment: Impairment) -> &mut Self {
        self.impairments.push(impairment);
        self
    }

    /// Set all impairments for this channel
    ///
    /// Replaces any previously added impairments with the provided vector.
    ///
    /// # Arguments
    /// * `impairments` - Vector of impairments to apply
    pub fn with_impairments(&mut self, impairments: Vec<Impairment>) -> &mut Self {
        self.impairments = impairments;
        self
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
    /// 2. Scale each carrier to achieve its target SNR relative to the noise floor
    /// 3. Combine all scaled carriers by summing samples
    /// 4. Add AWGN with power equal to the set noise floor
    ///
    /// Each carrier's SNR is achieved by scaling its power according to:
    /// - Required Power = 10^(SNR_dB/10) × Noise_Floor
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    /// ComplexVec<T> containing the combined signal with AWGN
    ///
    /// # Panics
    /// Panics if no carriers are present or if noise floor is not set
    pub fn generate<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        if self.carriers.is_empty() {
            panic!("Channel must have at least one carrier");
        }

        // Get noise power from the noise floor (required for SNR scaling)
        let noise_power = self.noise_floor_to_power();

        // Generate, scale, and combine carriers to achieve target SNRs
        let mut combined = ComplexVec::new();
        for carrier in &self.carriers {
            // Generate clean signal
            let clean_signal = carrier.generate_clean::<T>(num_samples);

            // Calculate oversample rate for power measurement (accounts for in-band power)
            let oversample_rate = 1.0 / carrier.bandwidth;

            // Scale to achieve target SNR
            // SNR = P_signal / P_noise  =>  P_signal = SNR_linear * P_noise
            let snr_linear = 10.0_f64.powf(carrier.snr_db / 10.0);
            let target_power = snr_linear * noise_power;

            // Scale the carrier to target power
            let scaled_signal = clean_signal.scale_to_power(target_power, Some(oversample_rate));

            // Add to combined signal
            if combined.len() == 0 {
                combined = scaled_signal;
            } else {
                combined = combined + scaled_signal;
            }
        }

        // Generate AWGN with the specified noise power
        // Use a fixed seed if provided, otherwise use entropy
        let sample_rate_hz = self.carriers[0].get_sample_rate();
        let mut awgn = match self.seed {
            Some(s) => AWGN::new_from_seed(sample_rate_hz, num_samples, noise_power, s),
            None => AWGN::new_from_entropy(sample_rate_hz, num_samples, noise_power),
        };

        let noise = awgn.generate_block::<T>();
        let mut result = combined + noise;

        // Apply impairments to the complete signal (after noise addition)
        if !self.impairments.is_empty() {
            // For impairments, we need to work with f64 samples (current limitation)
            // Convert to f64, apply impairments, convert back
            let mut result_f64: Vec<num_complex::Complex<f64>> = result
                .iter()
                .map(|c| {
                    num_complex::Complex::new(c.re.to_f64().unwrap(), c.im.to_f64().unwrap())
                })
                .collect();

            // Apply each impairment in order
            for impairment in &self.impairments {
                impairment.apply(&mut result_f64);
            }

            // Convert back to original type
            result = ComplexVec::from_vec(
                result_f64
                    .into_iter()
                    .map(|c| {
                        num_complex::Complex::new(
                            T::from(c.re).unwrap(),
                            T::from(c.im).unwrap(),
                        )
                    })
                    .collect(),
            );
        }

        result
    }

    /// Generate combined carrier signal without noise
    ///
    /// Each carrier is scaled to achieve its target SNR relative to the noise floor.
    /// This allows analysis of the clean signal with correct relative power levels.
    ///
    /// # Arguments
    /// * `num_samples` - Number of samples to generate
    ///
    /// # Returns
    /// ComplexVec<T> containing the combined clean signal with proper SNR scaling
    ///
    /// # Panics
    /// Panics if no carriers are present or if noise floor is not set
    pub fn generate_clean<T: Float>(&self, num_samples: usize) -> ComplexVec<T> {
        if self.carriers.is_empty() {
            panic!("Channel must have at least one carrier");
        }

        // Get noise power from the noise floor (required for SNR scaling)
        let noise_power = self.noise_floor_to_power();

        // Generate, scale, and combine clean signals from all carriers
        let mut combined = ComplexVec::new();
        for carrier in &self.carriers {
            // Generate clean signal
            let clean_signal = carrier.generate_clean::<T>(num_samples);

            // Calculate oversample rate for power measurement
            let oversample_rate = 1.0 / carrier.bandwidth;

            // Scale to achieve target SNR
            let snr_linear = 10.0_f64.powf(carrier.snr_db / 10.0);
            let target_power = snr_linear * noise_power;

            // Scale the carrier to target power
            let scaled_signal = clean_signal.scale_to_power(target_power, Some(oversample_rate));

            // Add to combined signal
            if combined.len() == 0 {
                combined = scaled_signal;
            } else {
                combined = combined + scaled_signal;
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
        let mut channel = Channel::new(vec![carrier]);
        channel.set_noise_floor_db(-100.0);

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
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        // Check if plotting is enabled
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());

        let sample_rate = 1e6;
        let num_samples = (2.0).powf(20.0) as usize;

        // Create two carriers with different target SNRs
        let carrier1 = Carrier::new(
            ModType::_QPSK,
            0.05,     // 50 kHz bandwidth (narrow)
            0.15,     // 150 kHz center freq
            10.0,     // Target 10 dB SNR
            0.35,     // RRC rolloff
            sample_rate,
            Some(42),
        );

        let carrier2 = Carrier::new(
            ModType::_16QAM,
            0.10,     // 100 kHz bandwidth (wider)
            -0.20,    // -200 kHz center freq
            5.0,      // Target 5 dB SNR
            0.35,
            sample_rate,
            Some(43),
        );

        // Define target SNRs and noise floor for the channel
        let target_snr1_db = 10.0;  // dB
        let target_snr2_db = 5.0;   // dB
        let noise_floor_db = -85.0; // dB
        let noise_floor = 10.0_f64.powf(noise_floor_db / 10.0);

        println!("\n=== Channel Spectrum: Two Carriers with Different SNRs ===");
        println!("Target SNR1: {:.1} dB", target_snr1_db);
        println!("Target SNR2: {:.1} dB", target_snr2_db);
        println!("Noise floor: {:.6} dB ({:.2e} linear)", noise_floor_db, noise_floor);

        // Create channel and set noise floor
        let mut channel = Channel::new(vec![carrier1, carrier2]);
        channel.set_noise_floor_db(noise_floor_db);
        channel.set_seed(999);

        // Generate combined signal with automatic SNR scaling
        let combined = channel.generate::<f64>(num_samples);
        assert_eq!(combined.len(), num_samples);

        // Calculate what the scaled powers should be based on noise floor and target SNRs
        let snr1_linear = 10.0_f64.powf(target_snr1_db / 10.0);
        let snr2_linear = 10.0_f64.powf(target_snr2_db / 10.0);
        let power1_target = snr1_linear * noise_floor;
        let power2_target = snr2_linear * noise_floor;

        println!("\nTarget scaled carrier powers:");
        println!("Carrier 1 power: {:.6} dB (SNR target: {:.1} dB)", 10.0 * power1_target.log10(), target_snr1_db);
        println!("Carrier 2 power: {:.6} dB (SNR target: {:.1} dB)", 10.0 * power2_target.log10(), target_snr2_db);
        println!("Power ratio (C1/C2): {:.2} dB", 10.0 * (power1_target / power2_target).log10());

        println!("\nNoise specification:");
        println!("Noise floor: {:.6} dB ({:.2e} linear)", noise_floor_db, noise_floor);

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
            println!("Skipping spectrum plot (set PLOT=true to enable)");
        }
    }

    #[test]
    fn test_channel_with_single_impairment() {
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
        channel.add_impairment(Impairment::DigitizerDroopAD9361);

        let result = channel.generate::<f64>(1000);
        assert_eq!(result.len(), 1000);
    }

    #[test]
    fn test_channel_with_multiple_impairments() {
        use std::env;
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        // Check if plotting is enabled
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());

        let sample_rate = 1e6;
        let num_samples = 1000000;

        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.1,
            10.0,
            0.35,
            sample_rate,
            Some(42),
        );

        let mut channel = Channel::new(vec![carrier]);
        channel.set_noise_floor_db(-5.0);
        channel.set_seed(998);

        // Add multiple impairments
        channel.add_impairment(Impairment::DigitizerDroopAD9361);
        channel.add_impairment(Impairment::FrequencyVariation {
            amplitude_db: 0.5,
            cycles: 2.0,
            phase_offset: 1.0,
        });

        let result = channel.generate::<f64>(num_samples);
        assert_eq!(result.len(), num_samples);

        println!("\n=== Channel with Multiple Impairments ===");
        println!("Impairments applied:");
        println!("  1. CosineTaperDigitizer (cosine taper, passband=0-42%, transition=42-48%)");
        println!("  2. FrequencyVariation (amplitude=1.0 dB, cycles=2.0)");

        // Only plot if enabled
        if plot.to_lowercase() == "true" {
            // Convert ComplexVec to Vec<Complex<f64>> for Welch processing
            let signal: Vec<_> = (0..result.len()).map(|i| result[i]).collect();

            // Compute Welch PSD
            let (freqs, psd) = welch(
                &signal,
                sample_rate,
                1024,
                Some(512),
                Some(1024),
                WindowType::Hamming,
                None,
            );

            // Convert PSD to dB scale
            let psd_db: Vec<f64> = vector_ops::to_db(&psd);

            // Plot Welch PSD
            plot_spectrum(&freqs, &psd_db, "Channel with Multiple Impairments (Droop + Frequency Variation)");
        } else {
            println!("Skipping spectrum plot (set PLOT=true to enable)");
        }
    }

    #[test]
    fn test_channel_with_custom_digitizer_droop() {
        use std::env;
        use crate::spectrum::welch::welch;
        use crate::spectrum::window::WindowType;
        use crate::vector_ops;
        use crate::plot::plot_spectrum;

        // Check if plotting is enabled
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());

        let sample_rate = 1e6;
        let num_samples = 10000;

        let carrier = Carrier::new(
            ModType::_QPSK,
            0.1,
            0.0,
            10.0,
            0.35,
            sample_rate,
            Some(42),
        );

        let mut channel = Channel::new(vec![carrier]);
        channel.set_noise_floor_db(-100.0);
        channel.set_seed(999);

        // Add custom digitizer droop (4th order, higher cutoff)
        channel.add_impairment(Impairment::DigitizerDroop {
            order: 4,
            cutoff: 0.48,
        });

        let result = channel.generate::<f64>(num_samples);
        assert_eq!(result.len(), num_samples);

        println!("\n=== Channel with Custom Digitizer Droop ===");
        println!("Digitizer Droop Configuration:");
        println!("  Order: 4");
        println!("  Cutoff: 0.48 Nyquist");
        println!("  (Steeper rolloff than AD9361, but not as steep as traditional)");

        // Only plot if enabled
        if plot.to_lowercase() == "true" {
            // Convert ComplexVec to Vec<Complex<f64>> for Welch processing
            let signal: Vec<_> = (0..result.len()).map(|i| result[i]).collect();

            // Compute Welch PSD
            let (freqs, psd) = welch(
                &signal,
                sample_rate,
                1024,
                None,
                None,
                WindowType::Hann,
                None,
            );

            // Convert PSD to dB scale
            let psd_db: Vec<f64> = vector_ops::to_db(&psd);

            // Plot Welch PSD
            plot_spectrum(&freqs, &psd_db, "Channel with Custom Digitizer Droop (4th Order, 0.48 Cutoff)");
        } else {
            println!("Skipping spectrum plot (set PLOT=true to enable)");
        }
    }

}
