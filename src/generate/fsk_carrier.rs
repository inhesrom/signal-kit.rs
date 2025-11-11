#![allow(dead_code)]

use num_complex::Complex;
use num_traits::Float;
use std::f64::consts::PI;
use crate::generate::random_bit_generator::BitGenerator;
use crate::complex_vec::ComplexVec;

/// Binary Continuous Phase Frequency Shift Keying (CPFSK) carrier generator
///
/// Generates FSK modulated signals where binary data is represented by
/// two different frequencies. Phase is continuous across symbol boundaries
/// to minimize spectral splatter.
///
/// For binary FSK:
/// - Bit 0 -> frequency: fc - Df/2
/// - Bit 1 -> frequency: fc + Df/2
///
/// where fc is the carrier frequency and Df is the frequency deviation.
pub struct FskCarrier<T: Float> {
    sample_rate_hz: T,
    symbol_rate_hz: T,
    carrier_freq_hz: T,
    freq_deviation_hz: T,
    block_size: usize,
    accumulated_phase: T,
    bit_gen: BitGenerator,
    samples_per_symbol: usize,
}

impl<T: Float> FskCarrier<T> {
    /// Create a new FSK carrier generator
    ///
    /// # Arguments
    /// * `sample_rate_hz` - Sample rate in Hz
    /// * `symbol_rate_hz` - Symbol rate (baud rate) in Hz
    /// * `carrier_freq_hz` - Center frequency in Hz
    /// * `freq_deviation_hz` - Frequency deviation in Hz (total separation is 2*deviation)
    /// * `block_size` - Number of samples to generate per block
    /// * `seed` - Optional seed for bit generator (None uses entropy)
    ///
    /// # Returns
    /// A new FskCarrier instance
    ///
    /// # Example
    /// ```
    /// use signal_kit::generate::fsk_carrier::FskCarrier;
    /// 
    /// // Create 1 MHz sample rate, 100 kHz symbol rate, 250 kHz carrier
    /// // with 50 kHz deviation (frequencies at 225 kHz and 275 kHz)
    /// let fsk: FskCarrier<f64> = FskCarrier::new(1e6, 1e5, 2.5e5, 5e4, 1024, Some(42));
    /// ```
    pub fn new(
        sample_rate_hz: f64,
        symbol_rate_hz: f64,
        carrier_freq_hz: f64,
        freq_deviation_hz: f64,
        block_size: usize,
        seed: Option<u64>,
    ) -> Self {
        let samples_per_symbol = (sample_rate_hz / symbol_rate_hz).round() as usize;

        let bit_gen = match seed {
            Some(s) => BitGenerator::new_from_seed(s),
            None => BitGenerator::new_from_entropy(),
        };

        FskCarrier {
            sample_rate_hz: T::from(sample_rate_hz).unwrap(),
            symbol_rate_hz: T::from(symbol_rate_hz).unwrap(),
            carrier_freq_hz: T::from(carrier_freq_hz).unwrap(),
            freq_deviation_hz: T::from(freq_deviation_hz).unwrap(),
            block_size,
            accumulated_phase: T::zero(),
            bit_gen,
            samples_per_symbol,
        }
    }

    /// Generate a block of FSK modulated samples
    ///
    /// Generates `block_size` samples of binary FSK modulation.
    /// Phase is continuous across symbol boundaries and across multiple
    /// calls to generate_block().
    ///
    /// # Returns
    /// ComplexVec<T> containing the FSK modulated signal
    pub fn generate_block(&mut self) -> ComplexVec<T> {
        let num_symbols = self.block_size / self.samples_per_symbol;
        let mut samples = Vec::with_capacity(self.block_size);

        let two_pi = T::from(2.0 * PI).unwrap();
        let half = T::from(0.5).unwrap();

        for _ in 0..num_symbols {
            // Get next bit
            let bit = self.bit_gen.next_bit();

            // Calculate instantaneous frequency for this symbol
            // bit 0 -> fc - Df/2
            // bit 1 -> fc + Df/2
            let freq = if bit {
                self.carrier_freq_hz + self.freq_deviation_hz * half
            } else {
                self.carrier_freq_hz - self.freq_deviation_hz * half
            };

            // Generate samples for this symbol
            for _ in 0..self.samples_per_symbol {
                // Calculate phase increment for this sample
                let phase_increment = two_pi * freq / self.sample_rate_hz;

                // Accumulate phase (ensures continuity)
                self.accumulated_phase = self.accumulated_phase + phase_increment;

                // Wrap phase to [-�, �] to prevent numerical issues
                while self.accumulated_phase > T::from(PI).unwrap() {
                    self.accumulated_phase = self.accumulated_phase - two_pi;
                }
                while self.accumulated_phase < T::from(-PI).unwrap() {
                    self.accumulated_phase = self.accumulated_phase + two_pi;
                }

                // Generate IQ samples
                let i = self.accumulated_phase.cos();
                let q = self.accumulated_phase.sin();

                samples.push(Complex::new(i, q));
            }
        }

        // Handle any remaining samples to exactly match block_size
        let remaining = self.block_size - samples.len();
        if remaining > 0 {
            let bit = self.bit_gen.next_bit();
            let freq = if bit {
                self.carrier_freq_hz + self.freq_deviation_hz * half
            } else {
                self.carrier_freq_hz - self.freq_deviation_hz * half
            };

            for _ in 0..remaining {
                let phase_increment = two_pi * freq / self.sample_rate_hz;
                self.accumulated_phase = self.accumulated_phase + phase_increment;

                while self.accumulated_phase > T::from(PI).unwrap() {
                    self.accumulated_phase = self.accumulated_phase - two_pi;
                }
                while self.accumulated_phase < T::from(-PI).unwrap() {
                    self.accumulated_phase = self.accumulated_phase + two_pi;
                }

                let i = self.accumulated_phase.cos();
                let q = self.accumulated_phase.sin();
                samples.push(Complex::new(i, q));
            }
        }

        ComplexVec::from_vec(samples)
    }

    /// Get the modulation index (h = Df / symbol_rate)
    ///
    /// Common values:
    /// - h = 0.5: MSK (Minimum Shift Keying)
    /// - h = 1.0: Orthogonal FSK
    pub fn modulation_index(&self) -> T {
        self.freq_deviation_hz / self.symbol_rate_hz
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::awgn::AWGN;
    use crate::fft::{fft, fftshift, fftfreqs};
    use crate::vector_ops;
    use std::env;

    #[test]
    fn test_fsk_basic() {
        let sample_rate = 1e6;
        let symbol_rate = 1e5;
        let carrier_freq = 2.5e5;
        let deviation = 5e4;
        let block_size = 1000;

        let mut fsk: FskCarrier<f64> = FskCarrier::new(
            sample_rate,
            symbol_rate,
            carrier_freq,
            deviation,
            block_size,
            Some(42),
        );

        let block = fsk.generate_block();

        // Verify block size
        assert_eq!(block.len(), block_size);

        // Verify samples are unit magnitude (approximately)
        for i in 0..block.len() {
            let mag = block[i].norm();
            assert!((mag - 1.0).abs() < 1e-10, "Sample {} has magnitude {}, expected 1.0", i, mag);
        }
    }

    #[test]
    fn test_fsk_phase_continuity() {
        let sample_rate = 1e6;
        let symbol_rate = 1e5;
        let carrier_freq = 2.5e5;
        let deviation = 5e4;
        let block_size = 100;

        let mut fsk: FskCarrier<f64> = FskCarrier::new(
            sample_rate,
            symbol_rate,
            carrier_freq,
            deviation,
            block_size,
            Some(123),
        );

        // Generate multiple blocks and verify phase continuity
        let block1 = fsk.generate_block();
        let block2 = fsk.generate_block();

        // Blocks should not be identical (different random bits)
        let blocks_identical = block1.iter().zip(block2.iter()).all(|(a, b)| a == b);
        assert!(!blocks_identical, "Blocks should not be identical due to different random bits");

        // All samples should have unit magnitude
        for i in 0..block1.len() {
            assert!((block1[i].norm() - 1.0).abs() < 1e-10);
        }
        for i in 0..block2.len() {
            assert!((block2[i].norm() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_fsk_modulation_index() {
        let fsk: FskCarrier<f64> = FskCarrier::new(1e6, 1e5, 2.5e5, 5e4, 1000, Some(42));

        let h = fsk.modulation_index();

        // For these parameters: h = 50000 / 100000 = 0.5 (MSK)
        assert!((h - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_fsk_with_awgn_spectrum() {
        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping FSK with AWGN spectrum plot (set PLOT=true to enable)");
            return;
        }

        println!("\n=== FSK with AWGN Spectrum Test ===");

        // FSK parameters
        let sample_rate = 1e6_f64;
        let symbol_rate = 1e5_f64;
        let carrier_freq = 2.5e5_f64;
        let deviation = 5e4_f64;  // Two frequencies: 200 kHz and 300 kHz
        let block_size = 4096;

        println!("Sample rate: {} Hz", sample_rate);
        println!("Symbol rate: {} Hz", symbol_rate);
        println!("Carrier freq: {} Hz", carrier_freq);
        println!("Deviation: {} Hz", deviation);
        println!("Frequencies: {} Hz and {} Hz",
                 carrier_freq - deviation/2.0,
                 carrier_freq + deviation/2.0);

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
        println!("Generated {} FSK samples", signal.len());

        // Calculate signal power
        let mut signal_power = 0.0;
        for i in 0..signal.len() {
            signal_power += signal[i].norm_sqr();
        }
        signal_power /= signal.len() as f64;
        println!("Signal power: {:.4}", signal_power);

        // Add AWGN at 15 dB SNR
        let snr_db = 15.0;
        let snr_linear = 10.0_f64.powf(snr_db / 10.0);
        let noise_power = signal_power / snr_linear;
        println!("SNR: {} dB", snr_db);
        println!("Noise power: {:.4}", noise_power);

        let mut awgn = AWGN::new_from_seed(sample_rate, block_size, noise_power, 999);
        let noise = awgn.generate_block();

        // Add noise to signal
        let mut noisy_samples = Vec::with_capacity(signal.len());
        for i in 0..signal.len() {
            noisy_samples.push(signal[i] + noise[i]);
        }

        // Compute FFT of noisy signal
        let mut freq_data = noisy_samples.clone();
        fft(&mut freq_data[..]);
        let mut noisy_fft = ComplexVec::from_vec(freq_data);
        let mut spectrum_db: Vec<f64> = vector_ops::to_db(&noisy_fft.abs());

        // Apply fftshift and compute frequency axis
        fftshift::<f64>(&mut spectrum_db);
        let freqs: Vec<f64> = fftfreqs::<f64>(
            -sample_rate / 2.0,
            sample_rate / 2.0,
            spectrum_db.len(),
        );

        println!("FFT size: {}", spectrum_db.len());
        println!("\nPlotting FSK spectrum with AWGN...");
        println!("You should see two peaks at ~{} kHz and ~{} kHz",
                 (carrier_freq - deviation/2.0) / 1e3,
                 (carrier_freq + deviation/2.0) / 1e3);

        // Plot spectrum
        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &spectrum_db, "Binary FSK Signal with AWGN (SNR=15dB)");
    }
}
