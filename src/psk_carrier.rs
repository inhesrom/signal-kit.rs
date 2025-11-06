#![allow(dead_code)]

use crate::mod_type::*;
use crate::mod_type::modulation::Modulate;
use crate::complex_vec::{ComplexVec, ConvMode};
use crate::random_bit_generator::BitGenerator;
use crate::rrc_filter::RRCFilter;
use num_traits::Float;

pub struct PskCarrier<T: Float> {
    sample_rate_hz: T,
    symbol_rate_hz: T,
    mod_type: ModType,
    rolloff_factor: T,
    block_size: usize,
    current_sample_num: usize, //Helps hold onto "time" state during generation of multiple blocks
    bit_gen: BitGenerator,
    filter_taps: usize,
    samples_per_symbol: usize, // Integer upsampling factor
    overlap_symbols: ComplexVec<T>,
    rrc_filter: ComplexVec<T>,
    modulation: modulation::Modulation<T>,
}

impl<T: Float> PskCarrier<T> {
    pub fn new(sample_rate_hz: T, symbol_rate_hz: T, mod_type: ModType, rolloff_factor: T, block_size: usize, filter_taps: usize, seed: Option<u64>) -> Self {
        let bit_gen = match seed {
            Some(s) => BitGenerator::new_from_seed(s),
            None => BitGenerator::new_from_entropy(),
        };

        // Calculate integer upsampling factor
        // For 1MHz sample rate / 800kHz symbol rate = 1.25, we round to nearest integer
        let sps_exact = sample_rate_hz.to_f64().unwrap() / symbol_rate_hz.to_f64().unwrap();
        let samples_per_symbol = sps_exact.round() as usize;
        if samples_per_symbol == 0 {
            panic!("samples_per_symbol must be at least 1");
        }

        // Build RRC filter once during initialization
        let rrc = RRCFilter::new(
            filter_taps,
            sample_rate_hz.to_f64().unwrap(),
            symbol_rate_hz.to_f64().unwrap(),
            rolloff_factor.to_f64().unwrap(),
        );
        let rrc_filter = rrc.build_filter::<T>();

        // Get modulation with pre-built HashMap once
        let modulation = modulation::get_mod_type_from_enum::<T>(mod_type);

        // Initialize overlap_symbols with (filter_taps - 1) zeros for proper first block sizing
        let overlap_size = filter_taps - 1;
        let zero_symbols = vec![num_complex::Complex::new(T::zero(), T::zero()); overlap_size];
        let overlap_symbols = ComplexVec::from_vec(zero_symbols);

        PskCarrier {
            sample_rate_hz,
            symbol_rate_hz,
            mod_type,
            rolloff_factor,
            block_size,
            current_sample_num: 0,
            bit_gen,
            filter_taps,
            samples_per_symbol,
            overlap_symbols,
            rrc_filter,
            modulation,
        }
    }

    pub fn generate_block(&mut self) -> ComplexVec<T> {
        // Calculate how many symbols we need to generate to get block_size samples
        // For sps=1: 1 symbol -> 1 sample (after upsampling and filtering)
        // We need to account for the overlap and filter delay
        let num_symbols = if self.samples_per_symbol == 1 {
            self.block_size
        } else {
            // For upsampling: symbols * sps ≈ samples
            (self.block_size + self.samples_per_symbol - 1) / self.samples_per_symbol
        };

        // Generate symbols for this block using the pre-built modulation
        let new_symbols = self.generate_symbols(num_symbols);

        // Combine overlap from previous block with new symbols
        let mut all_symbols = self.overlap_symbols.clone();
        all_symbols.extend(new_symbols.iter().cloned());

        // Upsample symbols by inserting zeros between them
        let upsampled = if self.samples_per_symbol > 1 {
            self.upsample_symbols(&all_symbols)
        } else {
            all_symbols
        };

        // Apply RRC pulse shaping via convolution using Full mode
        // Full mode preserves all convolution outputs
        let pulse_shaped = upsampled.convolve(&self.rrc_filter, ConvMode::Full);

        // For proper pulse shaping, we need to account for filter group delay
        // Group delay = (filter_taps - 1) / 2 samples
        let group_delay = (self.filter_taps - 1) / 2;

        // Extract the valid portion after accounting for group delay
        // Skip the first group_delay samples to remove the filter startup transient
        let valid_start = group_delay;
        let valid_end = valid_start + self.block_size;

        let mut output = Vec::new();
        for i in valid_start..valid_end.min(pulse_shaped.len()) {
            output.push(pulse_shaped[i]);
        }

        // Save last (filter_taps - 1) symbols for next block continuity
        self.overlap_symbols = self.extract_last_symbols(&new_symbols, self.filter_taps - 1);

        ComplexVec::from_vec(output)
    }

    /// Upsample symbols by inserting (sps - 1) zeros between each symbol
    fn upsample_symbols(&self, symbols: &ComplexVec<T>) -> ComplexVec<T> {
        let mut upsampled = Vec::new();
        let zero = num_complex::Complex::new(T::zero(), T::zero());

        for i in 0..symbols.len() {
            upsampled.push(symbols[i]);
            // Insert (sps - 1) zeros after each symbol (except the last one)
            if i < symbols.len() - 1 {
                for _ in 0..(self.samples_per_symbol - 1) {
                    upsampled.push(zero);
                }
            }
        }

        ComplexVec::from_vec(upsampled)
    }

    fn generate_symbols(&mut self, count: usize) -> ComplexVec<T> {
        let mut symbols = Vec::new();

        match &self.modulation {
            modulation::Modulation::BPSK(bpsk) => {
                for _ in 0..count {
                    let bit = self.bit_gen.next_bit();
                    let bits = if bit { 1u8 } else { 0u8 };
                    if let Some(symbol) = bpsk.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::QPSK(qpsk) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_2_bits();
                    if let Some(symbol) = qpsk.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::PSK8(psk8) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_3_bits();
                    if let Some(symbol) = psk8.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::APSK16(apsk16) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_bits(4) as u8;
                    if let Some(symbol) = apsk16.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::QAM16(qam16) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_bits(4) as u8;
                    if let Some(symbol) = qam16.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::QAM32(qam32) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_bits(5) as u8;
                    if let Some(symbol) = qam32.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
            modulation::Modulation::QAM64(qam64) => {
                for _ in 0..count {
                    let bits = self.bit_gen.next_bits(6) as u8;
                    if let Some(symbol) = qam64.modulate(bits) {
                        symbols.push(symbol);
                    }
                }
            },
        }

        ComplexVec::from_vec(symbols)
    }

    fn extract_last_symbols(&self, symbols: &ComplexVec<T>, count: usize) -> ComplexVec<T> {
        let start_idx = if symbols.len() > count {
            symbols.len() - count
        } else {
            0
        };

        let mut last_symbols = Vec::new();
        for i in start_idx..symbols.len() {
            last_symbols.push(symbols[i]);
        }
        ComplexVec::from_vec(last_symbols)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::fft::fft::{fft, fftshift, fftfreqs};
    use crate::vector_ops;

    #[test]
    fn test_qpsk_carrier_block_gen() {
        let block_size: usize = 1e6 as usize;
        let num_blocks = 1;
        let sample_rate_hz = 1e6_f64;
        let symbol_rate_hz = 800e3_f64;
        let filter_taps = 51;
        let rolloff_factor = 0.05_f64;

        let mut carrier = PskCarrier::new(
            sample_rate_hz,
            symbol_rate_hz,
            ModType::_QPSK,
            rolloff_factor,
            block_size,
            filter_taps,
            Some(42), // Seed for reproducibility
        );

        println!("\n=== QPSK Carrier Configuration ===");
        println!("Block size: {}", block_size);
        println!("Sample rate: {} Hz", sample_rate_hz);
        println!("Symbol rate: {} Hz", symbol_rate_hz);
        println!("Samples per symbol: {}", carrier.samples_per_symbol);
        println!("Filter taps: {}", filter_taps);
        println!("Rolloff factor: {}", rolloff_factor);

        // Generate blocks and merge into single ComplexVec
        let mut all_samples = ComplexVec::new();
        for _ in 0..num_blocks {
            let block = carrier.generate_block();
            all_samples.extend(block.iter().cloned());
        }

        println!("\n=== Generated Samples ===");
        println!("Total samples: {}", all_samples.len());
        println!("Expected samples: {}", block_size * num_blocks);

        // Compute signal statistics
        let mut max_mag = 0.0_f64;
        let mut sum_power = 0.0_f64;
        for i in 0..all_samples.len() {
            let mag = all_samples[i].norm();
            max_mag = max_mag.max(mag);
            sum_power += mag * mag;
        }
        let avg_power = sum_power / all_samples.len() as f64;

        println!("Max magnitude: {:.4}", max_mag);
        println!("Average power: {:.4}", avg_power);
        println!("First 10 samples:");
        for i in 0..10.min(all_samples.len()) {
            println!("  Sample {}: I={:.4}, Q={:.4}, Mag={:.4}",
                i, all_samples[i].re, all_samples[i].im, all_samples[i].norm());
        }
        println!("Samples 100-110 (after filter startup):");
        for i in 100..110.min(all_samples.len()) {
            println!("  Sample {}: I={:.4}, Q={:.4}, Mag={:.4}",
                i, all_samples[i].re, all_samples[i].im, all_samples[i].norm());
        }

        assert_eq!(all_samples.len(), block_size * num_blocks);

        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        println!("\nTEST_PLOT env var is {}", plot);
        if plot.to_lowercase() == "true" {
            // Convert to Vec for FFT
            let mut samples_vec: Vec<_> = (0..all_samples.len())
                .map(|i| all_samples[i])
                .collect();

            fft::<f64>(&mut samples_vec);
            let mut qpsk_fft = ComplexVec::from_vec(samples_vec);
            let mut qpsk_fft_abs: Vec<f64> = vector_ops::to_db(&qpsk_fft.abs());

            fftshift::<f64>(&mut qpsk_fft_abs);
            let freqs: Vec<f64> = fftfreqs::<f64>(
                -sample_rate_hz / 2_f64,
                sample_rate_hz / 2_f64,
                qpsk_fft_abs.len()
            );

            println!("\n=== Spectrum Analysis ===");
            println!("FFT bins: {}", qpsk_fft_abs.len());
            println!("Frequency resolution: {:.2} Hz", sample_rate_hz / qpsk_fft_abs.len() as f64);

            // Find peak and bandwidth
            let max_db = qpsk_fft_abs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!("Peak spectrum level: {:.2} dB", max_db);

            use crate::plot::plot_spectrum;
            plot_spectrum(&freqs, &qpsk_fft_abs, "QPSK Signal Spectrum");
        }
    }

    #[test]
    fn test_qpsk_symbols_constellation() {
        use std::env;
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;

        let plot_env = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        if plot_env.to_lowercase() != "true" {
            println!("Skipping symbols constellation plot (set TEST_PLOT=true to enable)");
            return;
        }

        let sample_rate_hz = 1e6_f64;
        let symbol_rate_hz = 800e3_f64;
        let filter_taps = 51;
        let rolloff_factor = 0.05_f64;

        let mut carrier = PskCarrier::new(
            sample_rate_hz,
            symbol_rate_hz,
            ModType::_QPSK,
            rolloff_factor,
            1000, // Small block for visualization
            filter_taps,
            Some(42),
        );

        // Generate just the raw symbols (before pulse shaping) for visualization
        let num_symbols = 1000;
        let symbols = carrier.generate_symbols(num_symbols);

        // Extract I and Q values
        let i_vals: Vec<f64> = (0..symbols.len()).map(|i| symbols[i].re).collect();
        let q_vals: Vec<f64> = (0..symbols.len()).map(|i| symbols[i].im).collect();

        // Create scatter plot
        let trace = Scatter::new(i_vals.clone(), q_vals.clone())
            .mode(Mode::Markers)
            .name("Generated QPSK Symbols");

        let mut plot = Plot::new();
        plot.add_trace(trace);

        // Add reference constellation points
        let ref_i = vec![0.7071, -0.7071, -0.7071, 0.7071];
        let ref_q = vec![0.7071, 0.7071, -0.7071, -0.7071];
        let ref_trace = Scatter::new(ref_i, ref_q)
            .mode(Mode::Markers)
            .name("Ideal QPSK Points");
        plot.add_trace(ref_trace);

        // Set layout
        use plotly::Layout;
        let layout = Layout::new()
            .title("Generated QPSK Symbols vs Ideal Constellation")
            .x_axis(plotly::layout::Axis::new().title("In-Phase (I)"))
            .y_axis(plotly::layout::Axis::new().title("Quadrature (Q)"));
        plot.set_layout(layout);

        println!("\nGenerated {} QPSK symbols", num_symbols);
        println!("First 10 symbols:");
        for i in 0..10.min(symbols.len()) {
            println!("Symbol {}: I={:.4}, Q={:.4}, Mag={:.4}",
                i, symbols[i].re, symbols[i].im, symbols[i].norm());
        }

        plot.show();
    }

    #[test]
    fn test_rrc_filter_frequency_response() {
        use std::env;
        use crate::fft::fft::{fft, fftshift, fftfreqs};
        use crate::vector_ops;

        let sample_rate_hz = 1e6_f64;
        let symbol_rate_hz = 800e3_f64;
        let filter_taps = 51;
        let rolloff_factor = 0.05_f64;

        // Build the RRC filter
        let rrc = RRCFilter::new(
            filter_taps,
            sample_rate_hz,
            symbol_rate_hz,
            rolloff_factor,
        );
        let filter = rrc.build_filter::<f64>();

        println!("\n=== RRC Filter Analysis ===");
        println!("Filter taps: {}", filter.len());
        println!("Sample rate: {} Hz", sample_rate_hz);
        println!("Symbol rate: {} Hz", symbol_rate_hz);
        println!("Samples per symbol (SPS): {}", sample_rate_hz / symbol_rate_hz);
        println!("Rolloff: {}", rolloff_factor);

        // Print all taps to see the shape
        println!("\nAll filter taps:");
        for i in 0..filter.len() {
            println!("  Tap {}: {:.8}", i, filter[i].re);
        }

        // Compute energy
        let energy: f64 = (0..filter.len()).map(|i| filter[i].re * filter[i].re).sum();
        println!("\nFilter energy (sum of squared taps): {:.6}", energy);

        let plot_env = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        if plot_env.to_lowercase() != "true" {
            println!("Set TEST_PLOT=true to see frequency response plot");
            return;
        }

        // Compute frequency response using FFT
        // Zero-pad to get better frequency resolution
        let fft_size = 8192;
        let mut padded: Vec<_> = (0..filter.len()).map(|i| filter[i]).collect();
        padded.resize(fft_size, num_complex::Complex::new(0.0, 0.0));

        fft::<f64>(&mut padded);
        let mut freq_response = ComplexVec::from_vec(padded);
        let mut freq_response_db: Vec<f64> = vector_ops::to_db(&freq_response.abs());

        fftshift::<f64>(&mut freq_response_db);
        let freqs: Vec<f64> = fftfreqs::<f64>(
            -sample_rate_hz / 2.0,
            sample_rate_hz / 2.0,
            freq_response_db.len()
        );

        use crate::plot::plot_spectrum;
        plot_spectrum(&freqs, &freq_response_db, "RRC Filter Frequency Response");
    }
}
