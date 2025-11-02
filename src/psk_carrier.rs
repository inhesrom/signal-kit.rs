#![allow(dead_code)]

use crate::mod_type::*;
use crate::mod_type::modulation::Modulate;
use crate::complex_vec::ComplexVec;
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
            overlap_symbols,
            rrc_filter,
            modulation,
        }
    }

    pub fn generate_block(&mut self) -> ComplexVec<T> {
        // Generate symbols for this block using the pre-built modulation
        let new_symbols = self.generate_symbols(self.block_size);

        // Combine overlap from previous block with new symbols
        let mut all_symbols = self.overlap_symbols.clone();
        all_symbols.extend(new_symbols.iter().cloned());

        // Apply RRC pulse shaping via convolution
        let pulse_shaped = all_symbols.convolve(&self.rrc_filter);

        // Save last (filter_taps - 1) symbols for next block continuity
        self.overlap_symbols = self.extract_last_symbols(&new_symbols, self.filter_taps - 1);

        pulse_shaped
    }

    fn generate_symbols(&mut self, count: usize) -> ComplexVec<T> {
        let mut symbols = Vec::new();

        match &self.modulation {
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
    use plotly::{Plot, Scatter};
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

        // Generate 100 blocks and merge into single ComplexVec
        let mut all_samples = ComplexVec::new();
        for _ in 0..num_blocks {
            let block = carrier.generate_block();
            all_samples.extend(block.iter().cloned());
        }

        assert_eq!(all_samples.len(), block_size * num_blocks);

        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        println!("TEST_PLOT env var is {}", plot);
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

            let mut plot = Plot::new();
            let trace = Scatter::new(freqs.clone(), qpsk_fft_abs.clone());
            plot.add_trace(trace);
            plot.show();
        }
    }
}
