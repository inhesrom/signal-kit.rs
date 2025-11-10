#![allow(dead_code)]

use num_complex::Complex;
use std::f64::consts::PI;
use num_traits::Float;

pub struct CW {
    freq_hz: f64,
    sample_rate_hz: f64,
    block_size: usize,
    sample_num: usize,
}

impl CW {
    pub fn new(freq_hz: f64, sample_rate_hz: f64, block_size: usize) -> Self {
        CW {
            freq_hz: freq_hz,
            sample_rate_hz: sample_rate_hz,
            block_size: block_size,
            sample_num: 0,
        }
    }

    pub fn generate_block<T: Float>(&mut self) -> Vec<Complex<T>> {
        let mut cw_block = Vec::with_capacity(self.block_size);
        for _ in 0..self.block_size {
            let t: T = T::from(self.sample_num).unwrap() / T::from(self.sample_rate_hz).unwrap();
            let i = (T::from(2.0).unwrap() * T::from(PI).unwrap() * T::from(self.freq_hz).unwrap() * t).cos();
            let q = (T::from(2.0).unwrap() * T::from(PI).unwrap() * T::from(self.freq_hz).unwrap() * t).sin();
            cw_block.push(Complex::<T>::new(i, q));
            self.sample_num += 1;
        }
        cw_block
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use crate::fft::fft;
    use crate::complex_vec::ComplexVec;

    #[test]
    fn test_cw_block_gen() {
        let block_size: usize = 1024;
        let num_blocks = 100;
        let freq_hz = -100e3;
        let sample_rate_hz = 500e3;
        let mut generator = CW::new(freq_hz, sample_rate_hz, block_size);

        let mut samples: Vec<_> = (0..num_blocks)
            .flat_map(|_| generator.generate_block::<f64>())
            .collect();

        assert_eq!(samples.len(), block_size * num_blocks);

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() == "true" {
            fft::fft::<f64>(&mut samples);
            let mut cw_fft = ComplexVec::from_vec(samples);
            let mut cw_fft_abs: Vec<f64> = cw_fft.abs();

            fft::fftshift::<f64>(&mut cw_fft_abs);
            let freqs: Vec<f64> = fft::fftfreqs::<f64>((-sample_rate_hz/2_f64) as f64, (sample_rate_hz/2_f64) as f64, cw_fft_abs.len());

            use crate::plot::plot_spectrum;
            plot_spectrum(&freqs, &cw_fft_abs, "CW Signal Spectrum");
        }
    }
}
