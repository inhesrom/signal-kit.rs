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

impl CW
where
{
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

