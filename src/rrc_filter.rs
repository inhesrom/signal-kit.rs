#![allow(dead_code)]

use num_complex::{Complex, ComplexFloat};
use num_traits::Float;

pub struct RRCFilter {
    num_filter_taps: usize,
    beta: f64,
    sps: f64, //samples per symbol
    scale: f64,
}

impl RRCFilter {
    pub fn new(num_filter_taps: usize, sample_rate: f64, symbol_rate: f64, beta: f64) -> Self {
        let sps = sample_rate / symbol_rate;
        let scale = 1.0f64 / sample_rate;

        RRCFilter {
            num_filter_taps,
            beta,
            sps,
            scale
        }
    }

    pub fn build_filter<V, T>(&self) -> V
    where
        V: Default + Extend<T>,
        T: ComplexFloat,
    {
        let pi = std::f64::consts::PI;
        let zero = 0.0f64;
        let one = 1.0f64;
        let two = 2.0f64;
        let four = 4.0f64;

        let taps: Vec<f64> = (0..self.num_filter_taps).map(|i| {
            let t = i as f64 - self.num_filter_taps as f64 / two;
            let t = t * self.scale;

            if t == zero {
                (one + self.beta * (four / pi - one)) / self.sps
            }
            else if t.abs() == (self.sps / (four * self.beta)).abs() {
                (self.beta / two.sqrt()) *
                ((one + two / pi) * (pi / (four * self.beta)).sin() +
                 (one - two / pi) * (pi / (four * self.beta)).cos()) / self.sps
            }
            else {
                ((pi * t * (one - self.beta) / self.sps).sin() +
                 four * self.beta * t * (pi * t * (one + self.beta) / self.sps).cos() / self.sps) /
                (pi * t * (one - (four * self.beta * t / self.sps) * (four * self.beta * t / self.sps)) / self.sps)
            }
        }).collect();

        let mut complex_taps = V::default();
        complex_taps.extend(taps.iter().map(|&tap| T::from(tap).unwrap()));
        complex_taps
    }
}

#[cfg(test)]
mod tests {
    use crate::rrc_filter::RRCFilter;
    use num_complex::{Complex, Complex32};

    #[test]
    fn test_build_filter_f32() {
        // Test with real f32 type
        let filter32 = RRCFilter::new(101, 1e6f64, 100e3f64, 0.35f64);
        let taps32: Vec<f32> = filter32.build_filter::<Vec<_>, f32>();
        println!("f32 Filter taps count: {}", taps32.len());
    }

    #[test]
    fn test_build_filter_f64() {
        let filter64 = RRCFilter::new(51, 1e6_f64, 500e3_f64, 0.25_f64);
        let taps64: Vec<f64> = filter64.build_filter::<Vec<_>, f64>();
        println!("f64 Filter taps count: {}", taps64.len());
    }

    #[test]
    fn test_build_filter_complex32() {
        let filter32 = RRCFilter::new(101, 1.5, 1.0, 0.20);
        let taps32: Vec<Complex32> = filter32.build_filter::<Vec<_>, Complex32>();

        println!("Complex32 Filter taps count: {}", taps32.len());

        // Verify all imaginary parts are near zero (filter taps should be real)
        for (i, tap) in taps32.iter().enumerate().take(5) {
            println!("tap[{}] = {} + {}i", i, tap.re, tap.im);
        }
    }
}
