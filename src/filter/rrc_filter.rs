#![allow(dead_code)]

use num_complex::Complex;
use num_traits::Float;
use crate::complex_vec::ComplexVec;

pub struct RRCFilter {
    num_filter_taps: usize,
    beta: f64,
    sps: f64, //samples per symbol
    symbol_rate: f64,
    scale: f64,
}

impl RRCFilter {
    pub fn new(num_filter_taps: usize, sample_rate: f64, symbol_rate: f64, beta: f64) -> Self {
        let sps: f64 = sample_rate / symbol_rate;
        let scale = 1.0f64 / sample_rate;

        RRCFilter {
            num_filter_taps,
            beta,
            sps,
            symbol_rate,
            scale
        }
    }

    pub fn build_filter<T: Float>(&self) -> ComplexVec<T> {
        let pi = std::f64::consts::PI;
        let zero = 0.0f64;
        let one = 1.0f64;
        let two = 2.0f64;
        let four = 4.0f64;

        let mut taps: Vec<f64> = (0..self.num_filter_taps).map(|i| {
            // Use integer center to ensure t=0 at center tap
            let center = (self.num_filter_taps - 1) as f64 / two;
            let t = (i as f64 - center) * self.scale;

            // Normalize time by symbol period: t_norm = t * symbol_rate = t / T
            let t_norm = t * self.symbol_rate;

            if t == zero {
                // At t=0: h(0) = (1 + β*(4/π - 1)) / T
                (one + self.beta * (four / pi - one)) * self.symbol_rate
            }
            else if (four * self.beta * t_norm).abs() == one {
                // At discontinuity: t = ±T/(4β)
                let sqrt2 = two.sqrt();
                (self.beta / sqrt2) *
                ((one + two / pi) * (pi / (four * self.beta)).sin() +
                 (one - two / pi) * (pi / (four * self.beta)).cos()) * self.symbol_rate
            }
            else {
                // General case:
                // h(t) = [sin(π*t/T*(1-β)) + 4*β*t/T*cos(π*t/T*(1+β))] / [π*t/T*(1-(4*β*t/T)²)]
                let numerator = (pi * t_norm * (one - self.beta)).sin() +
                                four * self.beta * t_norm * (pi * t_norm * (one + self.beta)).cos();
                let denominator = pi * t_norm * (one - (four * self.beta * t_norm).powi(2));
                numerator / denominator * self.symbol_rate
            }
        }).collect();

        // Normalize filter for unit energy
        // For a pulse shaping filter, we want the energy (sum of squares) to equal 1
        let energy: f64 = taps.iter().map(|&x| x * x).sum();
        let norm_factor = energy.sqrt();
        if norm_factor > zero {
            for tap in taps.iter_mut() {
                *tap /= norm_factor;
            }
        }

        // Convert real taps to complex with zero imaginary part
        let complex_taps: Vec<Complex<T>> = taps.iter()
            .map(|&tap| Complex::new(T::from(tap).unwrap(), T::zero()))
            .collect();

        ComplexVec::from_vec(complex_taps)
    }
}

#[cfg(test)]
mod tests {
    use crate::filter::rrc_filter::RRCFilter;

    #[test]
    fn test_build_filter_f32() {
        let filter32 = RRCFilter::new(101, 1e6f64, 100e3f64, 0.35f64);
        let taps32 = filter32.build_filter::<f32>();
        assert_eq!(taps32.len(), 101);
        println!("f32 Filter taps count: {}", taps32.len());
    }

    #[test]
    fn test_build_filter_f64() {
        let filter64 = RRCFilter::new(51, 1e6_f64, 500e3_f64, 0.25_f64);
        let taps64 = filter64.build_filter::<f64>();
        assert_eq!(taps64.len(), 51);
        println!("f64 Filter taps count: {}", taps64.len());
    }

    #[test]
    fn test_build_filter_complex32() {
        let filter32 = RRCFilter::new(101, 1.5, 1.0, 0.20);
        let taps32 = filter32.build_filter::<f32>();
        assert_eq!(taps32.len(), 101);

        println!("Complex32 Filter taps count: {}", taps32.len());

        // Verify all imaginary parts are near zero (filter taps should be real)
        for i in 0..5 {
            println!("tap[{}] = {} + {}i", i, taps32[i].re, taps32[i].im);
            assert!(taps32[i].im.abs() < 1e-10);
        }
    }
}
