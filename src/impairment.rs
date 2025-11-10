#![allow(dead_code)]

use num_traits::Float;
use std::f64::consts::PI;

pub fn frequency_dependent_amplitude_variation<T: Float>(num_samples: usize, amplitude_db: T, cycles: T, phase_offset: T,
) -> Vec<T> {
    let pi = T::from(PI).unwrap();
    (0..num_samples)
        .into_iter()
        .map(|idx| {
            amplitude_db
                * (T::from(2.0).unwrap() * pi * T::from(idx).unwrap()
                    / T::from(num_samples).unwrap()
                    * cycles
                    + phase_offset)
                    .sin()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::{ComplexVec, awgn::AWGN, impairment::frequency_dependent_amplitude_variation, vector_ops::{add, to_linear}, fft::fft};

    #[test]
    fn test_freq_ampl_variation() {
            let num_samples = 300_000;
            let freq_var_1 = frequency_dependent_amplitude_variation(num_samples, 1.0, 2.0, 0.0);
            let freq_var_2 = frequency_dependent_amplitude_variation(num_samples, 0.6, 5.0, 0.0);
            let freq_var_3 = frequency_dependent_amplitude_variation(num_samples, 0.4, 1.0, 1.5);
            let total_variation_db = add(&add(&freq_var_1, &freq_var_2), &freq_var_3);
            let total_variantion_lin = to_linear(&total_variation_db);
            let mut awgn = AWGN::new_from_seed(1e6, num_samples, 1.0, 0);
            let mut noise: ComplexVec<f32> = awgn.generate_block();
            fft::fft(&mut noise);
            noise.iter_mut().enumerate().for_each(|(idx, x)| *x *= total_variantion_lin[idx]);
        }
}
