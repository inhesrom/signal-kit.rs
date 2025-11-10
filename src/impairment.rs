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
    use crate::{ComplexVec, awgn::AWGN, impairment::frequency_dependent_amplitude_variation, vector_ops::{add, to_linear}, fft::fft, welch::welch, window::WindowType, plot::plot_spectrum, vector_ops};

    #[test]
    fn test_freq_ampl_variation() {
        use std::env;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping frequency-dependent amplitude variation plot (set PLOT=true to enable)");
            return;
        }

        let sample_rate = 1e6_f64;
        let num_samples = (2.0_f64).powf(20.0) as usize;  // Use power of 2 for better FFT performance
        // Generate three different frequency variation patterns
        let freq_var_1 = frequency_dependent_amplitude_variation(num_samples, 1.0, 2.0, 0.0);
        let freq_var_2 = frequency_dependent_amplitude_variation(num_samples, 0.6, 5.0, 0.0);
        let freq_var_3 = frequency_dependent_amplitude_variation(num_samples, 0.4, 1.0, 1.5);
        // Combine all variations
        let total_variation_db = add(&add(&freq_var_1, &freq_var_2), &freq_var_3);
        let total_variation_lin = to_linear(&total_variation_db);
        // Generate white noise
        let mut awgn = AWGN::new_from_seed(sample_rate, num_samples, 1.0, 0);
        let mut noise: ComplexVec<f32> = awgn.generate_block();
        // Apply frequency-dependent amplitude variation in frequency domain
        fft::fft(&mut noise);
        for (i, sample) in noise.iter_mut().enumerate() {
            *sample *= total_variation_lin[i] as f32;
        }
        fft::ifft(&mut noise);
        // Compute Welch PSD
        let (freqs, psd) = welch::<f32>(
            &noise,
            sample_rate as f32,
            2048,                    // 1024-point segments
            None,                    // 50% overlap (default)
            None,                    // No zero-padding (default)
            WindowType::Hann,        // Hann window (standard)
            None,                    // Mean averaging (default)
        );
        // Convert PSD to dB scale
        let psd_db: Vec<f32> = vector_ops::to_db(&psd);
        // Plot Welch PSD
        plot_spectrum(&freqs, &psd_db, "Frequency-Dependent Amplitude Variation Applied to AWGN");
    }
}
