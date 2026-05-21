//! Streaming cubic-Lagrange Farrow resampler over `Complex<T>` IQ samples.
//!
//! The resampler holds a 4-sample delay line and a fractional phase `mu` that
//! advances by `step = input_rate / output_rate` per produced output sample.
//! State persists across [`FarrowResampler::process_block`] calls, so feeding
//! the same input split across blocks yields the same output as feeding it in
//! one block (block-split continuity).
//!
//! ## Tap convention
//!
//! `delay = [x_0, x_1, x_2, x_3]` is ordered from oldest (`x_0`) to newest
//! (`x_3`). The polynomial output lies between `x_1` and `x_2`:
//! `mu = 0` returns `x_1` exactly and `mu = 1` returns `x_2` exactly.
//! Group delay is therefore 1–2 input samples; the first ~2 outputs after
//! [`FarrowResampler::reset`] reflect a zero-padded delay line and should
//! be discarded by callers needing exact alignment.

#![allow(dead_code)]

use num_complex::Complex;
use num_traits::Float;

/// Number of taps in the cubic Lagrange Farrow kernel.
const NUM_TAPS: usize = 4;

/// Number of polynomial orders in the Farrow kernel (constant through cubic).
const POLY_ROWS: usize = 4;

/// Streaming cubic-Lagrange Farrow resampler for `Complex<T>` IQ samples.
///
/// Holds a 4-sample delay line and a fractional phase that persist across
/// [`process_block`](Self::process_block) calls. Construct via
/// [`cubic_lagrange`](Self::cubic_lagrange); reset state for an independent
/// stream with [`reset`](Self::reset).
#[derive(Clone)]
pub struct FarrowResampler<T: Float> {
    delay: [Complex<T>; NUM_TAPS],
    head: usize,
    mu: T,
    step: T,
    coeffs: [[T; NUM_TAPS]; POLY_ROWS],
}

impl<T: Float> FarrowResampler<T> {
    /// Builds a cubic-Lagrange Farrow resampler stepping from `input_rate` to
    /// `output_rate`. Rates may be in any consistent unit (Hz, samples per
    /// symbol, normalised) — only their ratio is retained.
    ///
    /// # Panics
    ///
    /// Panics if either `input_rate` or `output_rate` is not strictly positive.
    pub fn cubic_lagrange(input_rate: T, output_rate: T) -> Self {
        assert!(input_rate > T::zero(), "input_rate must be positive");
        assert!(output_rate > T::zero(), "output_rate must be positive");
        Self {
            delay: zero_delay::<T>(),
            head: 0,
            mu: T::zero(),
            step: input_rate / output_rate,
            coeffs: cubic_lagrange_coeffs::<T>(),
        }
    }

    /// Clears the delay line and fractional phase so this instance can be
    /// reused on an independent stream. The resampling ratio is preserved.
    pub fn reset(&mut self) {
        self.delay = zero_delay::<T>();
        self.head = 0;
        self.mu = T::zero();
    }

    /// Resamples `input` and returns the produced samples in a new `Vec`.
    /// Streaming state advances so subsequent calls continue the same stream.
    pub fn process_block(&mut self, input: &[Complex<T>]) -> Vec<Complex<T>> {
        let mut output = Vec::with_capacity(self.capacity_hint(input.len()));
        self.process_block_into(input, &mut output);
        output
    }

    /// Resamples `input`, appending the produced samples to `output` without
    /// clearing pre-existing entries or reallocating when capacity allows.
    pub fn process_block_into(&mut self, input: &[Complex<T>], output: &mut Vec<Complex<T>>) {
        let one = T::one();
        for sample in input {
            self.push_sample(*sample);
            while self.mu < one {
                output.push(self.evaluate(self.mu));
                self.mu = self.mu + self.step;
            }
            self.mu = self.mu - one;
        }
    }

    /// Returns a rough output-length estimate for capacity allocation.
    fn capacity_hint(&self, input_len: usize) -> usize {
        let step_f64 = self.step.to_f64().unwrap_or(1.0).max(f64::MIN_POSITIVE);
        ((input_len as f64) / step_f64).ceil() as usize + 1
    }

    /// Pushes a sample into the delay ring, making it the newest tap (`x_3`).
    fn push_sample(&mut self, sample: Complex<T>) {
        self.head = (self.head + 1) % NUM_TAPS;
        self.delay[self.head] = sample;
    }

    /// Maps tap index (`k=0` oldest, `k=NUM_TAPS-1` newest) to a ring slot.
    fn slot(&self, k: usize) -> usize {
        (self.head + 1 + k) % NUM_TAPS
    }

    /// Returns `sum_k coeffs[power][k] * delay[slot(k)]` — one FIR pass at a
    /// single polynomial order.
    fn sum_at_order(&self, power: usize) -> Complex<T> {
        let mut acc = Complex::new(T::zero(), T::zero());
        for k in 0..NUM_TAPS {
            acc = acc + self.delay[self.slot(k)] * self.coeffs[power][k];
        }
        acc
    }

    /// Evaluates the cubic-Lagrange polynomial at fractional phase `mu` using
    /// Horner's method across the polynomial-FIR sum.
    fn evaluate(&self, mu: T) -> Complex<T> {
        let mut acc = self.sum_at_order(3);
        acc = acc * mu + self.sum_at_order(2);
        acc = acc * mu + self.sum_at_order(1);
        acc = acc * mu + self.sum_at_order(0);
        acc
    }

    /// Test-only accessor for the fractional phase state.
    #[cfg(test)]
    pub(super) fn mu_for_test(&self) -> T {
        self.mu
    }
}

/// Returns a zero-initialised IQ delay line.
fn zero_delay<T: Float>() -> [Complex<T>; NUM_TAPS] {
    [Complex::new(T::zero(), T::zero()); NUM_TAPS]
}

/// Builds the cubic-Lagrange coefficient table for a centred 4-tap Farrow
/// kernel. Rows index powers of `mu` (0..=3); columns index taps with `k=0`
/// oldest and `k=3` newest. At `mu=0` the polynomial returns `x_1` exactly;
/// at `mu=1` it returns `x_2` exactly.
fn cubic_lagrange_coeffs<T: Float>() -> [[T; NUM_TAPS]; POLY_ROWS] {
    let zero = T::zero();
    let one = T::one();
    let half = T::from(0.5).unwrap();
    let third = T::from(1.0 / 3.0).unwrap();
    let sixth = T::from(1.0 / 6.0).unwrap();
    [
        [zero,   one,   zero,  zero ],
        [-third, -half, one,   -sixth],
        [half,   -one,  half,  zero ],
        [-sixth, half,  -half, sixth],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fft::fft;
    use std::f64::consts::PI;

    #[test]
    #[should_panic(expected = "input_rate must be positive")]
    fn test_constructor_panics_on_zero_input_rate() {
        FarrowResampler::<f64>::cubic_lagrange(0.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "output_rate must be positive")]
    fn test_constructor_panics_on_negative_output_rate() {
        FarrowResampler::<f64>::cubic_lagrange(1.0, -1.0);
    }

    #[test]
    fn test_empty_input_no_op() {
        let mut a = FarrowResampler::<f64>::cubic_lagrange(1.0, 1.5);
        let mut b = FarrowResampler::<f64>::cubic_lagrange(1.0, 1.5);
        let input: Vec<Complex<f64>> = (0..20)
            .map(|i| Complex::new((i as f64).sin(), (i as f64).cos()))
            .collect();

        let direct = a.process_block(&input);
        let empty = b.process_block(&[]);
        let after_empty = b.process_block(&input);

        assert!(empty.is_empty());
        assert_eq!(direct, after_empty);
    }

    #[test]
    fn test_dc_input_yields_dc_output() {
        let dc = Complex::new(0.7, -0.3);
        let input = vec![dc; 200];
        for (input_rate, output_rate) in [(1.0, 1.0), (1.0, 1.5), (1.0, 0.8)] {
            let mut resampler = FarrowResampler::<f64>::cubic_lagrange(input_rate, output_rate);
            let output = resampler.process_block(&input);
            for sample in output.iter().skip(10) {
                assert!((sample.re - dc.re).abs() < 1e-12,
                    "rates=({},{}) sample={:?}", input_rate, output_rate, sample);
                assert!((sample.im - dc.im).abs() < 1e-12,
                    "rates=({},{}) sample={:?}", input_rate, output_rate, sample);
            }
        }
    }

    #[test]
    fn test_identity_at_unit_ratio() {
        let mut resampler = FarrowResampler::<f64>::cubic_lagrange(1.0, 1.0);
        let input: Vec<Complex<f64>> = (0..30)
            .map(|i| Complex::new(i as f64, -(i as f64) * 0.5))
            .collect();
        let output = resampler.process_block(&input);

        assert_eq!(output.len(), input.len());
        for k in 2..input.len() {
            assert!((output[k].re - input[k - 2].re).abs() < 1e-12);
            assert!((output[k].im - input[k - 2].im).abs() < 1e-12);
        }
    }

    #[test]
    fn test_block_split_continuity() {
        let input: Vec<Complex<f64>> = (0..1024)
            .map(|i| Complex::new((i as f64 * 0.1).sin(), (i as f64 * 0.13).cos()))
            .collect();

        for (input_rate, output_rate) in [(1.0, 1.5), (1.0, 2.0), (1.0, 0.8)] {
            for split in [1, 100, 400, 1000] {
                let mut full = FarrowResampler::<f64>::cubic_lagrange(input_rate, output_rate);
                let full_out = full.process_block(&input);

                let mut split_resampler = FarrowResampler::<f64>::cubic_lagrange(input_rate, output_rate);
                let mut split_out = split_resampler.process_block(&input[..split]);
                split_out.extend(split_resampler.process_block(&input[split..]));

                assert_eq!(full_out.len(), split_out.len(),
                    "len mismatch rates=({},{}) split={}", input_rate, output_rate, split);
                for (k, (a, b)) in full_out.iter().zip(split_out.iter()).enumerate() {
                    assert!((a.re - b.re).abs() < 1e-12,
                        "re mismatch rates=({},{}) split={} k={} {} vs {}",
                        input_rate, output_rate, split, k, a.re, b.re);
                    assert!((a.im - b.im).abs() < 1e-12,
                        "im mismatch rates=({},{}) split={} k={} {} vs {}",
                        input_rate, output_rate, split, k, a.im, b.im);
                }
            }
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let input: Vec<Complex<f64>> = (0..200)
            .map(|i| Complex::new((i as f64 * 0.07).sin(), (i as f64 * 0.11).cos()))
            .collect();

        let mut resampler = FarrowResampler::<f64>::cubic_lagrange(1.0, 1.5);
        let first = resampler.process_block(&input);
        resampler.reset();
        let second = resampler.process_block(&input);

        assert_eq!(first, second);
    }

    #[test]
    fn test_output_length_within_one() {
        for (input_rate, output_rate) in [(1.0, 1.5), (1.0, 2.0), (1.0, 0.8), (1.0, 3.7)] {
            for input_len in [128_usize, 1024] {
                let mut resampler = FarrowResampler::<f64>::cubic_lagrange(input_rate, output_rate);
                let input = vec![Complex::new(0.0, 0.0); input_len];
                let output = resampler.process_block(&input);
                let expected = ((input_len as f64) * output_rate / input_rate).round() as i64;
                let delta = (output.len() as i64 - expected).abs();
                assert!(delta <= 1,
                    "rates=({},{}) input_len={} output_len={} expected={}",
                    input_rate, output_rate, input_len, output.len(), expected);
            }
        }
    }

    #[test]
    fn test_process_block_into_appends() {
        let mut resampler = FarrowResampler::<f64>::cubic_lagrange(1.0, 1.5);
        let input = vec![Complex::new(1.0, 0.0); 10];
        let sentinel_a = Complex::new(99.0, -99.0);
        let sentinel_b = Complex::new(-7.0, 11.0);
        let mut output = vec![sentinel_a, sentinel_b];

        resampler.process_block_into(&input, &mut output);

        assert_eq!(output[0], sentinel_a);
        assert_eq!(output[1], sentinel_b);
        assert!(output.len() > 2);
    }

    #[test]
    fn test_mu_stays_in_unit_interval() {
        for step_input_rate in [0.4_f64, 1.0, 1.5, 2.5, 4.0] {
            let mut resampler = FarrowResampler::<f64>::cubic_lagrange(step_input_rate, 1.0);
            for i in 0..200 {
                let sample = Complex::new((i as f64).sin(), (i as f64).cos());
                resampler.process_block(&[sample]);
                let mu = resampler.mu_for_test();
                assert!(mu >= 0.0,
                    "step={} i={} mu={}", step_input_rate, i, mu);
                assert!(mu < step_input_rate + 1e-9,
                    "step={} i={} mu={}", step_input_rate, i, mu);
            }
        }
    }

    #[test]
    fn test_upsample_2x_sinusoid_peak_bin() {
        let n = 1024;
        let f_norm_input = 0.1;
        let input: Vec<Complex<f64>> = (0..n)
            .map(|i| {
                let phase = 2.0 * PI * f_norm_input * (i as f64);
                Complex::new(phase.cos(), phase.sin())
            })
            .collect();

        let mut resampler = FarrowResampler::<f64>::cubic_lagrange(1.0, 2.0);
        let output = resampler.process_block(&input);

        let warm_up = 64;
        let fft_size = 1024_usize.min(output.len() - warm_up);
        let mut spectrum: Vec<Complex<f64>> = output[warm_up..warm_up + fft_size].to_vec();
        fft(&mut spectrum);

        let (peak_bin, _) = spectrum.iter().enumerate()
            .max_by(|left, right| left.1.norm().partial_cmp(&right.1.norm()).unwrap())
            .unwrap();

        let f_norm_output = f_norm_input * 0.5;
        let expected_bin = (f_norm_output * fft_size as f64).round() as i64;
        let delta = (peak_bin as i64 - expected_bin).abs();
        assert!(delta <= 1,
            "Peak bin {} not within 1 of expected {} (fft_size={})",
            peak_bin, expected_bin, fft_size);
    }

    #[test]
    fn test_farrow_resample_4sps_to_1p5sps() {
        use crate::complex_vec::ComplexVec;
        use crate::fft::{fft, fftfreqs, fftshift};
        use crate::generate::psk_carrier::PskCarrier;
        use crate::mod_type::ModType;
        use crate::test_utils::should_plot;
        use crate::vector_ops;

        let input_rate_hz = 4e6_f64;
        let output_rate_hz = 1.5e6_f64;
        let symbol_rate_hz = 1e6_f64;
        let block_size: usize = 4096;

        let mut carrier = PskCarrier::new(
            input_rate_hz,
            symbol_rate_hz,
            ModType::_QPSK,
            0.35_f64,
            block_size,
            51,
            Some(7),
        );
        let signal = carrier.generate_block();

        let input_slice: Vec<Complex<f64>> = signal.iter().cloned().collect();
        let mut resampler = FarrowResampler::<f64>::cubic_lagrange(input_rate_hz, output_rate_hz);
        let resampled = resampler.process_block(&input_slice);

        let expected_len = (block_size as f64 * output_rate_hz / input_rate_hz).round() as i64;
        let length_delta = (resampled.len() as i64 - expected_len).abs();
        assert!(
            length_delta <= 1,
            "resampled length {} not within 1 of expected {}",
            resampled.len(),
            expected_len
        );

        if !should_plot() {
            println!("Skipping farrow resample plot (set PLOT=true to enable)");
            return;
        }

        let mut input_fft_data: Vec<Complex<f64>> = signal.iter().cloned().collect();
        fft(&mut input_fft_data[..]);
        let mut input_fft = ComplexVec::from_vec(input_fft_data);
        let mut input_db: Vec<f64> = vector_ops::to_db(&input_fft.abs());
        fftshift::<f64>(&mut input_db);
        let input_freqs: Vec<f64> =
            fftfreqs::<f64>(-input_rate_hz / 2.0, input_rate_hz / 2.0, input_db.len());

        let mut output_fft_data = resampled.clone();
        fft(&mut output_fft_data[..]);
        let mut output_fft = ComplexVec::from_vec(output_fft_data);
        let mut output_db: Vec<f64> = vector_ops::to_db(&output_fft.abs());
        fftshift::<f64>(&mut output_db);
        let output_freqs: Vec<f64> =
            fftfreqs::<f64>(-output_rate_hz / 2.0, output_rate_hz / 2.0, output_db.len());

        use crate::plot::plot_spectrum_pair;
        plot_spectrum_pair(
            (&input_freqs, &input_db, "4 sps input @ 4 MHz"),
            (&output_freqs, &output_db, "1.5 sps output @ 1.5 MHz"),
            "Farrow downsample: 4 → 1.5 samples per symbol",
        );
    }

    #[test]
    fn test_polynomial_coefficients_pass_dc() {
        let coeffs = cubic_lagrange_coeffs::<f64>();
        for mu_index in 0..21 {
            let mu = mu_index as f64 / 20.0;
            let mut sum_at_mu = 0.0;
            for power in (0..POLY_ROWS).rev() {
                let row_sum: f64 = coeffs[power].iter().sum();
                sum_at_mu = sum_at_mu * mu + row_sum;
            }
            assert!((sum_at_mu - 1.0).abs() < 1e-12,
                "Tap-sum at mu={} = {} (expected 1.0)", mu, sum_at_mu);
        }
    }
}
