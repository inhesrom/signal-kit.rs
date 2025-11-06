#![allow(dead_code)]

use num_traits::Float;
use std::f64::consts::PI;

/// Window function types for spectral analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WindowType {
    /// Rectangular window (no windowing)
    Rectangular,
    /// Hann window (raised cosine)
    Hann,
    /// Hamming window (modified raised cosine)
    Hamming,
    /// Blackman window (three-term cosine)
    Blackman,
}

/// Generate a window function of the specified type and size
///
/// # Arguments
/// * `window_type` - Type of window to generate
/// * `size` - Number of points in the window
///
/// # Returns
/// Vector containing the window coefficients
pub fn generate_window<T: Float>(window_type: WindowType, size: usize) -> Vec<T> {
    match window_type {
        WindowType::Rectangular => rectangular_window(size),
        WindowType::Hann => hann_window(size),
        WindowType::Hamming => hamming_window(size),
        WindowType::Blackman => blackman_window(size),
    }
}

/// Generate a rectangular window (all ones)
fn rectangular_window<T: Float>(size: usize) -> Vec<T> {
    vec![T::one(); size]
}

/// Generate a Hann window
///
/// w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
fn hann_window<T: Float>(size: usize) -> Vec<T> {
    if size == 1 {
        return vec![T::one()];
    }

    let pi = T::from(PI).unwrap();
    let two = T::from(2.0).unwrap();
    let half = T::from(0.5).unwrap();
    let n_minus_1 = T::from(size - 1).unwrap();

    (0..size)
        .map(|i| {
            let t = T::from(i).unwrap();
            let arg = (two * pi * t) / n_minus_1;
            half * (T::one() - arg.cos())
        })
        .collect()
}

/// Generate a Hamming window
///
/// w[n] = 0.54 - 0.46 * cos(2*pi*n / (N-1))
fn hamming_window<T: Float>(size: usize) -> Vec<T> {
    if size == 1 {
        return vec![T::one()];
    }

    let pi = T::from(PI).unwrap();
    let two = T::from(2.0).unwrap();
    let alpha = T::from(0.54).unwrap();
    let beta = T::from(0.46).unwrap();
    let n_minus_1 = T::from(size - 1).unwrap();

    (0..size)
        .map(|i| {
            let t = T::from(i).unwrap();
            let arg = (two * pi * t) / n_minus_1;
            alpha - beta * arg.cos()
        })
        .collect()
}

/// Generate a Blackman window
///
/// w[n] = 0.42 - 0.5 * cos(2*pi*n / (N-1)) + 0.08 * cos(4*pi*n / (N-1))
fn blackman_window<T: Float>(size: usize) -> Vec<T> {
    if size == 1 {
        return vec![T::one()];
    }

    let pi = T::from(PI).unwrap();
    let two = T::from(2.0).unwrap();
    let four = T::from(4.0).unwrap();
    let a0 = T::from(0.42).unwrap();
    let a1 = T::from(0.5).unwrap();
    let a2 = T::from(0.08).unwrap();
    let n_minus_1 = T::from(size - 1).unwrap();

    (0..size)
        .map(|i| {
            let t = T::from(i).unwrap();
            let arg1 = (two * pi * t) / n_minus_1;
            let arg2 = (four * pi * t) / n_minus_1;
            a0 - a1 * arg1.cos() + a2 * arg2.cos()
        })
        .collect()
}

/// Calculate the energy (sum of squares) of a window
pub fn window_energy<T: Float>(window: &[T]) -> T {
    window.iter().fold(T::zero(), |acc, &w| acc + w * w)
}

/// Calculate the sum of window coefficients
pub fn window_sum<T: Float>(window: &[T]) -> T {
    window.iter().fold(T::zero(), |acc, &w| acc + w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_window_properties() {
        // Test basic properties of windows
        let size = 256;

        // Rectangular window
        let rect: Vec<f64> = generate_window(WindowType::Rectangular, size);
        assert_eq!(rect.len(), size);
        assert!((rect[0] - 1.0).abs() < 1e-10);
        assert!((rect[size - 1] - 1.0).abs() < 1e-10);

        // Hann window
        let hann: Vec<f64> = generate_window(WindowType::Hann, size);
        assert_eq!(hann.len(), size);
        assert!(hann[0].abs() < 1e-10, "Hann window should start near 0");
        assert!(hann[size - 1].abs() < 1e-10, "Hann window should end near 0");
        assert!(hann[size / 2] > 0.99, "Hann window peak should be near 1");

        // Hamming window
        let hamming: Vec<f64> = generate_window(WindowType::Hamming, size);
        assert_eq!(hamming.len(), size);
        assert!((hamming[0] - 0.08).abs() < 0.01, "Hamming window should start near 0.08");
        assert!(hamming[size / 2] > 0.99, "Hamming window peak should be near 1");

        // Blackman window
        let blackman: Vec<f64> = generate_window(WindowType::Blackman, size);
        assert_eq!(blackman.len(), size);
        assert!(blackman[0].abs() < 0.01, "Blackman window should start near 0");
        assert!(blackman[size - 1].abs() < 0.01, "Blackman window should end near 0");
    }

    #[test]
    fn test_window_energy_sum() {
        let size = 128;

        // Rectangular window
        let rect: Vec<f64> = generate_window(WindowType::Rectangular, size);
        let rect_energy = window_energy(&rect);
        let rect_sum = window_sum(&rect);
        assert!((rect_energy - size as f64).abs() < 1e-10);
        assert!((rect_sum - size as f64).abs() < 1e-10);

        // Hann window
        let hann: Vec<f64> = generate_window(WindowType::Hann, size);
        let hann_energy = window_energy(&hann);
        let hann_sum = window_sum(&hann);
        // Hann window sum should be approximately N/2
        assert!((hann_sum - (size as f64 / 2.0)).abs() < 1.0);
        // Hann window energy should be approximately 3N/8
        assert!((hann_energy - (3.0 * size as f64 / 8.0)).abs() < 1.0);
    }

    #[test]
    fn test_window_shapes() {
        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping window shapes plot (set TEST_PLOT=true to enable)");
            return;
        }

        println!("\n=== Window Shapes Test ===");

        let size = 256;
        let indices: Vec<f64> = (0..size).map(|i| i as f64).collect();

        // Generate all window types
        let rect: Vec<f64> = generate_window(WindowType::Rectangular, size);
        let hann: Vec<f64> = generate_window(WindowType::Hann, size);
        let hamming: Vec<f64> = generate_window(WindowType::Hamming, size);
        let blackman: Vec<f64> = generate_window(WindowType::Blackman, size);

        println!("Generated {} point windows", size);

        // Plot using plotly
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_rect = Scatter::new(indices.clone(), rect)
            .mode(Mode::Lines)
            .name("Rectangular");
        let trace_hann = Scatter::new(indices.clone(), hann)
            .mode(Mode::Lines)
            .name("Hann");
        let trace_hamming = Scatter::new(indices.clone(), hamming)
            .mode(Mode::Lines)
            .name("Hamming");
        let trace_blackman = Scatter::new(indices.clone(), blackman)
            .mode(Mode::Lines)
            .name("Blackman");

        let layout = Layout::new()
            .title("Window Function Comparison")
            .x_axis(Axis::new().title("Sample Index"))
            .y_axis(Axis::new().title("Amplitude"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_rect);
        plot.add_trace(trace_hann);
        plot.add_trace(trace_hamming);
        plot.add_trace(trace_blackman);
        plot.set_layout(layout);

        plot.show();
    }

    #[test]
    fn test_window_frequency_response() {
        let plot = env::var("TEST_PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping window frequency response plot (set TEST_PLOT=true to enable)");
            return;
        }

        println!("\n=== Window Frequency Response Test ===");

        use num_complex::Complex;
        use crate::fft::fft::{fft, fftshift, fftfreqs};
        use crate::vector_ops;

        let size = 1024;

        // Generate windows
        let rect: Vec<f64> = generate_window(WindowType::Rectangular, size);
        let hann: Vec<f64> = generate_window(WindowType::Hann, size);
        let hamming: Vec<f64> = generate_window(WindowType::Hamming, size);
        let blackman: Vec<f64> = generate_window(WindowType::Blackman, size);

        // Compute FFT of each window
        let compute_window_fft = |window: &[f64]| -> Vec<f64> {
            let mut window_complex: Vec<Complex<f64>> = window
                .iter()
                .map(|&w| Complex::new(w, 0.0))
                .collect();
            fft(&mut window_complex);

            // Get magnitude in dB
            let magnitudes: Vec<f64> = window_complex.iter().map(|c| c.norm()).collect();
            let mut mag_db = vector_ops::to_db(&magnitudes);
            fftshift(&mut mag_db);
            mag_db
        };

        let rect_fft = compute_window_fft(&rect);
        let hann_fft = compute_window_fft(&hann);
        let hamming_fft = compute_window_fft(&hamming);
        let blackman_fft = compute_window_fft(&blackman);

        // Generate normalized frequency axis
        let freqs = fftfreqs(-0.5, 0.5, size);

        println!("Computed frequency response for {} point windows", size);

        // Plot
        use plotly::{Plot, Scatter};
        use plotly::common::Mode;
        use plotly::layout::{Axis, Layout};

        let trace_rect = Scatter::new(freqs.clone(), rect_fft)
            .mode(Mode::Lines)
            .name("Rectangular");
        let trace_hann = Scatter::new(freqs.clone(), hann_fft)
            .mode(Mode::Lines)
            .name("Hann");
        let trace_hamming = Scatter::new(freqs.clone(), hamming_fft)
            .mode(Mode::Lines)
            .name("Hamming");
        let trace_blackman = Scatter::new(freqs, blackman_fft)
            .mode(Mode::Lines)
            .name("Blackman");

        let layout = Layout::new()
            .title("Window Function Frequency Response")
            .x_axis(Axis::new().title("Normalized Frequency"))
            .y_axis(Axis::new().title("Magnitude (dB)"))
            .auto_size(true);

        let mut plot = Plot::new();
        plot.add_trace(trace_rect);
        plot.add_trace(trace_hann);
        plot.add_trace(trace_hamming);
        plot.add_trace(trace_blackman);
        plot.set_layout(layout);

        plot.show();
    }
}
