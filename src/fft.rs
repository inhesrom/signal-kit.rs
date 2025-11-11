#![allow(dead_code)]

pub mod fft {

    use std::fmt::Debug;
    use std::ops::{DivAssign, RemAssign};
    use num_complex::{Complex};
    use num_traits::{Float, Signed, FromPrimitive};
    use rustfft::FftPlanner;

    pub fn fft<T>(input : &mut [Complex<T>])
    where
        T: Float + RemAssign + DivAssign + Send + Sync + FromPrimitive + Signed + Debug + 'static,
    {
        let mut planner = FftPlanner::<T>::new();
        let fft_forward = planner.plan_fft_forward(input.len());
        fft_forward.process(input);
        scale::<T>(input);
    }

    pub fn ifft<T>(input: &mut [Complex<T>])
    where
        T: Float + Send + Sync + FromPrimitive + Signed + Debug + 'static,
    {
        let mut planner = FftPlanner::<T>::new();
        let fft_inverse = planner.plan_fft_inverse(input.len());
        fft_inverse.process(input);
    }

    pub fn scale<T>(input: &mut [Complex<T>])
    where
        T: Float + RemAssign + DivAssign,
    {
        let nfft = T::from(input.len()).unwrap();
        let scale_val = Complex::<T>::new(
            T::from(1.0).unwrap() / nfft,
            T::from(0.0).unwrap()
        );
        input.iter_mut().for_each(|x| *x = *x * scale_val);
    }

    pub fn fftshift<T>(input_vec: &mut [T]) {
        let n = input_vec.len();
        let mid = (n + 1) / 2;
        input_vec.rotate_left(mid);
    }

    pub fn fftfreqs<T: Float>(start: T, stop: T, num_points: usize) -> Vec<T> {
        let mut v = Vec::with_capacity(num_points);
        let step: T = (stop - start) / (T::from(num_points).unwrap() - T::from(1.0).unwrap());
        for i in 0..num_points {
           v.push(start + (T::from(i).unwrap() * step));
        }
        v
    }

}

#[cfg(test)]
mod tests {
    use num_complex::{Complex};
    use num_traits::FromBytes;
    use num_traits::{Float};
    use std::{env, vec};

    use crate::complex_vec::ComplexVec;
    use crate::generate::cw::CW;
    use crate::fft::*;
    use crate::vector_ops;

    fn assert_near<T: Float>(a: T, b: T, delta: T) {
        assert_eq!((a-b).abs() < delta, true);
    }

    #[test]
    fn test_fft() {
        let blocksize: usize = 2000;
        let freq_hz = 100.0f64;
        let sample_rate_hz = 1000.0f64;

        let mut cw_generator = CW::new(freq_hz, sample_rate_hz, blocksize);
        let mut cw: Vec<Complex<f32>> = cw_generator.generate_block::<f32>();
        fft::fft::<f32>(&mut cw);
        let mut cw_fft = ComplexVec::from_vec(cw);
        let mut cw_fft_abs: Vec<f32> = cw_fft.abs();
        fft::fftshift::<f32>(&mut cw_fft_abs);

        let freqs: Vec<f32> = fft::fftfreqs::<f32>((-sample_rate_hz/2_f64) as f32, (sample_rate_hz/2_f64) as f32, blocksize);

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        println!("PLOT env var is {}", plot);
        if plot.to_lowercase() == "true" {
            use crate::plot::plot_spectrum;
            plot_spectrum(&freqs, &cw_fft_abs, "CW Signal Spectrum");
        }

        let (max_ind, max_val) = vector_ops::max(&cw_fft_abs);
        let max_freq = freqs[max_ind].clone();
        println!("Max (ind,val) is: {max_freq},{max_val}");
        assert_eq!(((max_freq as f64) - freq_hz).abs() < 1.0, true);
    }

    #[test]
    fn test_ifft() {
        let signal_original = vec![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 2.0),
            Complex::new(3.0, 3.0),
            Complex::new(4.0, 4.0),
            Complex::new(5.0, 5.0),
            Complex::new(6.0, 6.0),
            Complex::new(7.0, 7.0),
            Complex::new(8.0, 8.0)
        ];
        let mut signal_modified = signal_original.clone();
        fft::fft(&mut signal_modified);
        fft::ifft(&mut signal_modified);

        for i in 0..signal_original.len() {
            println!("i: {i} -- Original value: {} vs Modified value {}", signal_original[i], signal_modified[i]);
            assert_near::<f32>(signal_original[i].norm(), signal_modified[i].norm(), 1e-3);
        }
    }

    #[test]
    fn test_scale() {
        let mut v: Vec<_> = vec![Complex::<f64>::new(100.0, 100.0); 10];
        fft::scale::<f64>(&mut v);
        assert_eq!(v, vec![Complex::<f64>::new(10.0, 10.0); 10]);
    }

    #[test]
    fn test_fftshift_even() {
        let vec_initial = vec![0, 1, 2, 3, 4, 5, 6, 7];
        let mut vec_shifted = vec_initial.clone();
        fft::fftshift(&mut vec_shifted);
        assert_eq!(vec_shifted, vec![4, 5, 6, 7, 0, 1, 2, 3]);
    }

    #[test]
    fn test_fftshift_odd() {
        let vec_initial = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        let mut vec_shifted = vec_initial.clone();
        fft::fftshift(&mut vec_shifted);
        assert_eq!(vec_shifted, vec![5, 6, 7, 8, 0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_fftshift_cw_even() {
        let vec_cw = vec![Complex::new(1.0_f32, 0.0_f32); 10];
        let mut vec_cv = ComplexVec::<f32>::from_vec(vec_cw);
        fft::fft(&mut vec_cv);
        fft::fftshift(&mut vec_cv);
        vec_cv.iter().for_each(|c| println!("{c}"));
        assert_eq!(vec_cv[5].norm(), 1.0);
    }

    #[test]
    fn test_fftshift_cw_odd() {
        let vec_cw = vec![Complex::new(1.0_f32, 0.0_f32); 9];
        let mut vec_cv = ComplexVec::<f32>::from_vec(vec_cw);
        fft::fft(&mut vec_cv);
        fft::fftshift(&mut vec_cv);
        vec_cv.iter().for_each(|c| println!("{c}"));
        assert_eq!(vec_cv[4].norm(), 1.0);
    }

    #[test]
    fn test_fftfreq() {
        let start = -100.0;
        let stop = 100.0;
        let num_points = 1000;
        let freqs = fft::fftfreqs::<f64>(start, stop, num_points);
        println!("start: {start}, stop: {stop}, num_points: {num_points}");
        assert_eq!(*freqs.first().unwrap(), start);
        assert_eq!(*freqs.last().unwrap(), stop);
    }

}
