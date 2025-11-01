#![allow(dead_code)]

use num_complex::{Complex};
use num_traits::Float;
use std::ops::{Index, IndexMut};

#[derive(Clone)]
pub struct ComplexVec<T> {
    vector: Vec<Complex<T>>,
}

impl<T> ComplexVec<T>
where
    T: Float,
{
    pub fn new() -> Self {
        ComplexVec {
            vector: Vec::<Complex<T>>::new()
        }
    }

    //Move vector into (ownership transfer)
    pub fn from_vec(vector: Vec<Complex<T>>) -> Self {
        ComplexVec {
            vector,
        }
    }

    pub fn replace_vec(&mut self, vector: Vec<Complex<T>>) {
        self.vector = vector;  // Ownership transferred, old vector dropped
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Complex<T>> {
        self.vector.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, Complex<T>> {
        self.vector.iter_mut()
    }

    pub fn extend<I: IntoIterator<Item = Complex<T>>>(&mut self, iter: I) {
        self.vector.extend(iter);
    }

    pub fn len(&self) -> usize {
        self.vector.len()
    }

    // Returns real vector of sqrt(r^2 + i^2)
    pub fn abs(&mut self) -> Vec<T> {
        self.vector.iter().map(|x| x.norm() ).collect()
    }

    // Normalize to unit magnitude
    pub fn normalize(&mut self) -> ComplexVec<T> {
        ComplexVec::from_vec(
            self.vector.iter()
            .map(|c| {
                let mag = c.norm();
                if mag > T::zero() {
                    *c / mag
                } else {
                    *c
                }
            }).collect()
        )
    }

    // In-place normalize to unit magnitude
    pub fn normalize_inplace(&mut self) {
        for c in self.vector.iter_mut() {
            let mag = c.norm();
            if mag > T::zero() {
                *c = *c / mag;
            }
        }
    }

    pub fn convolve(&self, kernel: &ComplexVec<T>) -> ComplexVec<T> {
        let output_size = self.vector.len() - kernel.vector.len() + 1;
        let mut result = vec![Complex::new(T::zero(), T::zero()); output_size];

        for i in 0..output_size {
            let mut sum = Complex::new(T::zero(), T::zero());
            for j in 0..kernel.vector.len() {
                sum = sum + self.vector[i + j] * kernel.vector[j];
            }
            result[i] = sum;
        }

        ComplexVec::from_vec(result)
    }

    pub fn convolve_inplace(&mut self, kernel: &ComplexVec<T>) {
        let output_size = self.vector.len() - kernel.vector.len() + 1;
        let mut result = vec![Complex::new(T::zero(), T::zero()); output_size];

        for i in 0..output_size {
            let mut sum = Complex::new(T::zero(), T::zero());
            for j in 0..kernel.vector.len() {
                sum = sum + self.vector[i + j] * kernel.vector[j];
            }
            result[i] = sum;
        }

        self.vector = result;  // Replace with new truncated vector
    }
}

impl<T> Index<usize> for ComplexVec<T>
where
    T: Float,
{
    type Output = Complex<T>;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.vector[idx]
    }
}

impl<T> IndexMut<usize> for ComplexVec<T>
where
    T: Float,
{
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        &mut self.vector[idx]
    }
}

impl<T> Extend<Complex<T>> for ComplexVec<T>
where
    T: Float,
{
    fn extend<I: IntoIterator<Item = Complex<T>>>(&mut self, iter: I) {
        self.vector.extend(iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex;

    #[test]
    fn test_new() {
        let cv = ComplexVec::<f64>::new();
        assert_eq!(cv.len(), 0);
    }

    #[test]
    fn test_from_vec() {
        let data = vec![Complex::new(1.0, 2.0), Complex::new(3.0, 4.0)];
        let cv = ComplexVec::from_vec(data);
        assert_eq!(cv.len(), 2);
    }

    #[test]
    fn test_replace_vec() {
        let mut cv = ComplexVec::new();
        let data = vec![Complex::new(1.0, 0.0), Complex::new(0.0, 1.0)];
        cv.replace_vec(data);
        assert_eq!(cv.len(), 2);
    }

    #[test]
    fn test_len() {
        let data = vec![Complex::new(1.0, 2.0); 5];
        let cv = ComplexVec::from_vec(data);
        assert_eq!(cv.len(), 5);
    }

    #[test]
    fn test_abs() {
        let data = vec![
            Complex::new(3.0, 4.0),  // mag = 5.0
            Complex::new(0.0, 1.0),  // mag = 1.0
        ];
        let mut cv = ComplexVec::from_vec(data);
        let magnitudes = cv.abs();

        assert!((magnitudes[0] - 5.0).abs() < 1e-10);
        assert!((magnitudes[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize() {
        let data = vec![
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 12.0),
        ];
        let mut cv = ComplexVec::from_vec(data);
        let normalized = cv.normalize();
        let mags = normalized.vector;

        assert!((mags[0].norm() - 1.0).abs() < 1e-10);
        assert!((mags[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_inplace() {
        let data = vec![
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 12.0),
        ];
        let mut cv = ComplexVec::from_vec(data);
        cv.normalize_inplace();

        assert!((cv.vector[0].norm() - 1.0).abs() < 1e-10);
        assert!((cv.vector[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_zero_magnitude() {
        let data = vec![
            Complex::new(0.0, 0.0),
            Complex::new(3.0, 4.0),
        ];
        let mut cv = ComplexVec::from_vec(data);
        cv.normalize_inplace();

        assert_eq!(cv.vector[0], Complex::new(0.0, 0.0));
        assert!((cv.vector[1].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_convolve() {
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
            Complex::new(5.0, 0.0),
        ];
        let kernel = vec![
            Complex::new(2.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        let result = sig.convolve(&ker);

        assert_eq!(result.len(), 4);
        assert_eq!(result.vector[0], Complex::new(4.0, 0.0));
        assert_eq!(result.vector[1], Complex::new(7.0, 0.0));
        assert_eq!(result.vector[2], Complex::new(10.0, 0.0));
        assert_eq!(result.vector[3], Complex::new(13.0, 0.0));
    }

    #[test]
    fn test_convolve_inplace() {
        let signal = vec![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let kernel = vec![
            Complex::new(1.0, 0.0),
            Complex::new(1.0, 0.0),
        ];

        let mut sig = ComplexVec::from_vec(signal);
        let ker = ComplexVec::from_vec(kernel);
        sig.convolve_inplace(&ker);

        assert_eq!(sig.len(), 2);
        assert_eq!(sig.vector[0], Complex::new(3.0, 0.0));
        assert_eq!(sig.vector[1], Complex::new(5.0, 0.0));
    }

    #[test]
    fn test_convolve_impulse() {
        let signal = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
        ];
        let impulse = vec![Complex::new(1.0, 0.0)];

        let sig = ComplexVec::from_vec(signal.clone());
        let imp = ComplexVec::from_vec(impulse);
        let result = sig.convolve(&imp);

        for i in 0..signal.len() {
            println!("Signal: {}, Convolved Result: {}", signal[i], result[i]);
        }
        assert_eq!(result.len(), 2);
        assert_eq!(result.vector[0], signal[0]);
        assert_eq!(result.vector[1], signal[1]);
    }

    #[test]
    fn test_indexing() {
        let data = vec![
            Complex::new(1.0, 2.0),
            Complex::new(3.0, 4.0),
            Complex::new(5.0, 6.0),
        ];
        let mut cv = ComplexVec::from_vec(data);

        // Test read indexing
        assert_eq!(cv[0], Complex::new(1.0, 2.0));
        assert_eq!(cv[1], Complex::new(3.0, 4.0));
        assert_eq!(cv[2], Complex::new(5.0, 6.0));

        // Test write indexing
        cv[1] = Complex::new(7.0, 8.0);
        assert_eq!(cv[1], Complex::new(7.0, 8.0));
    }
}
