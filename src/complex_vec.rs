#![allow(dead_code)]

use num_complex::{Complex};
use num_traits::Float;

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
