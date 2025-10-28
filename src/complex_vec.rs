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
    pub fn inplace_normalize(&mut self) {
        for c in self.vector.iter_mut() {
            let mag = c.norm();
            if mag > T::zero() {
                *c = *c / mag;
            }
        }
    }

}
