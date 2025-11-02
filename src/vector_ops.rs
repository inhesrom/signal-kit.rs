#![allow(dead_code)]

use std::cmp::PartialOrd;
use std::any::TypeId;

use num_traits::Float;

pub fn max<T: Clone + Copy + PartialOrd>(vector: &[T]) -> (usize, T) {
    let mut max_idx = 0;
    let mut max_val = vector[0].clone();

    for (i, &val) in vector.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val.clone();
            max_idx = i;
        }
    }

    (max_idx, max_val)
}

pub fn to_db<T: Float + 'static>(vector: &[T]) -> Vec<T> {
    let mut v = Vec::new();
    v.reserve(vector.len());

    let ten = T::from(10.0).unwrap();
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        v = vector.iter().map(|val| ten * val.log10()).collect();
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        v = vector.iter().map(|val| ten * val.log10()).collect();
    }

    v
}
