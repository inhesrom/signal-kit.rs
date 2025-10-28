#![allow(dead_code)]

use std::cmp::PartialOrd;

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
