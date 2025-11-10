#![allow(dead_code)]

use std::any::TypeId;
use std::cmp::PartialOrd;

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

pub fn to_linear<T: Float>(vector: &[T]) -> Vec<T> {
    let tenth = T::from(1.0 / 10.0).unwrap();
    vector.iter().map(|val| tenth.powf(*val * tenth)).collect()
}

pub fn add<T>(a: &[T], b: &[T]) -> Vec<T>
where
    T: std::ops::Add<Output = T> + Copy,
{
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| *x + *y).collect()
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vectors() {
        let a = vec![1, 2, 3];
        let b = vec![4, 5, 6];
        let c = add(&a, &b);
        assert_eq!(c, vec![5, 7, 9]);
    }
}