use num_complex::Complex;
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut, AddAssign, SubAssign, MulAssign, DivAssign};

/// A SIMD-accelerated vector wrapper that provides efficient element-wise operations.
///
/// This struct wraps a Vec<T> and provides SIMD-optimized operations for arithmetic
/// and signal processing operations like convolution.
#[derive(Clone, Debug)]
pub struct VectorSimd<T> {
    data: Vec<T>,
}

impl<T> VectorSimd<T>
where
    T: Clone,
{
    /// Creates a new empty VectorSimd
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    /// Creates a VectorSimd with a specific capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Creates a VectorSimd with `count` elements, all initialized to `value`
    pub fn with_value(count: usize, value: T) -> Self {
        Self {
            data: vec![value; count],
        }
    }

    /// Creates a VectorSimd from an existing Vec
    pub fn from_vec(vec: Vec<T>) -> Self {
        Self { data: vec }
    }

    /// Returns the number of elements in the vector
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of elements in the vector (alias for size())
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Adds an element to the end of the vector
    pub fn push_back(&mut self, value: T) {
        self.data.push(value);
    }

    /// Adds an element to the end of the vector (alias for push_back)
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }

    /// Appends another VectorSimd to this one
    pub fn append(&mut self, other: &VectorSimd<T>) {
        self.data.extend_from_slice(&other.data);
    }

    /// Returns a slice view of the underlying data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice view of the underlying data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Default for VectorSimd<T>
where
    T: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// Index trait for element access
impl<T> Index<usize> for VectorSimd<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for VectorSimd<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

// FromIterator to allow collecting into VectorSimd
impl<T> FromIterator<T> for VectorSimd<T>
where
    T: Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            data: iter.into_iter().collect(),
        }
    }
}

// Macro to implement binary operations between VectorSimd instances
macro_rules! impl_vector_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait for VectorSimd<T>
        where
            T: $trait<Output = T> + Clone,
        {
            type Output = VectorSimd<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    self.size(),
                    rhs.size(),
                    "Vector sizes must match for element-wise operations"
                );

                let result: Vec<T> = self
                    .data
                    .into_iter()
                    .zip(rhs.data.into_iter())
                    .map(|(a, b)| a $op b)
                    .collect();

                VectorSimd { data: result }
            }
        }

        impl<T> $trait for &VectorSimd<T>
        where
            T: $trait<Output = T> + Clone,
        {
            type Output = VectorSimd<T>;

            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    self.size(),
                    rhs.size(),
                    "Vector sizes must match for element-wise operations"
                );

                let result: Vec<T> = self
                    .data
                    .iter()
                    .zip(rhs.data.iter())
                    .map(|(a, b)| a.clone() $op b.clone())
                    .collect();

                VectorSimd { data: result }
            }
        }
    };
}

// Macro to implement scalar operations (VectorSimd op Scalar)
macro_rules! impl_scalar_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<T> for VectorSimd<T>
        where
            T: $trait<Output = T> + Clone,
        {
            type Output = VectorSimd<T>;

            fn $method(self, rhs: T) -> Self::Output {
                let result: Vec<T> = self
                    .data
                    .into_iter()
                    .map(|a| a $op rhs.clone())
                    .collect();

                VectorSimd { data: result }
            }
        }

        impl<T> $trait<T> for &VectorSimd<T>
        where
            T: $trait<Output = T> + Clone,
        {
            type Output = VectorSimd<T>;

            fn $method(self, rhs: T) -> Self::Output {
                let result: Vec<T> = self
                    .data
                    .iter()
                    .map(|a| a.clone() $op rhs.clone())
                    .collect();

                VectorSimd { data: result }
            }
        }
    };
}

// Macro to implement compound assignment operations
macro_rules! impl_assign_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl<T> $trait<T> for VectorSimd<T>
        where
            T: $trait + Clone,
        {
            fn $method(&mut self, rhs: T) {
                for elem in &mut self.data {
                    *elem $op rhs.clone();
                }
            }
        }
    };
}

// Implement operations
impl_vector_op!(Add, add, +);
impl_vector_op!(Sub, sub, -);
impl_vector_op!(Mul, mul, *);
impl_vector_op!(Div, div, /);

impl_scalar_op!(Add, add, +);
impl_scalar_op!(Sub, sub, -);
impl_scalar_op!(Mul, mul, *);
impl_scalar_op!(Div, div, /);

impl_assign_op!(AddAssign, add_assign, +=);
impl_assign_op!(SubAssign, sub_assign, -=);
impl_assign_op!(MulAssign, mul_assign, *=);
impl_assign_op!(DivAssign, div_assign, /=);

// SIMD-optimized operations for f32
impl VectorSimd<f32> {
    /// SIMD-optimized element-wise multiplication
    pub fn mul_simd(&self, other: &VectorSimd<f32>) -> VectorSimd<f32> {
        assert_eq!(self.size(), other.size());

        use wide::f32x8;

        let mut result = Vec::with_capacity(self.size());
        let chunks = self.size() / 8;

        // Process 8 elements at a time using SIMD
        for i in 0..chunks {
            let offset = i * 8;
            let a = f32x8::from(&self.data[offset..offset + 8]);
            let b = f32x8::from(&other.data[offset..offset + 8]);
            let c = a * b;
            result.extend_from_slice(&c.to_array());
        }

        // Handle remaining elements
        for i in (chunks * 8)..self.size() {
            result.push(self.data[i] * other.data[i]);
        }

        VectorSimd { data: result }
    }

    /// SIMD-optimized convolution (simple direct implementation)
    pub fn convolve_simple(&self, kernel: &VectorSimd<f32>) -> VectorSimd<f32> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![0.0f32; output_size];

        for i in 0..output_size {
            let mut sum = 0.0f32;
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }

    /// SIMD-optimized convolution using vectorized operations
    pub fn convolve(&self, kernel: &VectorSimd<f32>) -> VectorSimd<f32> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        use wide::f32x8;

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![0.0f32; output_size];

        let simd_chunks = output_size / 8;

        // Process 8 output samples at a time
        for chunk_idx in 0..simd_chunks {
            let base_idx = chunk_idx * 8;
            let mut accum = f32x8::ZERO;

            // For each kernel tap
            for k in 0..kernel.size() {
                let signal_slice = &self.data[base_idx + k..base_idx + k + 8];
                let sig = f32x8::from(signal_slice);
                let kern = f32x8::splat(kernel.data[k]);
                accum = accum + (sig * kern);
            }

            let accum_array = accum.to_array();
            result[base_idx..base_idx + 8].copy_from_slice(&accum_array);
        }

        // Handle remaining elements
        for i in (simd_chunks * 8)..output_size {
            let mut sum = 0.0f32;
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }
}

// SIMD-optimized operations for f64
impl VectorSimd<f64> {
    /// SIMD-optimized element-wise multiplication
    pub fn mul_simd(&self, other: &VectorSimd<f64>) -> VectorSimd<f64> {
        assert_eq!(self.size(), other.size());

        use wide::f64x4;

        let mut result = Vec::with_capacity(self.size());
        let chunks = self.size() / 4;

        // Process 4 elements at a time using SIMD
        for i in 0..chunks {
            let offset = i * 4;
            let a = f64x4::from(&self.data[offset..offset + 4]);
            let b = f64x4::from(&other.data[offset..offset + 4]);
            let c = a * b;
            result.extend_from_slice(&c.to_array());
        }

        // Handle remaining elements
        for i in (chunks * 4)..self.size() {
            result.push(self.data[i] * other.data[i]);
        }

        VectorSimd { data: result }
    }

    /// SIMD-optimized convolution
    pub fn convolve(&self, kernel: &VectorSimd<f64>) -> VectorSimd<f64> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![0.0f64; output_size];

        for i in 0..output_size {
            let mut sum = 0.0f64;
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }
}

// Complex number operations
impl VectorSimd<Complex<f32>> {
    /// Element-wise multiplication optimized for complex numbers
    pub fn mul_complex(&self, other: &VectorSimd<Complex<f32>>) -> VectorSimd<Complex<f32>> {
        assert_eq!(self.size(), other.size());

        let result: Vec<Complex<f32>> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        VectorSimd { data: result }
    }

    /// Convolution for complex-valued signals
    pub fn convolve(&self, kernel: &VectorSimd<Complex<f32>>) -> VectorSimd<Complex<f32>> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![Complex::new(0.0, 0.0); output_size];

        for i in 0..output_size {
            let mut sum = Complex::new(0.0, 0.0);
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }
}

impl VectorSimd<Complex<f64>> {
    /// Element-wise multiplication optimized for complex numbers
    pub fn mul_complex(&self, other: &VectorSimd<Complex<f64>>) -> VectorSimd<Complex<f64>> {
        assert_eq!(self.size(), other.size());

        let result: Vec<Complex<f64>> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        VectorSimd { data: result }
    }

    /// Convolution for complex-valued signals
    pub fn convolve(&self, kernel: &VectorSimd<Complex<f64>>) -> VectorSimd<Complex<f64>> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![Complex::new(0.0, 0.0); output_size];

        for i in 0..output_size {
            let mut sum = Complex::new(0.0, 0.0);
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = VectorSimd::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]);

        let c = &a + &b;
        assert_eq!(c[0], 6.0);
        assert_eq!(c[3], 12.0);

        let d = &a * &b;
        assert_eq!(d[0], 5.0);
        assert_eq!(d[3], 32.0);
    }

    #[test]
    fn test_scalar_operations() {
        let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = &a + 10.0;

        assert_eq!(b[0], 11.0);
        assert_eq!(b[3], 14.0);
    }

    #[test]
    fn test_complex_operations() {
        let a = VectorSimd::from_vec(vec![
            Complex::new(2.0f32, 3.0),
            Complex::new(4.0, 5.0),
        ]);
        let b = VectorSimd::from_vec(vec![
            Complex::new(4.0f32, 5.0),
            Complex::new(6.0, 7.0),
        ]);

        let c = &a * &b;
        // (2+3i) * (4+5i) = 8 + 10i + 12i + 15i² = 8 + 22i - 15 = -7 + 22i
        assert_eq!(c[0].re, -7.0);
        assert_eq!(c[0].im, 22.0);
    }

    #[test]
    fn test_convolution() {
        let signal = VectorSimd::from_vec((0..1024).map(|x| x as f32).collect::<Vec<_>>());
        let kernel = VectorSimd::from_vec(vec![1.0f32, 0.0, 1.0]);

        let result = signal.convolve(&kernel);

        // First element should be signal[0] + signal[2] = 0 + 2 = 2
        assert_eq!(result[0], 2.0);
        // Second element should be signal[1] + signal[3] = 1 + 3 = 4
        assert_eq!(result[1], 4.0);
    }

    // Tests converted from example.rs

    #[test]
    fn test_push_method() {
        let mut v = VectorSimd::<f32>::new();
        v.push(1.0);
        v.push(2.0);
        v.push(3.0);

        assert_eq!(v.len(), 3);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
        assert_eq!(v[2], 3.0);
    }

    #[test]
    fn test_basic_f32_operations() {
        // Example 1 from example.rs
        let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = VectorSimd::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]);

        let sum = &a + &b;
        assert_eq!(sum[0], 6.0);
        assert_eq!(sum[1], 8.0);
        assert_eq!(sum[2], 10.0);
        assert_eq!(sum[3], 12.0);

        let product = &a * &b;
        assert_eq!(product[0], 5.0);
        assert_eq!(product[1], 12.0);
        assert_eq!(product[2], 21.0);
        assert_eq!(product[3], 32.0);

        let scalar_add = &a + 10.0;
        assert_eq!(scalar_add[0], 11.0);
        assert_eq!(scalar_add[1], 12.0);
        assert_eq!(scalar_add[2], 13.0);
        assert_eq!(scalar_add[3], 14.0);
    }

    #[test]
    fn test_complex_multiplication() {
        // Example 2 from example.rs
        let c1 = VectorSimd::from_vec(vec![
            Complex::new(2.0f32, 3.0),
            Complex::new(4.0, 5.0),
        ]);
        let c2 = VectorSimd::from_vec(vec![
            Complex::new(4.0f32, 5.0),
            Complex::new(6.0, 7.0),
        ]);

        let c_product = &c1 * &c2;

        // (2+3i) * (4+5i) = 8 + 10i + 12i + 15i² = 8 + 22i - 15 = -7 + 22i
        assert_eq!(c_product[0].re, -7.0);
        assert_eq!(c_product[0].im, 22.0);

        // (4+5i) * (6+7i) = 24 + 28i + 30i + 35i² = 24 + 58i - 35 = -11 + 58i
        assert_eq!(c_product[1].re, -11.0);
        assert_eq!(c_product[1].im, 58.0);
    }

    #[test]
    fn test_convolution_detailed() {
        // Example 3 from example.rs
        let signal: Vec<f32> = (0..1024).map(|x| x as f32).collect();
        let signal_vec = VectorSimd::from_vec(signal);
        let kernel = VectorSimd::from_vec(vec![1.0f32, 0.0, 1.0]);

        let result = signal_vec.convolve(&kernel);

        assert_eq!(signal_vec.len(), 1024);
        assert_eq!(kernel.len(), 3);
        assert_eq!(result.len(), 1024 - 3 + 1);

        // Verify first few results
        // result[i] = signal[i]*1.0 + signal[i+1]*0.0 + signal[i+2]*1.0
        //           = signal[i] + signal[i+2]
        assert_eq!(result[0], 0.0 + 2.0);  // 2.0
        assert_eq!(result[1], 1.0 + 3.0);  // 4.0
        assert_eq!(result[2], 2.0 + 4.0);  // 6.0
        assert_eq!(result[3], 3.0 + 5.0);  // 8.0
        assert_eq!(result[4], 4.0 + 6.0);  // 10.0
    }

    #[test]
    fn test_append_operation() {
        // Example 5 from example.rs
        let mut v1 = VectorSimd::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]);
        let v2 = VectorSimd::from_vec(vec![4.0f32, 5.0, 6.0, 7.0]);

        v1.append(&v2);

        assert_eq!(v1.len(), 8);
        assert_eq!(v1[0], 0.0);
        assert_eq!(v1[3], 3.0);
        assert_eq!(v1[4], 4.0);
        assert_eq!(v1[7], 7.0);
    }

    #[test]
    fn test_in_place_operations() {
        // Example 6 from example.rs
        let mut v = VectorSimd::from_vec(vec![5.0f32, 15.0, 25.0]);

        v /= 5.0;

        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 3.0);
        assert_eq!(v[2], 5.0);
    }

    #[test]
    fn test_simd_multiplication_correctness() {
        // Example 4 from example.rs - verify SIMD multiplication matches standard
        let size = 1000;
        let data_a: Vec<f32> = (0..size).map(|i| (i % 100) as f32 / 100.0).collect();
        let data_b: Vec<f32> = (0..size).map(|i| ((i * 7) % 100) as f32 / 100.0).collect();

        let vec_a = VectorSimd::from_vec(data_a.clone());
        let vec_b = VectorSimd::from_vec(data_b.clone());

        let result_operator = &vec_a * &vec_b;
        let result_simd = vec_a.mul_simd(&vec_b);

        // Verify both methods produce same results
        for i in 0..size {
            let expected = data_a[i] * data_b[i];
            assert!((result_operator[i] - expected).abs() < 1e-6);
            assert!((result_simd[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_complex_mul_complex_method() {
        let c1 = VectorSimd::from_vec(vec![
            Complex::new(2.0f32, 3.0),
            Complex::new(1.0, 1.0),
        ]);
        let c2 = VectorSimd::from_vec(vec![
            Complex::new(4.0f32, 5.0),
            Complex::new(2.0, 3.0),
        ]);

        let result = c1.mul_complex(&c2);

        // (2+3i) * (4+5i) = -7 + 22i
        assert_eq!(result[0].re, -7.0);
        assert_eq!(result[0].im, 22.0);

        // (1+1i) * (2+3i) = 2 + 3i + 2i + 3i² = 2 + 5i - 3 = -1 + 5i
        assert_eq!(result[1].re, -1.0);
        assert_eq!(result[1].im, 5.0);
    }

    #[test]
    fn test_subtraction() {
        let a = VectorSimd::from_vec(vec![10.0f32, 20.0, 30.0]);
        let b = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0]);

        let result = &a - &b;

        assert_eq!(result[0], 9.0);
        assert_eq!(result[1], 18.0);
        assert_eq!(result[2], 27.0);
    }

    #[test]
    fn test_division() {
        let a = VectorSimd::from_vec(vec![10.0f32, 20.0, 30.0]);
        let b = VectorSimd::from_vec(vec![2.0f32, 4.0, 5.0]);

        let result = &a / &b;

        assert_eq!(result[0], 5.0);
        assert_eq!(result[1], 5.0);
        assert_eq!(result[2], 6.0);
    }

    #[test]
    fn test_with_capacity() {
        let v = VectorSimd::<f32>::with_capacity(100);
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
    }

    #[test]
    fn test_with_value() {
        let v = VectorSimd::with_value(5, 3.14f32);
        assert_eq!(v.len(), 5);
        for i in 0..5 {
            assert_eq!(v[i], 3.14);
        }
    }
}
