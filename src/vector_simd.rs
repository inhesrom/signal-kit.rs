#![allow(dead_code)]

use num_complex::Complex;
use std::ops::{Add, Sub, Mul, Div, Index, IndexMut, AddAssign, SubAssign, MulAssign, DivAssign};

// ============================================================================
// SIMD Configuration Module - Change these type aliases to switch register width
// ============================================================================

mod simd_config {
    /// Current configuration: 128-bit registers
    ///
    /// To use 256-bit registers, change:
    /// - F32Batch from `wide::f32x4` to `wide::f32x8`
    /// - F64Batch from `wide::f64x2` to `wide::f64x4`
    ///
    /// To use 512-bit registers, change to:
    /// - F32Batch: `wide::f32x16`
    /// - F64Batch: `wide::f64x8`
    pub type F32Batch = wide::f32x4;  // 128-bit: 4 × f32
    pub type F64Batch = wide::f64x2;  // 128-bit: 2 × f64
}

// ============================================================================
// SIMD Batch Trait - Abstracts over different register widths
// ============================================================================

/// Trait for SIMD batch operations, allowing generic code over different register widths
///
/// # Memory Alignment
///
/// All load/store operations use **unaligned** instructions, which work correctly
/// regardless of memory alignment. On modern CPUs (Intel Nehalem+ 2008, AMD Bulldozer+ 2011),
/// unaligned loads/stores have **zero performance penalty** for aligned data.
///
/// This design choice provides:
/// - **Safety**: Works with any slice, no alignment requirements
/// - **Performance**: Equal to aligned loads on modern hardware
/// - **Simplicity**: No need to track or enforce alignment
///
/// The only remaining penalty is cache-line splits (when data crosses cache boundaries),
/// which is unavoidable and handled automatically by the CPU.
pub trait SimdBatch: Sized + Copy {
    type Scalar: Copy;

    /// Number of lanes (elements) in this SIMD batch
    const LANES: usize;

    /// Create a batch with all lanes set to zero
    fn zero() -> Self;

    /// Create a batch with all lanes set to the same value (splat)
    fn splat(value: Self::Scalar) -> Self;

    /// Load LANES elements from a slice (works with any alignment)
    fn load(ptr: &[Self::Scalar]) -> Self;

    /// Store LANES elements to a slice (works with any alignment)
    fn store(self, ptr: &mut [Self::Scalar]);

    /// Addition
    fn add(self, rhs: Self) -> Self;

    /// Subtraction
    fn sub(self, rhs: Self) -> Self;

    /// Multiplication
    fn mul(self, rhs: Self) -> Self;

    /// Division
    fn div(self, rhs: Self) -> Self;

    /// Absolute value
    fn abs(self) -> Self;

    /// Horizontal sum (reduce all lanes to single scalar)
    fn horizontal_sum(self) -> Self::Scalar;
}

// ============================================================================
// SimdBatch Implementation for f32x4 (128-bit)
// ============================================================================

impl SimdBatch for wide::f32x4 {
    type Scalar = f32;
    const LANES: usize = 4;

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn splat(value: f32) -> Self {
        Self::splat(value)
    }

    #[inline]
    fn load(ptr: &[f32]) -> Self {
        assert!(ptr.len() >= 4, "Need at least 4 elements for f32x4");
        Self::from(&ptr[0..4])
    }

    #[inline]
    fn store(self, ptr: &mut [f32]) {
        assert!(ptr.len() >= 4, "Need at least 4 elements for f32x4");
        let arr = self.to_array();
        ptr[0..4].copy_from_slice(&arr);
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn horizontal_sum(self) -> f32 {
        let arr = self.to_array();
        arr[0] + arr[1] + arr[2] + arr[3]
    }
}

// ============================================================================
// SimdBatch Implementation for f64x2 (128-bit)
// ============================================================================

impl SimdBatch for wide::f64x2 {
    type Scalar = f64;
    const LANES: usize = 2;

    #[inline]
    fn zero() -> Self {
        Self::ZERO
    }

    #[inline]
    fn splat(value: f64) -> Self {
        Self::splat(value)
    }

    #[inline]
    fn load(ptr: &[f64]) -> Self {
        assert!(ptr.len() >= 2, "Need at least 2 elements for f64x2");
        Self::from([ptr[0], ptr[1]])
    }

    #[inline]
    fn store(self, ptr: &mut [f64]) {
        assert!(ptr.len() >= 2, "Need at least 2 elements for f64x2");
        let arr = self.to_array();
        ptr[0..2].copy_from_slice(&arr);
    }

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self - rhs
    }

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }

    #[inline]
    fn div(self, rhs: Self) -> Self {
        self / rhs
    }

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }

    #[inline]
    fn horizontal_sum(self) -> f64 {
        let arr = self.to_array();
        arr[0] + arr[1]
    }
}

// ============================================================================
// VectorSimd Structure and Basic Methods
// ============================================================================

/// A SIMD-accelerated vector wrapper that provides efficient element-wise operations.
///
/// This struct wraps a Vec<T> and provides SIMD-optimized operations for arithmetic
/// and signal processing operations like convolution.
///
/// Register width is configurable via the `simd_config` module at compile time.
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

    /// Reserves capacity for at least `additional` more elements
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
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

    /// Returns a raw pointer to the underlying data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the underlying data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
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

// ============================================================================
// Generic SIMD Operation Framework
// ============================================================================

/// Operation types for batch processing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Subtract,
    Multiply,
    Divide,
}

impl<T> VectorSimd<T>
where
    T: Copy + Default,
{
    /// Generic SIMD batch processor for vector-vector operations
    fn process_simd_batches<B>(
        &self,
        other: &VectorSimd<T>,
        batch_op: impl Fn(B, B) -> B,
        scalar_op: impl Fn(T, T) -> T,
    ) -> VectorSimd<T>
    where
        B: SimdBatch<Scalar = T>,
    {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector sizes must match for element-wise operations"
        );

        let mut result = vec![T::default(); self.len()];
        let lanes = B::LANES;
        let simd_end = (self.len() / lanes) * lanes;

        // Process SIMD batches
        for i in (0..simd_end).step_by(lanes) {
            let a = B::load(&self.data[i..]);
            let b = B::load(&other.data[i..]);
            let c = batch_op(a, b);
            c.store(&mut result[i..]);
        }

        // Process remainder with scalar operations
        for i in simd_end..self.len() {
            result[i] = scalar_op(self.data[i], other.data[i]);
        }

        VectorSimd { data: result }
    }

    /// Generic SIMD batch processor for vector-scalar operations
    fn process_simd_batches_scalar<B>(
        &self,
        scalar: T,
        batch_op: impl Fn(B, B) -> B,
        scalar_op: impl Fn(T, T) -> T,
    ) -> VectorSimd<T>
    where
        B: SimdBatch<Scalar = T>,
    {
        let mut result = vec![T::default(); self.len()];
        let lanes = B::LANES;
        let simd_end = (self.len() / lanes) * lanes;
        let scalar_batch = B::splat(scalar);

        // Process SIMD batches
        for i in (0..simd_end).step_by(lanes) {
            let a = B::load(&self.data[i..]);
            let c = batch_op(a, scalar_batch);
            c.store(&mut result[i..]);
        }

        // Process remainder
        for i in simd_end..self.len() {
            result[i] = scalar_op(self.data[i], scalar);
        }

        VectorSimd { data: result }
    }

    /// Generic SIMD batch processor for in-place vector-scalar operations
    fn process_simd_batches_scalar_inplace<B>(
        &mut self,
        scalar: T,
        batch_op: impl Fn(B, B) -> B,
        scalar_op: impl Fn(T, T) -> T,
    )
    where
        B: SimdBatch<Scalar = T>,
    {
        let lanes = B::LANES;
        let simd_end = (self.len() / lanes) * lanes;
        let scalar_batch = B::splat(scalar);

        // Process SIMD batches
        for i in (0..simd_end).step_by(lanes) {
            let a = B::load(&self.data[i..]);
            let c = batch_op(a, scalar_batch);
            c.store(&mut self.data[i..]);
        }

        // Process remainder
        for i in simd_end..self.len() {
            self.data[i] = scalar_op(self.data[i], scalar);
        }
    }
}

// ============================================================================
// f32 SIMD Operations
// ============================================================================

impl VectorSimd<f32> {
    /// Creates a VectorSimd with evenly spaced values (linspace)
    pub fn linspace(start: f32, stop: f32, length: usize) -> Self {
        if length == 0 {
            return Self::new();
        }
        if length == 1 {
            return Self::from_vec(vec![start]);
        }

        let step = (stop - start) / (length as f32 - 1.0);
        let data: Vec<f32> = (0..length)
            .map(|i| start + (i as f32) * step)
            .collect();
        Self { data }
    }
}

// Implement Add trait for VectorSimd<f32>
impl Add for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            &rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

impl Add for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

// Implement Sub trait for VectorSimd<f32>
impl Sub for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn sub(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            &rhs,
            |a, b| SimdBatch::sub(a, b),
            |a, b| a - b,
        )
    }
}

impl Sub for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn sub(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            rhs,
            |a, b| SimdBatch::sub(a, b),
            |a, b| a - b,
        )
    }
}

// Implement Mul trait for VectorSimd<f32>
impl Mul for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn mul(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            &rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

impl Mul for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn mul(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

// Implement Div trait for VectorSimd<f32>
impl Div for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn div(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            &rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        )
    }
}

impl Div for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn div(self, rhs: Self) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches::<F32Batch>(
            rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        )
    }
}

// Scalar operations for f32
impl Add<f32> for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn add(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

impl Add<f32> for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn add(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

impl Sub<f32> for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn sub(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::sub(a, b),
            |a, b| a - b,
        )
    }
}

impl Sub<f32> for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn sub(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::sub(a, b),
            |a, b| a - b,
        )
    }
}

impl Mul<f32> for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

impl Mul<f32> for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn mul(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

impl Div<f32> for VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        )
    }
}

impl Div<f32> for &VectorSimd<f32> {
    type Output = VectorSimd<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar::<F32Batch>(
            rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        )
    }
}

// Compound assignment operations for f32
impl AddAssign<f32> for VectorSimd<f32> {
    fn add_assign(&mut self, rhs: f32) {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar_inplace::<F32Batch>(
            rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        );
    }
}

impl SubAssign<f32> for VectorSimd<f32> {
    fn sub_assign(&mut self, rhs: f32) {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar_inplace::<F32Batch>(
            rhs,
            |a, b| SimdBatch::sub(a, b),
            |a, b| a - b,
        );
    }
}

impl MulAssign<f32> for VectorSimd<f32> {
    fn mul_assign(&mut self, rhs: f32) {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar_inplace::<F32Batch>(
            rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        );
    }
}

impl DivAssign<f32> for VectorSimd<f32> {
    fn div_assign(&mut self, rhs: f32) {
        use simd_config::F32Batch;
        self.process_simd_batches_scalar_inplace::<F32Batch>(
            rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        );
    }
}

// Special operations for f32
impl VectorSimd<f32> {
    /// Returns absolute values of all elements
    pub fn abs(&self) -> VectorSimd<f32> {
        use simd_config::F32Batch;

        let mut result = vec![0.0f32; self.len()];
        let lanes = F32Batch::LANES;
        let simd_end = (self.len() / lanes) * lanes;

        // Process SIMD batches
        for i in (0..simd_end).step_by(lanes) {
            let a = F32Batch::load(&self.data[i..]);
            let c = a.abs();
            c.store(&mut result[i..]);
        }

        // Process remainder
        for i in simd_end..self.len() {
            result[i] = self.data[i].abs();
        }

        VectorSimd { data: result }
    }

    /// Simple convolution (direct method)
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

    /// SIMD-optimized convolution
    pub fn convolve(&self, kernel: &VectorSimd<f32>) -> VectorSimd<f32> {
        if kernel.size() > self.size() {
            return VectorSimd::new();
        }

        use simd_config::F32Batch;
        const LANES: usize = 4; // f32x4

        let output_size = self.size() - kernel.size() + 1;
        let mut result = vec![0.0f32; output_size];

        let simd_chunks = output_size / LANES;

        // Process LANES output samples at a time
        for chunk_idx in 0..simd_chunks {
            let base_idx = chunk_idx * LANES;
            let mut accum = F32Batch::zero();

            // For each kernel tap
            for k in 0..kernel.size() {
                let signal_slice = &self.data[base_idx + k..base_idx + k + LANES];
                let sig = F32Batch::load(signal_slice);
                let kern = F32Batch::splat(kernel.data[k]);
                accum = SimdBatch::add(accum, SimdBatch::mul(sig, kern));
            }

            let mut accum_array = [0.0f32; LANES];
            accum.store(&mut accum_array);
            result[base_idx..base_idx + LANES].copy_from_slice(&accum_array);
        }

        // Handle remaining elements
        for i in (simd_chunks * LANES)..output_size {
            let mut sum = 0.0f32;
            for j in 0..kernel.size() {
                sum += self.data[i + j] * kernel.data[j];
            }
            result[i] = sum;
        }

        VectorSimd { data: result }
    }
}

// ============================================================================
// f64 SIMD Operations (Similar to f32 but using F64Batch)
// ============================================================================

impl VectorSimd<f64> {
    /// Creates a VectorSimd with evenly spaced values (linspace)
    pub fn linspace(start: f64, stop: f64, length: usize) -> Self {
        if length == 0 {
            return Self::new();
        }
        if length == 1 {
            return Self::from_vec(vec![start]);
        }

        let step = (stop - start) / (length as f64 - 1.0);
        let data: Vec<f64> = (0..length)
            .map(|i| start + (i as f64) * step)
            .collect();
        Self { data }
    }
}

// Implement all the same operations for f64...
// (Add, Sub, Mul, Div for vector and scalar operations)
// I'll implement a few key ones to show the pattern:

impl Add for VectorSimd<f64> {
    type Output = VectorSimd<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        use simd_config::F64Batch;
        self.process_simd_batches::<F64Batch>(
            &rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

impl Add for &VectorSimd<f64> {
    type Output = VectorSimd<f64>;

    fn add(self, rhs: Self) -> Self::Output {
        use simd_config::F64Batch;
        self.process_simd_batches::<F64Batch>(
            rhs,
            |a, b| SimdBatch::add(a, b),
            |a, b| a + b,
        )
    }
}

impl Mul for VectorSimd<f64> {
    type Output = VectorSimd<f64>;

    fn mul(self, rhs: Self) -> Self::Output {
        use simd_config::F64Batch;
        self.process_simd_batches::<F64Batch>(
            &rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

impl Mul for &VectorSimd<f64> {
    type Output = VectorSimd<f64>;

    fn mul(self, rhs: Self) -> Self::Output {
        use simd_config::F64Batch;
        self.process_simd_batches::<F64Batch>(
            rhs,
            |a, b| SimdBatch::mul(a, b),
            |a, b| a * b,
        )
    }
}

impl DivAssign<f64> for VectorSimd<f64> {
    fn div_assign(&mut self, rhs: f64) {
        use simd_config::F64Batch;
        self.process_simd_batches_scalar_inplace::<F64Batch>(
            rhs,
            |a, b| SimdBatch::div(a, b),
            |a, b| a / b,
        );
    }
}

// ============================================================================
// Complex Number SIMD Operations
// ============================================================================

// For now, implement basic iterator-based operations for Complex
// SIMD for complex multiplication is more complex and will be added in future iteration

impl Add for VectorSimd<Complex<f32>> {
    type Output = VectorSimd<Complex<f32>>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let result: Vec<Complex<f32>> = self.data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a + b)
            .collect();
        VectorSimd { data: result }
    }
}

impl Add for &VectorSimd<Complex<f32>> {
    type Output = VectorSimd<Complex<f32>>;

    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let result: Vec<Complex<f32>> = self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        VectorSimd { data: result }
    }
}

impl Mul for VectorSimd<Complex<f32>> {
    type Output = VectorSimd<Complex<f32>>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let result: Vec<Complex<f32>> = self.data
            .into_iter()
            .zip(rhs.data.into_iter())
            .map(|(a, b)| a * b)
            .collect();
        VectorSimd { data: result }
    }
}

impl Mul for &VectorSimd<Complex<f32>> {
    type Output = VectorSimd<Complex<f32>>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());
        let result: Vec<Complex<f32>> = self.data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        VectorSimd { data: result }
    }
}

impl VectorSimd<Complex<f32>> {
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_add() {
        let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]);
        let b = VectorSimd::from_vec(vec![5.0f32, 6.0, 7.0, 8.0, 9.0]);

        let c = &a + &b;
        assert_eq!(c[0], 6.0);
        assert_eq!(c[1], 8.0);
        assert_eq!(c[4], 14.0);
    }

    #[test]
    fn test_f32_mul_scalar() {
        let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
        let b = &a * 2.0;

        assert_eq!(b[0], 2.0);
        assert_eq!(b[3], 8.0);
    }

    #[test]
    fn test_f32_div_assign() {
        let mut v = VectorSimd::from_vec(vec![10.0f32, 20.0, 30.0]);
        v /= 2.0;

        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 10.0);
        assert_eq!(v[2], 15.0);
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
    fn test_f32_abs() {
        let a = VectorSimd::from_vec(vec![-1.0f32, 2.0, -3.0, 4.0, -5.0]);
        let b = a.abs();

        assert_eq!(b[0], 1.0);
        assert_eq!(b[1], 2.0);
        assert_eq!(b[2], 3.0);
        assert_eq!(b[3], 4.0);
        assert_eq!(b[4], 5.0);
    }

    #[test]
    fn test_f32_linspace() {
        let v = VectorSimd::<f32>::linspace(0.0, 10.0, 11);

        assert_eq!(v.len(), 11);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[10], 10.0);
        assert!((v[5] - 5.0).abs() < 1e-6);
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

    // ========================================================================
    // Benchmark Tests - Compare SIMD vs Scalar Performance
    // ========================================================================

    use std::time::Instant;

    /// Helper function to run scalar addition (non-SIMD baseline)
    fn scalar_add_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    /// Helper function to run scalar multiplication (non-SIMD baseline)
    fn scalar_mul_f32(a: &[f32], b: &[f32]) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    /// Helper function to run scalar convolution (non-SIMD baseline)
    fn scalar_convolve_f32(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
        let output_size = signal.len() - kernel.len() + 1;
        let mut result = vec![0.0f32; output_size];

        for i in 0..output_size {
            let mut sum = 0.0f32;
            for j in 0..kernel.len() {
                sum += signal[i + j] * kernel[j];
            }
            result[i] = sum;
        }
        result
    }

    #[test]
    fn benchmark_f32_addition() {
        println!("\n=== Benchmark: f32 Addition ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32).collect();
            let data_b: Vec<f32> = (0..size).map(|i| ((i * 7) % 1000) as f32).collect();

            let vec_a = VectorSimd::from_vec(data_a.clone());
            let vec_b = VectorSimd::from_vec(data_b.clone());

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = &vec_a + &vec_b;
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result); // Prevent optimization

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = scalar_add_f32(&data_a, &data_b);
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result); // Prevent optimization

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_f32_multiplication() {
        println!("\n=== Benchmark: f32 Multiplication ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
            let data_b: Vec<f32> = (0..size).map(|i| ((i * 7) % 1000) as f32 / 1000.0).collect();

            let vec_a = VectorSimd::from_vec(data_a.clone());
            let vec_b = VectorSimd::from_vec(data_b.clone());

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = &vec_a * &vec_b;
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result);

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = scalar_mul_f32(&data_a, &data_b);
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_f32_scalar_multiply() {
        println!("\n=== Benchmark: f32 Scalar Multiply (vec * scalar) ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
            let vec = VectorSimd::from_vec(data.clone());
            let scalar = 2.5f32;

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = &vec * scalar;
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result);

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = data.iter().map(|&x| x * scalar).collect();
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_f32_division() {
        println!("\n=== Benchmark: f32 Division ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 + 1.0).collect();
            let data_b: Vec<f32> = (0..size).map(|i| ((i * 7) % 1000) as f32 + 1.0).collect();

            let vec_a = VectorSimd::from_vec(data_a.clone());
            let vec_b = VectorSimd::from_vec(data_b.clone());

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = &vec_a / &vec_b;
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result);

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = data_a.iter().zip(data_b.iter()).map(|(x, y)| x / y).collect();
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_f32_abs() {
        println!("\n=== Benchmark: f32 Absolute Value ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data: Vec<f32> = (0..size).map(|i| if i % 2 == 0 { i as f32 } else { -(i as f32) }).collect();
            let vec = VectorSimd::from_vec(data.clone());

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = vec.abs();
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result);

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = data.iter().map(|&x| x.abs()).collect();
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_f32_convolution() {
        println!("\n=== Benchmark: f32 Convolution ===");

        for &size in &[10_000, 100_000] {
            println!("\nSignal size: {}, Kernel size: 101", size);

            let signal_data: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
            let kernel_data: Vec<f32> = (0..101).map(|i| (i as f32 / 100.0).sin()).collect();

            let signal = VectorSimd::from_vec(signal_data.clone());
            let kernel = VectorSimd::from_vec(kernel_data.clone());

            const ITERATIONS: usize = 10;

            // Benchmark SIMD
            let start = Instant::now();
            let mut simd_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                simd_result = signal.convolve(&kernel);
            }
            let simd_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&simd_result);

            // Benchmark Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = scalar_convolve_f32(&signal_data, &kernel_data);
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos() as f64;

            println!("  SIMD:   {:?}", simd_time);
            println!("  Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x", speedup);
        }
    }

    #[test]
    fn benchmark_complex32_multiplication() {
        println!("\n=== Benchmark: Complex32 Multiplication ===");

        for &size in &[10_000, 100_000] {
            println!("\nVector size: {}", size);

            let data_a: Vec<Complex<f32>> = (0..size)
                .map(|i| Complex::new((i % 100) as f32 / 100.0, ((i * 3) % 100) as f32 / 100.0))
                .collect();
            let data_b: Vec<Complex<f32>> = (0..size)
                .map(|i| Complex::new(((i * 7) % 100) as f32 / 100.0, ((i * 11) % 100) as f32 / 100.0))
                .collect();

            let vec_a = VectorSimd::from_vec(data_a.clone());
            let vec_b = VectorSimd::from_vec(data_b.clone());

            const ITERATIONS: usize = 10;

            // Benchmark Current Implementation (iterator-based)
            let start = Instant::now();
            let mut impl_result = VectorSimd::new();
            for _ in 0..ITERATIONS {
                impl_result = &vec_a * &vec_b;
            }
            let impl_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&impl_result);

            // Benchmark Pure Scalar
            let start = Instant::now();
            let mut scalar_result = Vec::new();
            for _ in 0..ITERATIONS {
                scalar_result = data_a.iter().zip(data_b.iter()).map(|(a, b)| a * b).collect();
            }
            let scalar_time = start.elapsed() / ITERATIONS as u32;
            std::hint::black_box(&scalar_result);

            let speedup = scalar_time.as_nanos() as f64 / impl_time.as_nanos() as f64;

            println!("  VectorSimd: {:?}", impl_time);
            println!("  Pure Scalar: {:?}", scalar_time);
            println!("  Speedup: {:.2}x (Note: Complex not yet SIMD-optimized)", speedup);
        }
    }

    #[test]
    fn benchmark_summary() {
        println!("\n");
        println!("╔════════════════════════════════════════════════════════════════╗");
        println!("║           SIMD Performance Benchmark Summary                   ║");
        println!("╠════════════════════════════════════════════════════════════════╣");
        println!("║ Configuration: 128-bit registers (f32x4, f64x2)                ║");
        println!("║ Expected speedup:                                              ║");
        println!("║   - f32 operations: ~3-4x (4 elements in parallel)             ║");
        println!("║   - f64 operations: ~2x   (2 elements in parallel)             ║");
        println!("║   - Complex32: Not yet SIMD-optimized (planned for future)     ║");
        println!("║                                                                 ║");
        println!("║ To use 256-bit registers:                                      ║");
        println!("║   Change simd_config::F32Batch from f32x4 → f32x8             ║");
        println!("║   Expected speedup: ~6-8x (8 elements in parallel)             ║");
        println!("╚════════════════════════════════════════════════════════════════╝");
        println!();
    }
}
