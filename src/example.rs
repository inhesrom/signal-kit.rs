use num_complex::Complex;
use std::time::Instant;

mod vector_simd;
use vector_simd::VectorSimd;

fn main() {
    println!("=== VectorSimd Examples ===\n");

    // Example 1: Basic f32 operations
    println!("1. Basic f32 operations:");
    let a = VectorSimd::from_vec(vec![1.0f32, 2.0, 3.0, 4.0]);
    let b = VectorSimd::from_vec(vec![5.0f32, 6.0, 7.0, 8.0]);

    let sum = &a + &b;
    println!("   a + b = {:?}", sum.as_slice());

    let product = &a * &b;
    println!("   a * b = {:?}", product.as_slice());

    let scalar_add = &a + 10.0;
    println!("   a + 10 = {:?}", scalar_add.as_slice());

    // Example 2: Complex number operations
    println!("\n2. Complex number operations:");
    let c1 = VectorSimd::from_vec(vec![
        Complex::new(2.0f32, 3.0),
        Complex::new(4.0, 5.0),
    ]);
    let c2 = VectorSimd::from_vec(vec![
        Complex::new(4.0f32, 5.0),
        Complex::new(6.0, 7.0),
    ]);

    let c_product = &c1 * &c2;
    println!("   Complex multiplication:");
    for i in 0..c_product.len() {
        println!("   [{i}] = {}", c_product[i]);
    }

    // Example 3: Convolution
    println!("\n3. Convolution:");
    let signal: Vec<f32> = (0..1024).map(|x| x as f32).collect();
    let signal_vec = VectorSimd::from_vec(signal);
    let kernel = VectorSimd::from_vec(vec![1.0f32, 0.0, 1.0]);

    let result = signal_vec.convolve(&kernel);
    println!("   Signal length: {}", signal_vec.len());
    println!("   Kernel length: {}", kernel.len());
    println!("   Result length: {}", result.len());
    println!("   First few results: {:?}", &result.as_slice()[..5]);

    // Example 4: Performance comparison
    println!("\n4. Performance comparison (10M elements):");

    let size = 10_000_000;

    // Generate simple test data (deterministic)
    let data_a: Vec<f32> = (0..size).map(|i| (i % 1000) as f32 / 1000.0).collect();
    let data_b: Vec<f32> = (0..size).map(|i| ((i * 7) % 1000) as f32 / 1000.0).collect();

    // Standard multiplication
    let start = Instant::now();
    let mut result_std = vec![0.0f32; size];
    for i in 0..size {
        result_std[i] = data_a[i] * data_b[i];
    }
    let duration_std = start.elapsed();
    println!("   Standard multiplication: {:?}", duration_std);

    // SIMD multiplication using operator overload
    let vec_a = VectorSimd::from_vec(data_a.clone());
    let vec_b = VectorSimd::from_vec(data_b.clone());

    let start = Instant::now();
    let result_simd = &vec_a * &vec_b;
    let duration_operator = start.elapsed();
    println!("   Operator overload (*):   {:?}", duration_operator);

    // SIMD multiplication using explicit SIMD function
    let start = Instant::now();
    let result_simd_explicit = vec_a.mul_simd(&vec_b);
    let duration_simd = start.elapsed();
    println!("   Explicit SIMD (mul_simd): {:?}", duration_simd);

    // Verify results match
    let max_diff = result_std.iter()
        .zip(result_simd.as_slice().iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("   Max difference: {:.2e}", max_diff);

    // Example 5: Append operation
    println!("\n5. Append operation:");
    let mut v1 = VectorSimd::from_vec(vec![0.0f32, 1.0, 2.0, 3.0]);
    let v2 = VectorSimd::from_vec(vec![4.0f32, 5.0, 6.0, 7.0]);
    v1.append(&v2);
    println!("   After append: {:?}", v1.as_slice());

    // Example 6: In-place operations
    println!("\n6. In-place operations:");
    let mut v = VectorSimd::from_vec(vec![5.0f32, 15.0, 25.0]);
    println!("   Original: {:?}", v.as_slice());
    v /= 5.0;
    println!("   After /= 5.0: {:?}", v.as_slice());
}
