use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;
use signal_kit::vector_simd::{self, IQ_LANES};

const EPSILON: f32 = 1.0e-4;
const POWER_ABSOLUTE_EPSILON: f32 = 2.0e-3;
const POWER_RELATIVE_EPSILON: f32 = 2.0e-5;
const LEFT_OFFSET: f32 = 0.13;
const RIGHT_OFFSET: f32 = -0.21;
const SCALE: f32 = -1.75;
const AXPY_SCALE_RE: f32 = 0.75;
const AXPY_SCALE_IM: f32 = -0.25;
const CONVOLUTION_SUMMARY_ITERATIONS: usize = 10;

/// Operation type for writing one binary IQ result slice.
type BinaryToOperation = fn(&[Complex<f32>], &[Complex<f32>], &mut [Complex<f32>]);

/// Operation type for mutating one IQ slice from another IQ slice.
type BinaryInplaceOperation = fn(&mut [Complex<f32>], &[Complex<f32>]);

/// Operation type for scaling one IQ slice in place.
type ScaleInplaceOperation = fn(&mut [Complex<f32>], f32);

/// Operation type for adding a scaled IQ slice in place.
type AxpyInplaceOperation = fn(&mut [Complex<f32>], Complex<f32>, &[Complex<f32>]);

/// Operation type for returning one scalar IQ reduction.
type PowerSumOperation = fn(&[Complex<f32>]) -> f32;

/// Operation type for writing one real-valued IQ result slice.
type MagnitudeToOperation = fn(&[Complex<f32>], &mut [f32]);

/// Operation type for writing one full IQ convolution result slice.
type ConvolveFullToOperation = fn(&[Complex<f32>], &[Complex<f32>], &mut [Complex<f32>]);

/// Manual timing result for one convolution smoke summary row.
struct ConvolutionSmokeTiming {
    input_size: usize,
    kernel_size: usize,
    selected_time: Duration,
    scalar_time: Duration,
}

/// Registers all SIMD backend benchmarks against scalar references.
fn vector_simd_benchmarks(criterion: &mut Criterion) {
    bench_binary_to_operation(criterion, "iq_add_to", vector_simd::iq_add_to, scalar_iq_add_to);
    bench_binary_inplace_operation(criterion, "iq_add_inplace", vector_simd::iq_add_inplace, scalar_iq_add_inplace);
    bench_scale_inplace_operation(criterion);
    bench_axpy_inplace_operation(criterion);
    bench_binary_to_operation(criterion, "iq_mul_to", vector_simd::iq_mul_to, scalar_iq_mul_to);
    bench_power_sum_operation(criterion);
    bench_magnitude_to_operation(criterion);
    bench_convolve_full_to_operation(criterion);
    print_convolution_summary_if_requested();
}

/// Registers one binary output operation for all benchmark block sizes.
fn bench_binary_to_operation(
    criterion: &mut Criterion,
    operation_name: &str,
    selected_operation: BinaryToOperation,
    scalar_operation: BinaryToOperation,
) {
    let mut group = criterion.benchmark_group(format!("vector_simd/{operation_name}"));
    for size in benchmark_sizes() {
        assert_binary_to_matches(size, selected_operation, scalar_operation);
        set_element_throughput(&mut group, size);
        bench_binary_to_variant(&mut group, "selected_backend", size, selected_operation);
        bench_binary_to_variant(&mut group, "scalar_reference", size, scalar_operation);
    }
    group.finish();
}

/// Registers one binary in-place operation for all benchmark block sizes.
fn bench_binary_inplace_operation(
    criterion: &mut Criterion,
    operation_name: &str,
    selected_operation: BinaryInplaceOperation,
    scalar_operation: BinaryInplaceOperation,
) {
    let mut group = criterion.benchmark_group(format!("vector_simd/{operation_name}"));
    for size in benchmark_sizes() {
        assert_binary_inplace_matches(size, selected_operation, scalar_operation);
        set_element_throughput(&mut group, size);
        bench_binary_inplace_variant(&mut group, "selected_backend", size, selected_operation);
        bench_binary_inplace_variant(&mut group, "scalar_reference", size, scalar_operation);
    }
    group.finish();
}

/// Registers in-place scaling benchmarks for all benchmark block sizes.
fn bench_scale_inplace_operation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_simd/iq_scale_inplace");
    for size in benchmark_sizes() {
        assert_scale_inplace_matches(size);
        set_element_throughput(&mut group, size);
        bench_scale_inplace_variant(&mut group, "selected_backend", size, vector_simd::iq_scale_inplace);
        bench_scale_inplace_variant(&mut group, "scalar_reference", size, scalar_iq_scale_inplace);
    }
    group.finish();
}

/// Registers in-place AXPY benchmarks for all benchmark block sizes.
fn bench_axpy_inplace_operation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_simd/iq_axpy_inplace");
    for size in benchmark_sizes() {
        assert_axpy_inplace_matches(size);
        set_element_throughput(&mut group, size);
        bench_axpy_inplace_variant(&mut group, "selected_backend", size, vector_simd::iq_axpy_inplace);
        bench_axpy_inplace_variant(&mut group, "scalar_reference", size, scalar_iq_axpy_inplace);
    }
    group.finish();
}

/// Registers power-sum benchmarks for all benchmark block sizes.
fn bench_power_sum_operation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_simd/iq_power_sum");
    for size in benchmark_sizes() {
        assert_power_sum_matches(size);
        set_element_throughput(&mut group, size);
        bench_power_sum_variant(&mut group, "selected_backend", size, vector_simd::iq_power_sum);
        bench_power_sum_variant(&mut group, "scalar_reference", size, scalar_iq_power_sum);
    }
    group.finish();
}

/// Registers magnitude-output benchmarks for all benchmark block sizes.
fn bench_magnitude_to_operation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_simd/magnitude_to");
    for size in benchmark_sizes() {
        assert_magnitude_to_matches(size);
        set_element_throughput(&mut group, size);
        bench_magnitude_to_variant(&mut group, "selected_backend", size, vector_simd::magnitude_to);
        bench_magnitude_to_variant(&mut group, "scalar_reference", size, scalar_magnitude_to);
    }
    group.finish();
}

/// Registers full direct convolution benchmarks for DSP block and kernel sizes.
fn bench_convolve_full_to_operation(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("vector_simd/iq_convolve_full_to");
    for input_size in convolution_input_sizes() {
        for kernel_size in convolution_kernel_sizes() {
            assert_convolve_full_to_matches(input_size, kernel_size);
            set_element_throughput(&mut group, input_size);
            bench_convolve_full_to_variant(&mut group, "selected_backend", input_size, kernel_size, vector_simd::iq_convolve_full_to);
            bench_convolve_full_to_variant(&mut group, "scalar_reference", input_size, kernel_size, scalar_iq_convolve_full_to);
        }
    }
    group.finish();
}

/// Registers one binary output benchmark variant.
fn bench_binary_to_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: BinaryToOperation) {
    let left = make_iq(size, LEFT_OFFSET);
    let right = make_iq(size, RIGHT_OFFSET);
    let mut output = zero_iq(size);
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter(|| {
            operation(black_box(left.as_slice()), black_box(right.as_slice()), black_box(output.as_mut_slice()));
        });
    });
}

/// Registers one binary in-place benchmark variant.
fn bench_binary_inplace_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: BinaryInplaceOperation) {
    let target = make_iq(size, LEFT_OFFSET);
    let input = make_iq(size, RIGHT_OFFSET);
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter_batched(
            || target.clone(),
            |mut target| {
                operation(black_box(target.as_mut_slice()), black_box(input.as_slice()));
                black_box(target);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one in-place scaling benchmark variant.
fn bench_scale_inplace_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: ScaleInplaceOperation) {
    let target = make_iq(size, LEFT_OFFSET);
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter_batched(
            || target.clone(),
            |mut target| {
                operation(black_box(target.as_mut_slice()), black_box(SCALE));
                black_box(target);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one in-place AXPY benchmark variant.
fn bench_axpy_inplace_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: AxpyInplaceOperation) {
    let target = make_iq(size, LEFT_OFFSET);
    let input = make_iq(size, RIGHT_OFFSET);
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter_batched(
            || target.clone(),
            |mut target| {
                operation(black_box(target.as_mut_slice()), black_box(axpy_scale()), black_box(input.as_slice()));
                black_box(target);
            },
            BatchSize::LargeInput,
        );
    });
}

/// Registers one power-sum benchmark variant.
fn bench_power_sum_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: PowerSumOperation) {
    let input = make_iq(size, LEFT_OFFSET);
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter(|| {
            black_box(operation(black_box(input.as_slice())));
        });
    });
}

/// Registers one magnitude-output benchmark variant.
fn bench_magnitude_to_variant(group: &mut BenchmarkGroup<'_, WallTime>, variant_name: &str, size: usize, operation: MagnitudeToOperation) {
    let input = make_iq(size, LEFT_OFFSET);
    let mut output = vec![0.0; size];
    group.bench_function(BenchmarkId::new(variant_name, size), |bencher| {
        bencher.iter(|| {
            operation(black_box(input.as_slice()), black_box(output.as_mut_slice()));
        });
    });
}

/// Registers one full direct convolution benchmark variant.
fn bench_convolve_full_to_variant(
    group: &mut BenchmarkGroup<'_, WallTime>,
    variant_name: &str,
    input_size: usize,
    kernel_size: usize,
    operation: ConvolveFullToOperation,
) {
    let input = make_iq(input_size, LEFT_OFFSET);
    let kernel = make_iq(kernel_size, RIGHT_OFFSET);
    let mut output = zero_iq(full_convolution_len(input_size, kernel_size));
    group.bench_function(
        BenchmarkId::new(variant_name, convolution_case_name(input_size, kernel_size)),
        |bencher| {
            bencher.iter(|| {
                operation(
                    black_box(input.as_slice()),
                    black_box(kernel.as_slice()),
                    black_box(output.as_mut_slice()),
                );
            });
        },
    );
}

/// Prints the manual convolution timing summary for Criterion smoke tests.
fn print_convolution_summary_if_requested() {
    let args = benchmark_cli_args();
    if should_print_convolution_summary(&args) {
        print_convolution_summary(CONVOLUTION_SUMMARY_ITERATIONS);
    }
}

/// Returns benchmark CLI arguments without the executable path.
fn benchmark_cli_args() -> Vec<String> {
    env::args().skip(1).collect()
}

/// Returns true when the current Criterion command asks for this summary.
fn should_print_convolution_summary(args: &[String]) -> bool {
    is_criterion_test_mode(args) && convolution_filter_allows_summary(args)
}

/// Returns true when Criterion is running each benchmark as a smoke test.
fn is_criterion_test_mode(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--test") || !args.iter().any(|arg| arg == "--bench")
}

/// Returns true when the positional filter is absent or targets convolution.
fn convolution_filter_allows_summary(args: &[String]) -> bool {
    match criterion_filter_arg(args) {
        Some(filter) => convolution_filter_targets_summary(filter),
        None => true,
    }
}

/// Returns the positional Criterion filter, if one was provided.
fn criterion_filter_arg(args: &[String]) -> Option<&str> {
    let mut skip_value = false;
    for arg in args {
        if should_skip_filter_arg(arg, &mut skip_value) {
            continue;
        }
        return Some(arg);
    }
    None
}

/// Returns true when an argument should not be treated as a filter.
fn should_skip_filter_arg(arg: &str, skip_value: &mut bool) -> bool {
    if *skip_value {
        *skip_value = false;
        return true;
    }
    if criterion_option_takes_value(arg) {
        *skip_value = true;
        return true;
    }
    arg == "--test" || arg.starts_with('-')
}

/// Returns true when a Criterion option consumes the following argument.
fn criterion_option_takes_value(arg: &str) -> bool {
    matches!(
        arg,
        "-b" | "-c"
            | "-s"
            | "--baseline"
            | "--baseline-lenient"
            | "--color"
            | "--confidence-level"
            | "--format"
            | "--load-baseline"
            | "--measurement-time"
            | "--noise-threshold"
            | "--nresamples"
            | "--output-format"
            | "--plotting-backend"
            | "--profile-time"
            | "--sample-size"
            | "--save-baseline"
            | "--significance-level"
            | "--warm-up-time"
    )
}

/// Returns true when a filter names the convolution benchmarks.
fn convolution_filter_targets_summary(filter: &str) -> bool {
    let filter = filter.to_ascii_lowercase();
    filter.contains("convolve") || filter.contains("convolution")
}

/// Prints the manual convolution timing summary table.
fn print_convolution_summary(iterations: usize) {
    println!();
    println!("SIMD convolution summary (manual smoke timing, {iterations} iterations per case)");
    println!("{:<12}{:<13}{:<13}speedup", "case", "selected", "scalar");
    print_convolution_summary_rows(iterations);
}

/// Prints one manual timing row for each convolution benchmark case.
fn print_convolution_summary_rows(iterations: usize) {
    for input_size in convolution_input_sizes() {
        for kernel_size in convolution_kernel_sizes() {
            let timing = measure_convolution_case(input_size, kernel_size, iterations);
            print_convolution_summary_row(&timing);
        }
    }
}

/// Prints one manual convolution timing table row.
fn print_convolution_summary_row(timing: &ConvolutionSmokeTiming) {
    println!(
        "{:<12}{:<13}{:<13}{:.2}x",
        convolution_case_name(timing.input_size, timing.kernel_size),
        format_duration(timing.selected_time),
        format_duration(timing.scalar_time),
        calculate_speedup(timing.selected_time, timing.scalar_time)
    );
}

/// Measures selected and scalar convolution timings for one case.
fn measure_convolution_case(input_size: usize, kernel_size: usize, iterations: usize) -> ConvolutionSmokeTiming {
    let input = make_iq(input_size, LEFT_OFFSET);
    let kernel = make_iq(kernel_size, RIGHT_OFFSET);
    let output_len = full_convolution_len(input_size, kernel_size);
    let selected_time = measure_convolution_operation(&input, &kernel, output_len, iterations, vector_simd::iq_convolve_full_to);
    let scalar_time = measure_convolution_operation(&input, &kernel, output_len, iterations, scalar_iq_convolve_full_to);
    ConvolutionSmokeTiming {
        input_size,
        kernel_size,
        selected_time,
        scalar_time,
    }
}

/// Measures average elapsed time for one convolution operation.
fn measure_convolution_operation(
    input: &[Complex<f32>],
    kernel: &[Complex<f32>],
    output_len: usize,
    iterations: usize,
    operation: ConvolveFullToOperation,
) -> Duration {
    let mut output = zero_iq(output_len);
    let total_time = measure_convolution_iterations(input, kernel, &mut output, iterations, operation);
    average_duration(total_time, iterations)
}

/// Measures elapsed time for repeated convolution operations.
fn measure_convolution_iterations(
    input: &[Complex<f32>],
    kernel: &[Complex<f32>],
    output: &mut [Complex<f32>],
    iterations: usize,
    operation: ConvolveFullToOperation,
) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        operation(black_box(input), black_box(kernel), black_box(&mut *output));
    }
    let elapsed = start.elapsed();
    black_box(&mut *output);
    elapsed
}

/// Returns average duration per measured operation.
fn average_duration(total_time: Duration, iterations: usize) -> Duration {
    Duration::from_secs_f64(total_time.as_secs_f64() / iterations as f64)
}

/// Returns the scalar-over-selected speedup ratio.
fn calculate_speedup(selected_time: Duration, scalar_time: Duration) -> f64 {
    scalar_time.as_secs_f64() / selected_time.as_secs_f64()
}

/// Formats a duration using compact ASCII time units.
fn format_duration(duration: Duration) -> String {
    let nanoseconds = duration.as_secs_f64() * 1.0e9;
    if nanoseconds < 1.0e3 {
        format!("{nanoseconds:.2} ns")
    } else if nanoseconds < 1.0e6 {
        format!("{:.2} us", nanoseconds / 1.0e3)
    } else if nanoseconds < 1.0e9 {
        format!("{:.2} ms", nanoseconds / 1.0e6)
    } else {
        format!("{:.2} s", nanoseconds / 1.0e9)
    }
}

/// Sets group throughput in complex IQ samples.
fn set_element_throughput(group: &mut BenchmarkGroup<'_, WallTime>, size: usize) {
    group.throughput(Throughput::Elements(size as u64));
}

/// Asserts that a binary output operation matches its scalar reference.
fn assert_binary_to_matches(size: usize, selected_operation: BinaryToOperation, scalar_operation: BinaryToOperation) {
    let left = make_iq(size, LEFT_OFFSET);
    let right = make_iq(size, RIGHT_OFFSET);
    let mut actual = zero_iq(size);
    let mut expected = zero_iq(size);
    selected_operation(&left, &right, &mut actual);
    scalar_operation(&left, &right, &mut expected);
    assert_complex_slices_close(&actual, &expected);
}

/// Asserts that a binary in-place operation matches its scalar reference.
fn assert_binary_inplace_matches(size: usize, selected_operation: BinaryInplaceOperation, scalar_operation: BinaryInplaceOperation) {
    let mut actual = make_iq(size, LEFT_OFFSET);
    let mut expected = actual.clone();
    let input = make_iq(size, RIGHT_OFFSET);
    selected_operation(&mut actual, &input);
    scalar_operation(&mut expected, &input);
    assert_complex_slices_close(&actual, &expected);
}

/// Asserts that in-place scaling matches its scalar reference.
fn assert_scale_inplace_matches(size: usize) {
    let mut actual = make_iq(size, LEFT_OFFSET);
    let mut expected = actual.clone();
    vector_simd::iq_scale_inplace(&mut actual, SCALE);
    scalar_iq_scale_inplace(&mut expected, SCALE);
    assert_complex_slices_close(&actual, &expected);
}

/// Asserts that in-place AXPY matches its scalar reference.
fn assert_axpy_inplace_matches(size: usize) {
    let mut actual = make_iq(size, LEFT_OFFSET);
    let mut expected = actual.clone();
    let input = make_iq(size, RIGHT_OFFSET);
    vector_simd::iq_axpy_inplace(&mut actual, axpy_scale(), &input);
    scalar_iq_axpy_inplace(&mut expected, axpy_scale(), &input);
    assert_complex_slices_close(&actual, &expected);
}

/// Asserts that power summing matches its scalar reference.
fn assert_power_sum_matches(size: usize) {
    let input = make_iq(size, LEFT_OFFSET);
    let actual = vector_simd::iq_power_sum(&input);
    let expected = scalar_iq_power_sum(&input);
    assert_close(actual, expected, power_sum_epsilon(expected));
}

/// Asserts that magnitude output matches its scalar reference.
fn assert_magnitude_to_matches(size: usize) {
    let input = make_iq(size, LEFT_OFFSET);
    let mut actual = vec![0.0; size];
    let mut expected = vec![0.0; size];
    vector_simd::magnitude_to(&input, &mut actual);
    scalar_magnitude_to(&input, &mut expected);
    assert_slices_close(&actual, &expected);
}

/// Asserts that full convolution matches its scalar reference.
fn assert_convolve_full_to_matches(input_size: usize, kernel_size: usize) {
    let input = make_iq(input_size, LEFT_OFFSET);
    let kernel = make_iq(kernel_size, RIGHT_OFFSET);
    let mut actual = zero_iq(full_convolution_len(input_size, kernel_size));
    let mut expected = zero_iq(full_convolution_len(input_size, kernel_size));
    vector_simd::iq_convolve_full_to(&input, &kernel, &mut actual);
    scalar_iq_convolve_full_to(&input, &kernel, &mut expected);
    assert_complex_slices_close(&actual, &expected);
}

/// Adds scalar complex samples into `output`.
fn scalar_iq_add_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
    for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
        *output_sample = *left_sample + *right_sample;
    }
}

/// Adds scalar complex samples into `target`.
fn scalar_iq_add_inplace(target: &mut [Complex<f32>], input: &[Complex<f32>]) {
    for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
        *target_sample += *input_sample;
    }
}

/// Scales scalar complex samples in place.
fn scalar_iq_scale_inplace(target: &mut [Complex<f32>], scale: f32) {
    for target_sample in target {
        *target_sample *= scale;
    }
}

/// Adds scalar `scale * input` products into `target`.
fn scalar_iq_axpy_inplace(target: &mut [Complex<f32>], scale: Complex<f32>, input: &[Complex<f32>]) {
    for (target_sample, input_sample) in target.iter_mut().zip(input.iter()) {
        *target_sample += scale * *input_sample;
    }
}

/// Multiplies scalar complex samples into `output`.
fn scalar_iq_mul_to(left: &[Complex<f32>], right: &[Complex<f32>], output: &mut [Complex<f32>]) {
    for ((left_sample, right_sample), output_sample) in left.iter().zip(right.iter()).zip(output.iter_mut()) {
        *output_sample = *left_sample * *right_sample;
    }
}

/// Returns a scalar sum of squared magnitudes.
fn scalar_iq_power_sum(input: &[Complex<f32>]) -> f32 {
    input.iter().map(|sample| sample.norm_sqr()).sum()
}

/// Writes scalar magnitudes into `output`.
fn scalar_magnitude_to(input: &[Complex<f32>], output: &mut [f32]) {
    for (sample, output_sample) in input.iter().zip(output.iter_mut()) {
        *output_sample = sample.norm();
    }
}

/// Writes scalar full direct convolution samples into `output`.
fn scalar_iq_convolve_full_to(input: &[Complex<f32>], kernel: &[Complex<f32>], output: &mut [Complex<f32>]) {
    assert_eq!(output.len(), full_convolution_len(input.len(), kernel.len()));
    for (full_index, output_sample) in output.iter_mut().enumerate() {
        *output_sample = scalar_iq_convolve_sample(input, kernel, full_index);
    }
}

/// Returns one scalar direct convolution output sample.
fn scalar_iq_convolve_sample(input: &[Complex<f32>], kernel: &[Complex<f32>], full_index: usize) -> Complex<f32> {
    let kernel_start = (kernel.len() - 1).saturating_sub(full_index);
    let kernel_end = kernel.len().min(full_convolution_len(input.len(), kernel.len()) - full_index);
    scalar_iq_convolve_overlap(input, kernel, full_index, kernel_start, kernel_end)
}

/// Returns the scalar sum for one direct convolution overlap.
fn scalar_iq_convolve_overlap(
    input: &[Complex<f32>],
    kernel: &[Complex<f32>],
    full_index: usize,
    kernel_start: usize,
    kernel_end: usize,
) -> Complex<f32> {
    let mut sum = Complex::new(0.0, 0.0);
    for kernel_index in kernel_start..kernel_end {
        let input_index = full_index + kernel_index + 1 - kernel.len();
        sum += input[input_index] * kernel[kernel_index];
    }
    sum
}

/// Returns benchmark block sizes around lanes and larger DSP blocks.
fn benchmark_sizes() -> Vec<usize> {
    let mut sizes = Vec::new();
    for size in raw_benchmark_sizes() {
        push_unique_nonzero(&mut sizes, size);
    }
    sizes
}

/// Returns raw benchmark sizes before deduplication.
fn raw_benchmark_sizes() -> [usize; 6] {
    [IQ_LANES.saturating_sub(1), IQ_LANES, IQ_LANES + 1, 1024, 16_384, 65_536]
}

/// Returns convolution input sizes for short-FIR benchmark coverage.
fn convolution_input_sizes() -> [usize; 3] {
    [1024, 16_384, 65_536]
}

/// Returns convolution kernel sizes including the common 51-tap RRC case.
fn convolution_kernel_sizes() -> [usize; 3] {
    [5, 51, 257]
}

/// Returns the full convolution output length.
fn full_convolution_len(input_len: usize, kernel_len: usize) -> usize {
    input_len + kernel_len - 1
}

/// Returns a readable convolution benchmark case name.
fn convolution_case_name(input_size: usize, kernel_size: usize) -> String {
    format!("{input_size}x{kernel_size}")
}

/// Pushes a nonzero size when it is not already present.
fn push_unique_nonzero(sizes: &mut Vec<usize>, size: usize) {
    if size != 0 && !sizes.contains(&size) {
        sizes.push(size);
    }
}

/// Builds deterministic IQ samples for repeatable checks and timings.
fn make_iq(len: usize, offset: f32) -> Vec<Complex<f32>> {
    (0..len)
        .map(|index| {
            let value = index as f32 + offset;
            Complex::new(value.sin() * 0.5, value.cos() * -0.25)
        })
        .collect()
}

/// Builds a zeroed IQ buffer.
fn zero_iq(len: usize) -> Vec<Complex<f32>> {
    vec![Complex::new(0.0, 0.0); len]
}

/// Returns the complex AXPY scale used by checks and benchmarks.
fn axpy_scale() -> Complex<f32> {
    Complex::new(AXPY_SCALE_RE, AXPY_SCALE_IM)
}

/// Returns tolerance for f32 reductions with different accumulation orders.
fn power_sum_epsilon(expected: f32) -> f32 {
    POWER_ABSOLUTE_EPSILON.max(expected.abs() * POWER_RELATIVE_EPSILON)
}

/// Asserts that two complex slices are close.
fn assert_complex_slices_close(actual: &[Complex<f32>], expected: &[Complex<f32>]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_sample, expected_sample) in actual.iter().zip(expected.iter()) {
        assert_close(actual_sample.re, expected_sample.re, EPSILON);
        assert_close(actual_sample.im, expected_sample.im, EPSILON);
    }
}

/// Asserts that two real-valued slices are close.
fn assert_slices_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected.iter()) {
        assert_close(*actual_value, *expected_value, EPSILON);
    }
}

/// Asserts that two real values are close.
fn assert_close(actual: f32, expected: f32, epsilon: f32) {
    assert!(
        (actual - expected).abs() <= epsilon,
        "actual {actual} expected {expected} epsilon {epsilon}"
    );
}

criterion_group!(benches, vector_simd_benchmarks);
criterion_main!(benches);
