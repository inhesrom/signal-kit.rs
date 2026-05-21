use std::hint::black_box;

use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use num_complex::Complex;
use signal_kit::complex_vec::ComplexVec;
use signal_kit::filter::FarrowResampler;
use signal_kit::filter::fft_interpolator::resample;

const SIGNAL_OFFSET: f64 = 0.17;

/// One resampling ratio expressed as separate input and output rates.
#[derive(Clone, Copy)]
struct RatioCase {
    input_rate: f64,
    output_rate: f64,
}

impl RatioCase {
    fn factor(&self) -> f64 {
        self.output_rate / self.input_rate
    }
}

/// Registers Farrow streaming and FFT resample benchmarks across ratios and sizes.
fn farrow_benchmarks(criterion: &mut Criterion) {
    bench_farrow_process_block(criterion);
    bench_farrow_fft_resample_reference(criterion);
}

/// Registers streaming Farrow benchmarks across ratios and sizes.
fn bench_farrow_process_block(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("farrow/process_block");
    for case in ratio_cases() {
        for size in input_sizes() {
            set_element_throughput(&mut group, size);
            bench_farrow_variant(&mut group, case, size);
        }
    }
    group.finish();
}

/// Registers FFT-based resample reference benchmarks across the same ratios and sizes.
fn bench_farrow_fft_resample_reference(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("farrow/fft_resample");
    for case in ratio_cases() {
        for size in input_sizes() {
            set_element_throughput(&mut group, size);
            bench_fft_resample_variant(&mut group, case, size);
        }
    }
    group.finish();
}

/// Registers one streaming Farrow benchmark variant.
fn bench_farrow_variant(group: &mut BenchmarkGroup<'_, WallTime>, case: RatioCase, size: usize) {
    let signal = make_signal(size);
    let template = FarrowResampler::<f64>::cubic_lagrange(case.input_rate, case.output_rate);
    group.bench_function(case_id(case, size), |bencher| {
        bencher.iter_batched(
            || template.clone(),
            |mut resampler| {
                let output = resampler.process_block(black_box(&signal));
                black_box(output);
            },
            BatchSize::SmallInput,
        );
    });
}

/// Registers one FFT-based resample reference variant.
fn bench_fft_resample_variant(group: &mut BenchmarkGroup<'_, WallTime>, case: RatioCase, size: usize) {
    let signal = ComplexVec::from_vec(make_signal(size));
    group.bench_function(case_id(case, size), |bencher| {
        bencher.iter(|| {
            let output = resample(black_box(&signal), case.factor());
            black_box(output);
        });
    });
}

/// Returns the resampling ratios benchmarked here.
fn ratio_cases() -> [RatioCase; 3] {
    [
        RatioCase { input_rate: 1.0, output_rate: 1.5 },
        RatioCase { input_rate: 1.0, output_rate: 2.0 },
        RatioCase { input_rate: 1.0, output_rate: 0.8 },
    ]
}

/// Returns the input block sizes benchmarked here.
fn input_sizes() -> [usize; 3] {
    [1024, 16_384, 65_536]
}

/// Sets group throughput in input complex IQ samples.
fn set_element_throughput(group: &mut BenchmarkGroup<'_, WallTime>, size: usize) {
    group.throughput(Throughput::Elements(size as u64));
}

/// Returns a readable benchmark case ID combining ratio and block size.
fn case_id(case: RatioCase, size: usize) -> BenchmarkId {
    let label = format!("ratio={:.2}/n={}", case.factor(), size);
    BenchmarkId::from_parameter(label)
}

/// Builds deterministic complex IQ samples for repeatable timing.
fn make_signal(len: usize) -> Vec<Complex<f64>> {
    (0..len)
        .map(|index| {
            let value = index as f64 + SIGNAL_OFFSET;
            Complex::new((value * 0.07).sin() * 0.5, (value * 0.11).cos() * -0.25)
        })
        .collect()
}

criterion_group!(benches, farrow_benchmarks);
criterion_main!(benches);
