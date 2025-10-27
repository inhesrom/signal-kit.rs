#[warn(dead_code)]

mod fft {
    use std::fmt::Debug;
    use num_complex::{Complex, ComplexFloat};
    use num_traits::{Float, Signed, FromPrimitive};
    use rustfft::FftPlanner;

    pub fn fft<T>(input : &mut [Complex<T>])
    where
        T: Float + Send + Sync + FromPrimitive + Signed + Debug + 'static,
    {
        let mut planner = FftPlanner::<T>::new();
        let fft_forward = planner.plan_fft_forward(input.len());
        fft_forward.process(input);
    }

}

#[cfg(test)]
mod tests {
    use crate::cw::CW;
    use crate::fft::fft;
    use num_complex::{Complex, ComplexFloat};

    #[test]
    fn test_fft() {
        let mut cw_generator = CW::new(100e3, 500e3, 500_000);
        let mut cw: Vec<Complex<f32>> = (0..1000).flat_map(|_| cw_generator.generate_block::<f32>()).collect();
        let cw_fft = fft::fft::<f32>(&mut cw);
        let mut c = Complex::<f32>::new(3.0, 2.0);
        let r = c.norm();
    }
}
