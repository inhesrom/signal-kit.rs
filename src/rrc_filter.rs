use num_traits::Float;

pub struct RRCFilter<T> {
    num_filter_taps: usize,
    beta: T,
    sps: T, //samples per symbol
    scale: T,
}

impl<T: Float> RRCFilter<T> {
    pub fn new(num_filter_taps: usize, sample_rate: T, symbol_rate: T, beta: T) -> Self {
        let one = T::one();
        let sps = sample_rate / symbol_rate;
        let scale = one / sample_rate;

        RRCFilter {
            num_filter_taps,
            beta,
            sps,
            scale
        }
    }

    pub fn build_filter<V>(&self) -> V
    where
        V: Default + Extend<T>,
    {
        let pi = T::from(std::f64::consts::PI).unwrap();
        let zero = T::zero();
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let four = T::from(4.0).unwrap();

        let taps: Vec<T> = (0..self.num_filter_taps).map(|i| {
            let t = T::from(i).unwrap() - T::from(self.num_filter_taps).unwrap() / two;
            let t = t * self.scale;

            if t == zero {
                (one + self.beta * (four / pi - one)) / self.sps
            }
            else if t.abs() == self.sps / (four * self.beta) {
                (self.beta / two.sqrt()) *
                ((one + two / pi) * (pi / (four * self.beta)).sin() +
                 (one - two / pi) * (pi / (four * self.beta)).cos()) / self.sps
            }
            else {
                ((pi * t * (one - self.beta) / self.sps).sin() +
                 four * self.beta * t * (pi * t * (one + self.beta) / self.sps).cos() / self.sps) /
                (pi * t * (one - (four * self.beta * t / self.sps) * (four * self.beta * t / self.sps)) / self.sps)
            }
        }).collect();

        let mut h = V::default();
        h.extend(taps);
        h
    }
}

#[cfg(test)]
mod tests {
    use crate::rrc_filter::RRCFilter;
    use num_complex::{Complex32, Complex64};

    #[test]
    fn test_vec_f32() {
        let filter32 = RRCFilter::<Complex32>::new(101, 1e6_f32, 100e3_f32, 0.35_f32);
        let taps32: Vec<f32> = filter32.build_filter();
        println!("Filter taps are: {}", taps32.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "));
    }

    #[test]
    fn test_vec_f64() {
        let filter64 = RRCFilter::<f64>::new(51, 1e6_f64, 500e3_f64, 0.25_f64);
        let taps64: Vec<f64> = filter64.build_filter();
        println!("Filter taps are: {}", taps64.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(", "));
    }
}
