use num_traits::Float;

pub struct RRCFilter<T> {
    num_filter_taps: usize,
    sample_rate: T,
    symbol_rate: T,
    beta: T,
    symbol_period: T,
    scale: T,
}

impl<T: Float> RRCFilter<T> {
    pub fn new(num_filter_taps: usize, sample_rate: T, symbol_rate: T, beta: T) -> Self {
        let one = T::one();
        let symbol_period = one / symbol_rate;
        let scale = one / sample_rate;

        RRCFilter {
            num_filter_taps,
            sample_rate,
            symbol_rate,
            beta,
            symbol_period,
            scale
        }
    }

    pub fn build_filter<V>(&self) -> V
    where
        V: Default + Extend<T>,
    {
        let mut h = V::default();
        let pi = T::from(std::f64::consts::PI).unwrap();
        let zero = T::zero();
        let one = T::one();
        let two = T::from(2.0).unwrap();
        let four = T::from(4.0).unwrap();

        for i in 0..self.num_filter_taps {
            let t = T::from(i).unwrap() - T::from(self.num_filter_taps).unwrap() / two;
            let t = t * self.scale;

            if t == zero {
                h.push((one + self.beta * (four / pi - one)) / self.symbol_period);
            }
            else if t.abs() == self.symbol_period / (four * self.beta) {
                h.push(
                    (self.beta / two.sqrt()) *
                    ((one + two / pi) * (pi / (four * self.beta)).sin() +
                     (one - two / pi) * (pi / (four * self.beta)).cos()) / self.symbol_period
                );
            }
            else {
                h.push(
                    ((pi * t * (one - self.beta) / self.symbol_period).sin() +
                     four * self.beta * t * (pi * t * (one + self.beta) / self.symbol_period).cos() / self.symbol_period) /
                    (pi * t * (one - (four * self.beta * t / self.symbol_period) * (four * self.beta * t / self.symbol_period)) / self.symbol_period)
                );
            }
        }

        h
    }
}

#[cfg(test)]
mod tests {
    use crate::RRCFilter;
    #[test]
    fn test_vec_f32() {
        let filter32 = RRCFilter::<f32>::new(101, 1e6_f32, 100e3_f32, 0.35_f32);
        let taps32: Vec<f32> = filter32.build_filter();
    }
}
