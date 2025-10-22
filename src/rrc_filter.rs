pub struct RRCFilter<V> {
    num_filter_taps: usize,
    sample_rate: f64,
    symbol_rate: f64,
    beta: f64,
    T: f64,
    scale: f64
}

impl<V> RRCFilter<V> {

    pub fn new(num_filter_taps: usize, sample_rate: f64, symbol_rate: f64, beta: f64) -> Self {
        let T: f64 = sample_rate / symbol_rate;
        let scale: f64 = 1.0 / sample_rate;

        RRCFilter {
            num_filter_taps,
            sample_rate,
            symbol_rate,
            beta,
            T,
            scale
        }
    }

    pub fn build_filter(&self) -> V {
        let mut h: V;
        for i in 0..self.num_filter_taps {
            let t: f64 = (i as f64 - (num_filter_taps as f64) / 2.0) * self.scale;
            if t == 0.0 {

            } else if t.abs() == (self.T) {}
        }
    }
}
