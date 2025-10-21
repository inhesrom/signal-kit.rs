use bimap::BiHashMap;
use num_complex::Complex;
use num_traits::Float;

pub struct MapDemap<C> {
    qpsk: BiHashMap<u8, Complex<C>>,
}

impl<C: Float> MapDemap<C> {
    fn new() -> Self {
        let sqrt_half = C::from(1.0).unwrap() / C::from(0.5).unwrap().sqrt();

        let mut qpsk = BiHashMap::<u8, Complex<C>>::new();
        qpsk.insert(0b00, Complex::<C>::new(sqrt_half, sqrt_half));
        qpsk.insert(0b01, Complex::<C>::new(-sqrt_half, sqrt_half));
        qpsk.insert(0b10, Complex::<C>::new(sqrt_half, -sqrt_half));
        qpsk.insert(0b11, Complex::<C>::new(-sqrt_half, -sqrt_half));

        MapDemap { qpsk }
    }

    // Map bits to symbol
    pub fn modulate(&self, bits: u8) -> Option<&Complex<C>> {
        self.qpsk.get_by_left(&bits)
    }

    // Map symbol to bits
    pub fn demodulate(&self, symbol: &Complex<C>) -> Option<&u8> {
        self.qpsk.get_by_right(symbol)
    }
}
