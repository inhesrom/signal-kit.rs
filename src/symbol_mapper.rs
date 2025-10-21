#![allow(dead_code)]
use std::collections::HashMap;
use num_complex::Complex;
use num_traits::Float;

pub struct MapDemap<C> {
    qpsk: HashMap<u8, Complex<C>>,
    // eight_psk: HashMap<u8, Complex<C>>,
    // sixteen_apsk: HashMap<u8, Complex<C>>,
}

impl<C: Float> MapDemap<C> {
    pub fn new() -> Self {
        let mut qpsk = HashMap::new();
        let one_root_two = C::from(1.0).unwrap() / C::from(2.0).unwrap().sqrt();
        qpsk.insert(0b00, Complex::new(one_root_two, one_root_two));
        qpsk.insert(0b01, Complex::new(-one_root_two, one_root_two));
        qpsk.insert(0b10, Complex::new(one_root_two, -one_root_two));
        qpsk.insert(0b11, Complex::new(-one_root_two, -one_root_two));

        MapDemap {
            qpsk,
        }
    }

    // Map bits to symbol
    pub fn modulate(&self, bits: u8) -> Complex<C> {
        self.qpsk.get(&bits).expect("Entered bits must be 11, 00, 10, or 01").clone()
    }

    // Map symbol to bits (finds nearest constellation point)
    pub fn demodulate(&self, symbol: &Complex<C>) -> Option<u8> {
        self.qpsk
            .iter()
            .min_by(|(_, s1), (_, s2)| {
                let d1 = (*s1 - *symbol).norm_sqr();
                let d2 = (*s2 - *symbol).norm_sqr();
                d1.partial_cmp(&d2).unwrap()
            })
            .map(|(bits, _)| *bits)
    }
}

#[cfg(test)]
mod tests {
    use num_complex::Complex32;

    use crate::symbol_mapper::MapDemap;

    #[test]
    fn test_qpsk_mapper() {
        let zero_zero = 0b00;
        let zero_one = 0b01;
        let one_zero = 0b10;
        let one_one = 0b11;

        let one_root_two = 1.0f32 / 2.0f32.sqrt();
        let first_quad = Complex32::new(one_root_two, one_root_two);
        let second_quad = Complex32::new(-one_root_two, one_root_two);
        let third_quad = Complex32::new(one_root_two, -one_root_two);
        let fourth_quad = Complex32::new(-one_root_two, -one_root_two);

        let mapper = MapDemap::<f32>::new();

        let iq = mapper.modulate(zero_zero);
        assert_eq!(iq, first_quad);
        let iq = mapper.modulate(zero_one);
        assert_eq!(iq, second_quad);
        let iq = mapper.modulate(one_zero);
        assert_eq!(iq, third_quad);
        let iq = mapper.modulate(one_one);
        assert_eq!(iq, fourth_quad);
    }
}
