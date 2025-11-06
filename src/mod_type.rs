#![allow(dead_code)]

/// Enum representing different modulation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModType {
    _BPSK,
    _QPSK,
    _8PSK,
    _16APSK,
    _16QAM,
    _32QAM,
    _64QAM,
    _CW,
}

pub mod modulation {

use std::collections::HashMap;
use num_complex::Complex;
use num_traits::Float;
use crate::symbol_maps::*;
use super::ModType;

    /// Trait for modulation/demodulation operations
    pub trait Modulate<T: Float> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>>;
        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8>;
    }

    /// Enum wrapper to hold any modulation type
    pub enum Modulation<T: Float> {
        BPSK(_BPSK<T>),
        QPSK(_QPSK<T>),
        PSK8(_8PSK<T>),
        APSK16(_16APSK<T>),
        QAM16(_16QAM<T>),
        QAM32(_32QAM<T>),
        QAM64(_64QAM<T>),
    }

    pub fn get_mod_type_from_enum<T: Float>(mod_enum: ModType) -> Modulation<T> {
        match mod_enum {
            ModType::_BPSK => Modulation::BPSK(_BPSK::new(bpsk_map())),
            ModType::_QPSK => Modulation::QPSK(_QPSK::new(qpsk_gray_map())),
            ModType::_8PSK => Modulation::PSK8(_8PSK::new(psk8_gray_map())),
            ModType::_16APSK => Modulation::APSK16(_16APSK::new(apsk16_gray_map())),
            ModType::_16QAM => Modulation::QAM16(_16QAM::new(qam16_gray_map())),
            ModType::_32QAM => Modulation::QAM32(_32QAM::new(qam32_gray_map())),
            ModType::_64QAM => Modulation::QAM64(_64QAM::new(qam64_gray_map())),
            ModType::_CW => unreachable!("CW is handled separately by Carrier, not through modulation system"),
        }
    }

    pub struct ModProperties<T> {
        pub bits_per_symbol: usize,
        pub num_symbols: usize,
        pub bit_symbol_map: HashMap<u8, Complex<T>>,
    }

    pub struct _BPSK<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _BPSK<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _BPSK {
                mod_properties: ModProperties {
                    bits_per_symbol: 1,
                    num_symbols: 2,
                    bit_symbol_map
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _BPSK<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _QPSK<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _QPSK<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _QPSK {
                mod_properties: ModProperties {
                    bits_per_symbol: 2,
                    num_symbols: 4,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _QPSK<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _8PSK<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _8PSK<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _8PSK {
                mod_properties: ModProperties {
                    bits_per_symbol: 3,
                    num_symbols: 8,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _8PSK<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _16APSK<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _16APSK<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _16APSK {
                mod_properties: ModProperties {
                    bits_per_symbol: 4,
                    num_symbols: 16,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _16APSK<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _16QAM<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _16QAM<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _16QAM {
                mod_properties: ModProperties {
                    bits_per_symbol: 4,
                    num_symbols: 16,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _16QAM<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _32QAM<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _32QAM<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _32QAM {
                mod_properties: ModProperties {
                    bits_per_symbol: 5,
                    num_symbols: 32,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _32QAM<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
                .iter()
                .min_by(|(_, s1), (_, s2)| {
                    let d1 = (*s1 - *symbol).norm_sqr();
                    let d2 = (*s2 - *symbol).norm_sqr();
                    d1.partial_cmp(&d2).unwrap()
                })
                .map(|(bits, _)| *bits)
        }
    }

    pub struct _64QAM<T> {
        mod_properties: ModProperties<T>
    }

    impl<T: Float> _64QAM<T> {
        pub fn new(bit_symbol_map: HashMap<u8, Complex<T>>) -> Self {
            _64QAM {
                mod_properties: ModProperties {
                    bits_per_symbol: 6,
                    num_symbols: 64,
                    bit_symbol_map,
                }
            }
        }
    }

    impl<T: Float> Modulate<T> for _64QAM<T> {
        fn modulate(&self, bits: u8) -> Option<Complex<T>> {
            self.mod_properties.bit_symbol_map.get(&bits).copied()
        }

        fn demodulate(&self, symbol: &Complex<T>) -> Option<u8> {
            self.mod_properties.bit_symbol_map
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
        use super::*;

        #[test]
        fn test_bpsk_modulate() {
            let bpsk = _BPSK::<f64>::new(crate::symbol_maps::bpsk_map());

            let symbol = bpsk.modulate(0b0).unwrap();
            assert!(symbol.re > 0.0);
            assert_eq!(symbol.im, 0.0);

            let symbol = bpsk.modulate(0b1).unwrap();
            assert!(symbol.re < 0.0);
            assert_eq!(symbol.im, 0.0);
        }

        #[test]
        fn test_bpsk_demodulate() {
            let bpsk = _BPSK::<f64>::new(crate::symbol_maps::bpsk_map());

            let symbol = bpsk.modulate(0b0).unwrap();
            let bits = bpsk.demodulate(&symbol).unwrap();
            assert_eq!(bits, 0b0);

            let symbol = bpsk.modulate(0b1).unwrap();
            let bits = bpsk.demodulate(&symbol).unwrap();
            assert_eq!(bits, 0b1);
        }

        #[test]
        fn test_qpsk_modulate() {
            let qpsk = _QPSK::<f64>::new(crate::symbol_maps::qpsk_gray_map());

            let symbol = qpsk.modulate(0b00).unwrap();
            assert!(symbol.re > 0.0 && symbol.im > 0.0);

            let symbol = qpsk.modulate(0b11).unwrap();
            assert!(symbol.re < 0.0 && symbol.im < 0.0);
        }

        #[test]
        fn test_qpsk_demodulate() {
            let qpsk = _QPSK::<f64>::new(crate::symbol_maps::qpsk_gray_map());

            let symbol = qpsk.modulate(0b01).unwrap();
            let bits = qpsk.demodulate(&symbol).unwrap();
            assert_eq!(bits, 0b01);
        }

        #[test]
        fn test_psk8_modulate() {
            let psk8 = _8PSK::<f64>::new(crate::symbol_maps::psk8_gray_map());

            let symbol = psk8.modulate(0b000).unwrap();
            assert!(symbol.norm() > 0.99 && symbol.norm() < 1.01);
        }

        #[test]
        fn test_qam16_modulate_demodulate() {
            let qam16 = _16QAM::<f64>::new(crate::symbol_maps::qam16_gray_map());

            for bits in 0..16u8 {
                let symbol = qam16.modulate(bits).unwrap();
                let decoded = qam16.demodulate(&symbol).unwrap();
                assert_eq!(bits, decoded);
            }
        }

        #[test]
        fn test_get_mod_type_from_enum() {
            let modulation = get_mod_type_from_enum::<f64>(ModType::_QPSK);

            match modulation {
                Modulation::QPSK(qpsk) => {
                    let symbol = qpsk.modulate(0b00).unwrap();
                    assert!(symbol.norm() > 0.0);
                },
                _ => panic!("Expected QPSK variant"),
            }
        }
    }
}
