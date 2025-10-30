#![allow(dead_code)]

/// Enum representing different modulation types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModType {
    _QPSK,
    _8PSK,
    _16APSK,
    _16QAM,
    _32QAM,
    _64QAM,
}

pub mod modulation {

use std::collections::HashMap;
use num_complex::Complex;
use num_traits::Float;

    struct ModProperties<T> {
        bits_per_symbol: usize,
        num_symbols: usize,
        bit_symbol_map: HashMap<u8, Complex<T>>,
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
}
