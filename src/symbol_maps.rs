#![allow(dead_code)]

use std::collections::HashMap;
use num_complex::Complex;
use num_traits::Float;
use std::f64::consts::PI;

pub fn bpsk_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();
    let zero = T::from(0).unwrap();
    let one = T::from(1).unwrap();

    map.insert(0b0, Complex::new(one, zero));
    map.insert(0b1, Complex::new(-one, zero));

    map
}

/// QPSK Gray-coded constellation map (2 bits per symbol)
/// Constellation points are normalized to unit power
pub fn qpsk_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();
    let scale = T::from(1.0).unwrap() / T::from(2.0).unwrap().sqrt();

    // Gray coding: 00, 01, 11, 10
    map.insert(0b00, Complex::new(scale, scale));      // Q1
    map.insert(0b01, Complex::new(-scale, scale));     // Q2
    map.insert(0b11, Complex::new(-scale, -scale));    // Q3
    map.insert(0b10, Complex::new(scale, -scale));     // Q4

    map
}

/// 8-PSK Gray-coded constellation map (3 bits per symbol)
/// Constellation points are on the unit circle
pub fn psk8_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();

    // Gray coding for 8-PSK, starting at π/8
    let angles = [
        (0b000, PI / 8.0),
        (0b001, 3.0 * PI / 8.0),
        (0b011, 5.0 * PI / 8.0),
        (0b010, 7.0 * PI / 8.0),
        (0b110, 9.0 * PI / 8.0),
        (0b111, 11.0 * PI / 8.0),
        (0b101, 13.0 * PI / 8.0),
        (0b100, 15.0 * PI / 8.0),
    ];

    for (bits, angle) in angles.iter() {
        let i = T::from(angle.cos()).unwrap();
        let q = T::from(angle.sin()).unwrap();
        map.insert(*bits, Complex::new(i, q));
    }

    map
}

/// 16-APSK Gray-coded constellation map (4 bits per symbol)
/// DVB-S2 standard: 4 symbols on inner ring (r1=0.5), 12 on outer ring (r2=1.0)
/// Normalized for unit average power
pub fn apsk16_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();

    // DVB-S2 radii ratio γ = 2.6 for 16-APSK
    let r1 = T::from(0.5).unwrap();
    let r2 = T::from(1.0).unwrap();

    // Inner ring: 4 symbols at 0°, 90°, 180°, 270°
    let inner_angles = [
        (0b0000, 0.0),
        (0b0001, PI / 2.0),
        (0b0011, PI),
        (0b0010, 3.0 * PI / 2.0),
    ];

    for (bits, angle) in inner_angles.iter() {
        let i = r1 * T::from(angle.cos()).unwrap();
        let q = r1 * T::from(angle.sin()).unwrap();
        map.insert(*bits, Complex::new(i, q));
    }

    // Outer ring: 12 symbols
    let outer_angles = [
        (0b0110, PI / 12.0),
        (0b0111, 3.0 * PI / 12.0),
        (0b0101, 5.0 * PI / 12.0),
        (0b0100, 7.0 * PI / 12.0),
        (0b1100, 9.0 * PI / 12.0),
        (0b1101, 11.0 * PI / 12.0),
        (0b1111, 13.0 * PI / 12.0),
        (0b1110, 15.0 * PI / 12.0),
        (0b1010, 17.0 * PI / 12.0),
        (0b1011, 19.0 * PI / 12.0),
        (0b1001, 21.0 * PI / 12.0),
        (0b1000, 23.0 * PI / 12.0),
    ];

    for (bits, angle) in outer_angles.iter() {
        let i = r2 * T::from(angle.cos()).unwrap();
        let q = r2 * T::from(angle.sin()).unwrap();
        map.insert(*bits, Complex::new(i, q));
    }

    map
}

/// 16-QAM Gray-coded constellation map (4 bits per symbol)
/// Square constellation normalized to unit average power
pub fn qam16_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();

    // Normalized to unit average power
    let scale = T::from(1.0 / 10.0_f64.sqrt()).unwrap();
    let levels = [-3.0, -1.0, 1.0, 3.0];

    for (idx, &i_val) in levels.iter().enumerate() {
        for (jdx, &q_val) in levels.iter().enumerate() {
            // Gray code mapping for 16-QAM
            let i_bits = match idx {
                0 => 0b10,  // -3
                1 => 0b11,  // -1
                2 => 0b01,  //  1
                3 => 0b00,  //  3
                _ => 0,
            };
            let q_bits = match jdx {
                0 => 0b10,  // -3
                1 => 0b11,  // -1
                2 => 0b01,  //  1
                3 => 0b00,  //  3
                _ => 0,
            };

            let bits = (i_bits << 2) | q_bits;
            let i = scale * T::from(i_val).unwrap();
            let q = scale * T::from(q_val).unwrap();
            map.insert(bits, Complex::new(i, q));
        }
    }

    map
}

/// 32-QAM Gray-coded constellation map (5 bits per symbol)
/// Cross constellation normalized to unit average power
pub fn qam32_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();

    // 32-QAM uses a cross constellation pattern
    // Normalized approximately for unit average power
    let scale = T::from(0.3).unwrap();

    // Define the cross-shaped constellation points with Gray coding
    let points = [
        // Center cross (horizontal)
        (0b00000, 0.0, 0.0),
        (0b00001, 2.0, 0.0),
        (0b00011, 4.0, 0.0),
        (0b00010, -2.0, 0.0),
        (0b00110, -4.0, 0.0),

        // Center cross (vertical)
        (0b00111, 0.0, 2.0),
        (0b00101, 0.0, 4.0),
        (0b00100, 0.0, -2.0),
        (0b01100, 0.0, -4.0),

        // Upper right quadrant
        (0b01101, 2.0, 2.0),
        (0b01111, 4.0, 2.0),
        (0b01110, 2.0, 4.0),

        // Upper left quadrant
        (0b01010, -2.0, 2.0),
        (0b01011, -4.0, 2.0),
        (0b01001, -2.0, 4.0),

        // Lower right quadrant
        (0b01000, 2.0, -2.0),
        (0b11000, 4.0, -2.0),
        (0b11001, 2.0, -4.0),

        // Lower left quadrant
        (0b11011, -2.0, -2.0),
        (0b11010, -4.0, -2.0),
        (0b11110, -2.0, -4.0),

        // Outer corners
        (0b11111, 4.0, 4.0),
        (0b11101, -4.0, 4.0),
        (0b11100, 4.0, -4.0),
        (0b10100, -4.0, -4.0),

        // Additional points
        (0b10101, 6.0, 0.0),
        (0b10111, -6.0, 0.0),
        (0b10110, 0.0, 6.0),
        (0b10010, 0.0, -6.0),

        // Fill remaining
        (0b10011, 6.0, 2.0),
        (0b10001, 2.0, 6.0),
        (0b10000, -6.0, -2.0),
    ];

    for (bits, i_val, q_val) in points.iter() {
        let i = scale * T::from(*i_val).unwrap();
        let q = scale * T::from(*q_val).unwrap();
        map.insert(*bits, Complex::new(i, q));
    }

    map
}

/// 64-QAM Gray-coded constellation map (6 bits per symbol)
/// Square constellation normalized to unit average power
pub fn qam64_gray_map<T: Float>() -> HashMap<u8, Complex<T>> {
    let mut map = HashMap::new();

    // Normalized to unit average power
    let scale = T::from(1.0 / 42.0_f64.sqrt()).unwrap();
    let levels = [-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0];

    for (idx, &i_val) in levels.iter().enumerate() {
        for (jdx, &q_val) in levels.iter().enumerate() {
            // Gray code mapping for 64-QAM
            let i_bits = match idx {
                0 => 0b100,  // -7
                1 => 0b101,  // -5
                2 => 0b111,  // -3
                3 => 0b110,  // -1
                4 => 0b010,  //  1
                5 => 0b011,  //  3
                6 => 0b001,  //  5
                7 => 0b000,  //  7
                _ => 0,
            };
            let q_bits = match jdx {
                0 => 0b100,  // -7
                1 => 0b101,  // -5
                2 => 0b111,  // -3
                3 => 0b110,  // -1
                4 => 0b010,  //  1
                5 => 0b011,  //  3
                6 => 0b001,  //  5
                7 => 0b000,  //  7
                _ => 0,
            };

            let bits = (i_bits << 3) | q_bits;
            let i = scale * T::from(i_val).unwrap();
            let q = scale * T::from(q_val).unwrap();
            map.insert(bits, Complex::new(i, q));
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bpsk_map() {
        let map = bpsk_map::<f32>();
        assert_eq!(map.len(), 2);

        for val in map.values() {
            let power = val.norm_sqr();
            assert!((power - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_qpsk_gray_map() {
        let map = qpsk_gray_map::<f64>();
        assert_eq!(map.len(), 4);

        // Verify unit power normalization
        for val in map.values() {
            let power = val.norm_sqr();
            assert!((power - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_psk8_gray_map() {
        let map = psk8_gray_map::<f64>();
        assert_eq!(map.len(), 8);

        // Verify unit circle
        for val in map.values() {
            let radius = val.norm();
            assert!((radius - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_apsk16_gray_map() {
        let map = apsk16_gray_map::<f64>();
        assert_eq!(map.len(), 16);
    }

    #[test]
    fn test_qam16_gray_map() {
        let map = qam16_gray_map::<f64>();
        assert_eq!(map.len(), 16);
    }

    #[test]
    fn test_qam32_gray_map() {
        let map = qam32_gray_map::<f64>();
        assert_eq!(map.len(), 32);
    }

    #[test]
    fn test_qam64_gray_map() {
        let map = qam64_gray_map::<f64>();
        assert_eq!(map.len(), 64);
    }

    #[test]
    fn test_qpsk_constellation_plot() {
        use std::env;
        use crate::random_bit_generator::BitGenerator;
        use crate::plot::plot_constellation;

        let plot = env::var("PLOT").unwrap_or_else(|_| "false".to_string());
        if plot.to_lowercase() != "true" {
            println!("Skipping constellation plot (set PLOT=true to enable)");
            return;
        }

        // Get QPSK constellation map
        let map = qpsk_gray_map::<f64>();

        // Generate actual QPSK symbols
        let mut bit_gen = BitGenerator::new_from_seed(42);
        let num_symbols = 1000;
        let mut symbols = Vec::with_capacity(num_symbols);

        for _ in 0..num_symbols {
            let bits = bit_gen.next_2_bits();
            if let Some(symbol) = map.get(&bits) {
                symbols.push(*symbol);
            }
        }

        // Plot the constellation using the utility function
        plot_constellation(&symbols, "QPSK Constellation");
    }
}
