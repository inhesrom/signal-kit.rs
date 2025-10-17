use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;

pub struct BitGenerator {
    rng: StdRng,
    buffer: u64,
    bits_remaining: u8,
}

impl BitGenerator {
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            buffer: 0,
            bits_remaining: 0,
        }
    }

    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            buffer: 0,
            bits_remaining: 0,
        }
    }

    // Get N bits (1-64) as a u64
    pub fn next_bits(&mut self, num_bits: u8) -> u64 {
        assert!(num_bits > 0 && num_bits <= 64, "num_bits must be 1-64");

        // Refill buffer if we don't have enough bits
        if self.bits_remaining < num_bits {
            self.buffer = self.rng.next_u64();
            self.bits_remaining = 64;
        }

        // Extract the requested bits from the low end
        let mask = if num_bits == 64 {
            u64::MAX
        } else {
            (1u64 << num_bits) - 1
        };
        let result = self.buffer & mask;

        // Shift buffer and update remaining count
        self.buffer >>= num_bits;
        self.bits_remaining -= num_bits;

        result
    }

    // Convenience methods for common cases
    pub fn next_bit(&mut self) -> bool {
        self.next_bits(1) != 0
    }

    pub fn next_2_bits(&mut self) -> u8 {
        self.next_bits(2) as u8
    }

    pub fn next_3_bits(&mut self) -> u8 {
        self.next_bits(3) as u8
    }
}
