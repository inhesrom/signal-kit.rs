use bitvec::prelude::*;
use rand::prelude::*;
use rand::Rng;
use std::error::Error;

pub struct RandomBitGenerator {
    rng: StdRng,
    dist: rand::distributions::Bernoulli,
}

impl RandomBitGenerator {
    // Create a new generator with hardware random seed
    pub fn new() -> Self {
        Self {
            rng: StdRng::from_entropy(),
            dist: rand::distributions::Bernoulli::new(0.5).unwrap(),
        }
    }

    // Create a new generator with specified seed
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            dist: rand::distributions::Bernoulli::new(0.5).unwrap(),
        }
    }

    // Generate a single random bit
    pub fn gen_bit(&mut self) -> bool {
        self.dist.sample(&mut self.rng)
    }

    // Generate n random bits as BitVec
    pub fn generate_bits(&mut self, n: usize) -> BitVec {
        let mut bits = BitVec::with_capacity(n);
        for _ in 0..n {
            bits.push(self.gen_bit());
        }
        bits
    }

    // Generate exactly 64 random bits
    pub fn get_bits_64(&mut self) -> BitVec {
        let mut bits = BitVec::with_capacity(64);
        let random_value = self.rng.gen::<u64>();
        for i in 0..64 {
            bits.push((random_value >> i) & 1 == 1);
        }
        bits
    }

    // Generate n random bits packed into an integer
    pub fn get_bits_as_int(&mut self, n: usize) -> Result<u64, Box<dyn Error>> {
        if n > 64 {
            return Err("Can't generate more than 64 bits as int".into());
        }
        
        let bits = self.generate_bits(n);
        let mut result = 0u64;
        for (i, &bit) in bits.iter().enumerate().take(n) {
            if bit {
                result |= 1u64 << i;
            }
        }
        Ok(result)
    }
}

// Implement Default trait to match C++ default constructor behavior
impl Default for RandomBitGenerator {
    fn default() -> Self {
        Self::new()
    }
}
