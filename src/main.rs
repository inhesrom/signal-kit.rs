mod random_bit_generator;
mod symbol_mapper;

use random_bit_generator::BitGenerator;
use symbol_mapper::MapDemap;

use num_complex::Complex32;

fn main()
{
    let mut bit_gen = BitGenerator::new_from_seed(42);

    let mut iq_samples = Vec::<Complex32>::new();
    let mapper = MapDemap::<f32>::new();

    for _ in 0..80_000_000 {
        let two_bits = bit_gen.next_2_bits();
        // println!("2 bits: {}", two_bits); // 0-3
        iq_samples.push(mapper.modulate(two_bits));
        // println!("{:?}", iq_samples[i]);
    }

}
