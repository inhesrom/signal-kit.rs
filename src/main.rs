mod random_bit_generator;
mod symbol_mapper;

use random_bit_generator::BitGenerator;
use symbol_mapper::Mapper;

fn main()
{
    let mut bit_gen = BitGenerator::new_from_seed(42);

    // Pull out 2 bits at a time
    for _ in 0..10 {
        let two_bits = bit_gen.next_2_bits();
        println!("2 bits: {}", two_bits); // 0-3
    }

    // Pull out 3 bits at a time
    for _ in 0..10 {
        let three_bits = bit_gen.next_3_bits();
        println!("3 bits: {}", three_bits); // 0-7
    }

    // Single bits
    for _ in 0..10 {
        let bit = bit_gen.next_bit();
        println!("1 bit: {}", bit);
    }

    // Variable amounts
    println!("5 bits: {}", bit_gen.next_bits(5)); // 0-31
}
