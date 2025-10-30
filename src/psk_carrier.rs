#![allow(dead_code)]

pub struct PskCarrier<T> {
    sample_rate: T,
    symbol_rate: T,
    mod_type: ModType,
    rolloff_factor: T,
    block_size: usize,
    current_sample_num: usize
}

impl<T> PskCarrier<T> {

}
