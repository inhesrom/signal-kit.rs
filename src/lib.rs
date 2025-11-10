// Core data structures and utilities
pub mod complex_vec;
pub mod fft;
pub mod mod_type;
pub mod symbol_maps;
pub mod vector_ops;
pub mod vector_simd;

// Organized modules
pub mod filter;
pub mod generate;
pub mod spectrum;

// Python bindings (conditional)
#[cfg(feature = "python")]
mod python_bindings;

// Public re-exports for convenience
pub use complex_vec::ComplexVec;
pub use mod_type::ModType;

// Re-export commonly used items from submodules
pub use generate::{Carrier, Channel};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// Plot module only available in tests
#[cfg(test)]
pub mod plot;