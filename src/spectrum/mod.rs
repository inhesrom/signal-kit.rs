pub mod welch;
pub mod window;

pub use welch::{welch, AveragingMethod};
pub use window::{WindowType, generate_window, window_energy};