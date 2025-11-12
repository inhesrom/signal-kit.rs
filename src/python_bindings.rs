#![cfg(feature = "python")]

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, IntoPyArray};
use num_complex::Complex;
use crate::{Carrier, Channel, ModType};
use crate::generate::Impairment;

/// Helper function to parse modulation type from string
///
/// Supports the following modulation types (case-insensitive):
///   - BPSK: Binary Phase Shift Keying
///   - QPSK: Quadrature Phase Shift Keying
///   - 8PSK: 8-ary Phase Shift Keying
///   - 16APSK: 16-ary Amplitude-Phase Shift Keying
///   - 16QAM: 16-ary Quadrature Amplitude Modulation
///   - 32QAM: 32-ary Quadrature Amplitude Modulation
///   - 64QAM: 64-ary Quadrature Amplitude Modulation
///   - CW: Continuous Wave (unmodulated carrier)
fn parse_modulation(mod_str: &str) -> PyResult<ModType> {
    match mod_str.to_uppercase().as_str() {
        "BPSK" => Ok(ModType::_BPSK),
        "QPSK" => Ok(ModType::_QPSK),
        "8PSK" => Ok(ModType::_8PSK),
        "16APSK" => Ok(ModType::_16APSK),
        "16QAM" => Ok(ModType::_16QAM),
        "32QAM" => Ok(ModType::_32QAM),
        "64QAM" => Ok(ModType::_64QAM),
        "CW" => Ok(ModType::_CW),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!(
                "Unknown modulation type: '{}'. Supported types (case-insensitive):\n  \
                 BPSK (Binary PSK), QPSK (Quadrature PSK), 8PSK, 16APSK, \n  \
                 16QAM, 32QAM, 64QAM, CW (Continuous Wave)",
                mod_str
            ),
        )),
    }
}

/// Helper function to parse impairment configuration from Python
///
/// Accepts either a string shorthand or a dictionary with detailed parameters.
///
/// String formats:
///     "digitizer_droop_ad9361" - 3rd order, 0.45 cutoff (AD9361 SDR style)
///     "digitizer_droop_traditional" - 6th order, 0.42 cutoff (traditional)
///     "cosine_taper_digitizer" - cosine taper, 0.42-0.48 transition
///
/// Dictionary format:
///     {"type": "digitizer_droop", "order": 3, "cutoff": 0.45}
///     {"type": "frequency_variation", "amplitude_db": 1.0, "cycles": 3.0, "phase_offset": 0.0}
///     {"type": "cosine_taper", "passband_end": 0.42, "stopband_start": 0.48}
fn parse_impairment(config: &Bound<PyAny>) -> PyResult<Impairment> {
    // Try as string first
    if let Ok(s) = config.extract::<String>() {
        return match s.to_lowercase().as_str() {
            "digitizer_droop_ad9361" => Ok(Impairment::DigitizerDroopAD9361),
            "digitizer_droop_traditional" => Ok(Impairment::DigitizerDroopTraditional),
            "cosine_taper_digitizer" => Ok(Impairment::CosineTaperDigitizer),
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Unknown impairment: '{}'. Supported strings: digitizer_droop_ad9361, digitizer_droop_traditional, cosine_taper_digitizer. Or use a dict with 'type' key.",
                    s
                ),
            )),
        };
    }

    // Try as dictionary
    if let Ok(dict) = config.downcast::<PyDict>() {
        let impairment_type: String = dict
            .get_item("type")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Dictionary must have 'type' key",
            ))?
            .extract()?;

        return match impairment_type.to_lowercase().as_str() {
            "digitizer_droop" => {
                let order: i32 = dict
                    .get_item("order")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'digitizer_droop' requires 'order' parameter",
                    ))?
                    .extract()?;
                let cutoff: f64 = dict
                    .get_item("cutoff")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'digitizer_droop' requires 'cutoff' parameter",
                    ))?
                    .extract()?;

                if !(0.0..=0.5).contains(&cutoff) {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "cutoff must be in range [0.0, 0.5]",
                    ));
                }

                Ok(Impairment::DigitizerDroop { order, cutoff })
            }
            "frequency_variation" => {
                let amplitude_db: f64 = dict
                    .get_item("amplitude_db")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'frequency_variation' requires 'amplitude_db' parameter",
                    ))?
                    .extract()?;
                let cycles: f64 = dict
                    .get_item("cycles")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'frequency_variation' requires 'cycles' parameter",
                    ))?
                    .extract()?;
                let phase_offset: f64 = match dict.get_item("phase_offset") {
                    Ok(Some(v)) => v.extract().unwrap_or(0.0),
                    _ => 0.0,
                };

                Ok(Impairment::FrequencyVariation {
                    amplitude_db,
                    cycles,
                    phase_offset,
                })
            }
            "cosine_taper" => {
                let passband_end: f64 = dict
                    .get_item("passband_end")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'cosine_taper' requires 'passband_end' parameter",
                    ))?
                    .extract()?;
                let stopband_start: f64 = dict
                    .get_item("stopband_start")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "'cosine_taper' requires 'stopband_start' parameter",
                    ))?
                    .extract()?;

                if stopband_start <= passband_end {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "stopband_start must be greater than passband_end",
                    ));
                }

                Ok(Impairment::CosineTaper {
                    passband_end,
                    stopband_start,
                })
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!(
                    "Unknown impairment type: '{}'. Supported types: digitizer_droop, frequency_variation, cosine_taper",
                    impairment_type
                ),
            )),
        };
    }

    Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
        "Impairment config must be a string or dictionary",
    ))
}

/// Python wrapper for the Carrier struct
///
/// A high-level carrier generator that combines modulation, frequency shifting, and noise.
///
/// Parameters:
///     modulation (str): Modulation type - one of: BPSK, QPSK, 8PSK, 16APSK, 16QAM, 32QAM, 64QAM, CW
///     bandwidth (float): Normalized occupied bandwidth (0.0-1.0)
///     center_freq (float): Normalized center frequency (-0.5 to 0.5)
///     snr_db (float): Target signal-to-noise ratio in dB
///     rolloff (float): RRC rolloff factor (0.0-1.0), typically 0.35
///     sample_rate_hz (float): Sample rate in Hz
///     seed (int, optional): Seed for reproducible generation
///
/// Example:
///     >>> import signal_kit
///     >>> carrier = signal_kit.Carrier(
///     ...     modulation="QPSK",
///     ...     bandwidth=0.1,
///     ...     center_freq=0.1,
///     ...     snr_db=10.0,
///     ...     rolloff=0.35,
///     ...     sample_rate_hz=1e6,
///     ...     seed=42
///     ... )
///     >>> iq_samples = carrier.generate(1000)
#[pyclass(name = "Carrier")]
pub struct PythonCarrier {
    inner: Carrier,
}

#[pymethods]
impl PythonCarrier {
    #[new]
    #[pyo3(signature = (modulation, bandwidth, center_freq, snr_db, rolloff, sample_rate_hz, seed=None))]
    fn new(
        modulation: &str,
        bandwidth: f64,
        center_freq: f64,
        snr_db: f64,
        rolloff: f64,
        sample_rate_hz: f64,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let mod_type = parse_modulation(modulation)?;

        // Validate parameters (same validation as Rust API)
        if bandwidth <= 0.0 || bandwidth > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "bandwidth must be in range (0.0, 1.0]",
            ));
        }
        if center_freq < -0.5 || center_freq > 0.5 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "center_freq must be in range [-0.5, 0.5]",
            ));
        }
        if rolloff < 0.0 || rolloff > 1.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "rolloff must be in range [0.0, 1.0]",
            ));
        }
        if sample_rate_hz <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "sample_rate_hz must be positive",
            ));
        }

        Ok(PythonCarrier {
            inner: Carrier::new(
                mod_type,
                bandwidth,
                center_freq,
                snr_db,
                rolloff,
                sample_rate_hz,
                seed,
            ),
        })
    }

    /// Generate IQ samples from the carrier with noise
    ///
    /// Parameters:
    ///     num_samples (int): Number of samples to generate
    ///
    /// Returns:
    ///     numpy.ndarray: Complex-valued IQ samples as numpy array (complex128)
    fn generate(&self, py: Python, num_samples: usize) -> Py<PyArray1<Complex<f64>>> {
        let samples = self.inner.generate::<f64>(num_samples);

        // Convert ComplexVec to Vec<Complex<f64>>
        let vec: Vec<Complex<f64>> = (0..samples.len())
            .map(|i| samples[i])
            .collect();

        // Transfer ownership to Python
        let array = vec.into_pyarray_bound(py);
        array.into()
    }

    /// Generate clean IQ samples without noise
    ///
    /// Useful for multi-carrier scenarios where noise is added to the combined signal.
    ///
    /// Parameters:
    ///     num_samples (int): Number of samples to generate
    ///
    /// Returns:
    ///     numpy.ndarray: Complex-valued IQ samples as numpy array (complex128)
    fn generate_clean(&self, py: Python, num_samples: usize) -> Py<PyArray1<Complex<f64>>> {
        let samples = self.inner.generate_clean::<f64>(num_samples);

        // Convert ComplexVec to Vec<Complex<f64>>
        let vec: Vec<Complex<f64>> = (0..samples.len())
            .map(|i| samples[i])
            .collect();

        // Transfer ownership to Python
        let array = vec.into_pyarray_bound(py);
        array.into()
    }
}

/// Python wrapper for the Channel struct
///
/// Manages multiple carriers combined into a single channel with shared AWGN.
/// Noise is added once to the combined signal, modeling realistic transponder scenarios.
///
/// Example:
///     >>> import signal_kit
///     >>> carrier1 = signal_kit.Carrier("QPSK", 0.1, 0.1, 10.0, 0.35, 1e6, seed=42)
///     >>> carrier2 = signal_kit.Carrier("QPSK", 0.1, -0.1, 10.0, 0.35, 1e6, seed=43)
///     >>> channel = signal_kit.Channel([carrier1, carrier2])
///     >>> channel.set_noise_floor_db(-100.0)
///     >>> iq_combined = channel.generate(10000)
#[pyclass(name = "Channel")]
pub struct PythonChannel {
    inner: Channel,
}

#[pymethods]
impl PythonChannel {
    #[new]
    fn new(carriers: Vec<PyRef<PythonCarrier>>) -> PyResult<Self> {
        let rust_carriers: Vec<Carrier> = carriers
            .into_iter()
            .map(|c| c.inner.clone())
            .collect();

        if rust_carriers.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Channel must have at least one carrier",
            ));
        }

        Ok(PythonChannel {
            inner: Channel::new(rust_carriers),
        })
    }

    /// Set the noise floor in dB
    ///
    /// Parameters:
    ///     noise_floor_db (float): Noise floor in dB (e.g., -100.0)
    fn set_noise_floor_db(&mut self, noise_floor_db: f64) {
        self.inner.set_noise_floor_db(noise_floor_db);
    }

    /// Set the noise floor in linear units
    ///
    /// Parameters:
    ///     noise_floor_linear (float): Noise floor as a linear power value
    fn set_noise_floor_linear(&mut self, noise_floor_linear: f64) -> PyResult<()> {
        if noise_floor_linear <= 0.0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "noise_floor_linear must be positive",
            ));
        }
        self.inner.set_noise_floor_linear(noise_floor_linear);
        Ok(())
    }

    /// Set the seed for reproducible AWGN generation
    ///
    /// Parameters:
    ///     seed (int): Seed value for the AWGN random number generator
    fn set_seed(&mut self, seed: u64) {
        self.inner.set_seed(seed);
    }

    /// Generate combined carrier signal with shared AWGN
    ///
    /// Parameters:
    ///     num_samples (int): Number of samples to generate
    ///
    /// Returns:
    ///     numpy.ndarray: Complex-valued IQ samples as numpy array (complex128)
    fn generate(&self, py: Python, num_samples: usize) -> PyResult<Py<PyArray1<Complex<f64>>>> {
        let samples = self.inner.generate::<f64>(num_samples);

        // Convert ComplexVec to Vec<Complex<f64>>
        let vec: Vec<Complex<f64>> = (0..samples.len())
            .map(|i| samples[i])
            .collect();

        // Transfer ownership to Python
        let array = vec.into_pyarray_bound(py);
        Ok(array.into())
    }

    /// Generate combined carrier signal without noise
    ///
    /// Useful for analysis or when you want to add noise separately.
    ///
    /// Parameters:
    ///     num_samples (int): Number of samples to generate
    ///
    /// Returns:
    ///     numpy.ndarray: Complex-valued IQ samples as numpy array (complex128)
    fn generate_clean(&self, py: Python, num_samples: usize) -> Py<PyArray1<Complex<f64>>> {
        let samples = self.inner.generate_clean::<f64>(num_samples);

        // Convert ComplexVec to Vec<Complex<f64>>
        let vec: Vec<Complex<f64>> = (0..samples.len())
            .map(|i| samples[i])
            .collect();

        // Transfer ownership to Python
        let array = vec.into_pyarray_bound(py);
        array.into()
    }

    /// Get the number of carriers in this channel
    fn num_carriers(&self) -> usize {
        self.inner.num_carriers()
    }

    /// Add a channel impairment (e.g., digitizer droop, frequency variation)
    ///
    /// Impairments are applied to the combined signal after noise addition.
    /// Multiple impairments can be added and will be applied sequentially.
    ///
    /// Parameters:
    ///     impairment: Either a string shorthand or a dictionary with parameters
    ///
    /// String examples:
    ///     "digitizer_droop_ad9361" - 3rd order, 0.45 cutoff (AD9361 SDR style)
    ///     "digitizer_droop_traditional" - 6th order, 0.42 cutoff (high-quality)
    ///     "cosine_taper_digitizer" - cosine taper, 0.42-0.48 transition
    ///
    /// Dictionary examples:
    ///     {"type": "digitizer_droop", "order": 3, "cutoff": 0.45}
    ///     {"type": "frequency_variation", "amplitude_db": 1.0, "cycles": 3.0, "phase_offset": 0.0}
    ///     {"type": "cosine_taper", "passband_end": 0.42, "stopband_start": 0.48}
    ///
    /// Example:
    ///     >>> channel.add_impairment("digitizer_droop_ad9361")
    ///     >>> channel.add_impairment({"type": "frequency_variation", "amplitude_db": 1.0, "cycles": 3.0})
    fn add_impairment(&mut self, py: Python, impairment: &Bound<PyAny>) -> PyResult<()> {
        let imp = parse_impairment(impairment)?;
        self.inner.add_impairment(imp);
        Ok(())
    }
}

/// Initialize the Python module
#[pymodule]
fn _signal_kit(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PythonCarrier>()?;
    m.add_class::<PythonChannel>()?;
    m.add("__version__", "0.1.0")?;

    Ok(())
}
