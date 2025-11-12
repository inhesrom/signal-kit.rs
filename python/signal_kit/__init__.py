"""
signal-kit: Digital Signal Processing for Communication Systems

A Rust library with Python bindings for digital signal processing, focused on
communication systems. Implements components for generating, modulating, filtering,
and processing digital signals commonly used in wireless communications.

Module Contents:
    - Carrier: High-level signal generator with modulation and noise
    - Channel: Multi-carrier channel simulator with shared AWGN and impairments

Supported Modulations:
    BPSK, QPSK, 8PSK, 16APSK, 16QAM, 32QAM, 64QAM, CW

Single Carrier Example:
    >>> import signal_kit
    >>>
    >>> # Create a QPSK carrier
    >>> carrier = signal_kit.Carrier(
    ...     modulation="QPSK",
    ...     bandwidth=0.1,
    ...     center_freq=0.1,
    ...     snr_db=10.0,
    ...     rolloff=0.35,
    ...     sample_rate_hz=1e6,
    ...     seed=42
    ... )
    >>>
    >>> # Generate IQ samples (returns numpy array with noise)
    >>> iq_samples = carrier.generate(1000)

Multi-Carrier Channel Example:
    >>> # Create multiple carriers
    >>> carrier1 = signal_kit.Carrier(
    ...     modulation="QPSK",
    ...     bandwidth=0.1,
    ...     center_freq=0.1,
    ...     snr_db=10.0,
    ...     rolloff=0.35,
    ...     sample_rate_hz=1e6,
    ...     seed=42
    ... )
    >>> carrier2 = signal_kit.Carrier(
    ...     modulation="16QAM",
    ...     bandwidth=0.15,
    ...     center_freq=-0.15,
    ...     snr_db=15.0,
    ...     rolloff=0.35,
    ...     sample_rate_hz=1e6,
    ...     seed=43
    ... )
    >>>
    >>> # Create channel with multiple carriers and shared noise
    >>> channel = signal_kit.Channel([carrier1, carrier2])
    >>> channel.set_noise_floor_db(-85.0)
    >>> iq_combined = channel.generate(10000)

Channel Impairments Example:
    >>> # Add realistic impairments to simulate channel effects
    >>> channel.add_impairment("digitizer_droop_ad9361")
    >>>
    >>> # Or add custom frequency-dependent ripple
    >>> channel.add_impairment({
    ...     "type": "frequency_variation",
    ...     "amplitude_db": 1.0,
    ...     "cycles": 3.0,
    ...     "phase_offset": 0.0
    ... })
    >>>
    >>> # Generate signal with impairments applied
    >>> iq_impaired = channel.generate(10000)

Supported Impairments:
    - "digitizer_droop_ad9361": 3rd order @ 45% Nyquist (SDR style)
    - "digitizer_droop_traditional": 6th order @ 42% Nyquist (high-quality)
    - "cosine_taper_digitizer": Cosine taper @ 42-48% transition
    - {"type": "digitizer_droop", "order": N, "cutoff": X.XX}
    - {"type": "frequency_variation", "amplitude_db": X, "cycles": Y, "phase_offset": Z}
    - {"type": "cosine_taper", "passband_end": X, "stopband_start": Y}
"""

from ._signal_kit import Carrier, Channel, welch

__version__ = "0.2.0"
__all__ = ["Carrier", "Channel", "welch"]


def combine_carriers(*carrier_arrays):
    """
    Combine multiple carrier IQ samples by element-wise addition.

    Parameters:
        *carrier_arrays: Variable number of numpy arrays to combine

    Returns:
        numpy.ndarray: Combined signal

    Example:
        >>> import signal_kit as sk
        >>> carrier1 = sk.Carrier(...).generate(1000)
        >>> carrier2 = sk.Carrier(...).generate(1000)
        >>> combined = sk.combine_carriers(carrier1, carrier2)
    """
    import numpy as np
    return np.sum(carrier_arrays, axis=0)
