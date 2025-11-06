"""
signal-kit: Digital Signal Processing for Communication Systems

A Rust library with Python bindings for digital signal processing, focused on
communication systems. Implements components for generating, modulating, filtering,
and processing digital signals commonly used in wireless communications.

Example:
    >>> import signal_kit
    >>> import numpy as np
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
    >>> # Generate IQ samples
    >>> iq_samples = carrier.generate(1000)
    >>>
    >>> # Combine multiple carriers
    >>> carrier2 = signal_kit.Carrier(
    ...     modulation="8PSK",
    ...     bandwidth=0.05,
    ...     center_freq=-0.2,
    ...     snr_db=5.0,
    ...     rolloff=0.35,
    ...     sample_rate_hz=1e6,
    ...     seed=43
    ... )
    >>> iq2 = carrier2.generate(1000)
    >>> combined = iq_samples + iq2  # Simulate multiple signals in a channel
"""

from ._signal_kit import Carrier

__version__ = "0.1.0"
__all__ = ["Carrier"]


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
