"""
Tests for the Python Welch PSD binding.
"""

import pytest
import numpy as np

try:
    import signal_kit
except ImportError:
    pytest.skip("signal-kit not installed, run 'maturin develop' first", allow_module_level=True)


def sample_signal():
    """Create a simple complex-valued signal for Welch validation tests."""
    return np.ones(16, dtype=np.complex128)


class TestWelchValidation:
    """Test Welch argument validation."""

    def test_rejects_zero_segment_length(self):
        """Test that nperseg == 0 raises error."""
        with pytest.raises(ValueError, match="nperseg must be greater than 0"):
            signal_kit.welch(sample_signal(), sample_rate=8.0, nperseg=0)

    @pytest.mark.parametrize("sample_rate", [0.0, -8.0])
    def test_rejects_non_positive_sample_rate(self, sample_rate):
        """Test that sample_rate <= 0 raises error."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            signal_kit.welch(sample_signal(), sample_rate=sample_rate, nperseg=8)

    @pytest.mark.parametrize("noverlap", [8, 9])
    def test_rejects_invalid_overlap(self, noverlap):
        """Test that noverlap >= nperseg raises error."""
        with pytest.raises(ValueError, match="noverlap must be less than nperseg"):
            signal_kit.welch(
                sample_signal(),
                sample_rate=8.0,
                nperseg=8,
                noverlap=noverlap,
            )

    def test_rejects_short_fft(self):
        """Test that nfft < nperseg raises error."""
        with pytest.raises(
            ValueError,
            match="nfft must be greater than or equal to nperseg",
        ):
            signal_kit.welch(
                sample_signal(),
                sample_rate=8.0,
                nperseg=8,
                nfft=4,
            )
