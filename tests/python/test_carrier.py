"""
Tests for the Python bindings of signal-kit Carrier.
"""

import pytest
import numpy as np

# Import will work when run via maturin develop
try:
    import signal_kit
except ImportError:
    pytest.skip("signal-kit not installed, run 'maturin develop' first", allow_module_level=True)


class TestCarrierCreation:
    """Test Carrier object creation and validation."""

    def test_create_qpsk_carrier(self):
        """Test creating a QPSK carrier."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        assert carrier is not None

    def test_create_carrier_without_seed(self):
        """Test creating a carrier without a seed (entropy-based)."""
        carrier = signal_kit.Carrier(
            modulation="BPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6
        )
        assert carrier is not None

    def test_invalid_modulation(self):
        """Test that invalid modulation type raises error."""
        with pytest.raises(ValueError, match="Unknown modulation type"):
            signal_kit.Carrier(
                modulation="INVALID_MOD",
                bandwidth=0.1,
                center_freq=0.0,
                snr_db=10.0,
                rolloff=0.35,
                sample_rate_hz=1e6
            )

    def test_invalid_bandwidth_too_large(self):
        """Test that bandwidth > 1.0 raises error."""
        with pytest.raises(ValueError, match="bandwidth must be in range"):
            signal_kit.Carrier(
                modulation="QPSK",
                bandwidth=1.5,
                center_freq=0.0,
                snr_db=10.0,
                rolloff=0.35,
                sample_rate_hz=1e6
            )

    def test_invalid_bandwidth_zero(self):
        """Test that bandwidth <= 0 raises error."""
        with pytest.raises(ValueError, match="bandwidth must be in range"):
            signal_kit.Carrier(
                modulation="QPSK",
                bandwidth=0.0,
                center_freq=0.0,
                snr_db=10.0,
                rolloff=0.35,
                sample_rate_hz=1e6
            )

    def test_invalid_center_freq_too_large(self):
        """Test that center_freq > 0.5 raises error."""
        with pytest.raises(ValueError, match="center_freq must be in range"):
            signal_kit.Carrier(
                modulation="QPSK",
                bandwidth=0.1,
                center_freq=0.6,
                snr_db=10.0,
                rolloff=0.35,
                sample_rate_hz=1e6
            )

    def test_invalid_sample_rate(self):
        """Test that sample_rate_hz <= 0 raises error."""
        with pytest.raises(ValueError, match="sample_rate_hz must be positive"):
            signal_kit.Carrier(
                modulation="QPSK",
                bandwidth=0.1,
                center_freq=0.0,
                snr_db=10.0,
                rolloff=0.35,
                sample_rate_hz=0
            )


class TestCarrierGeneration:
    """Test signal generation from Carrier objects."""

    def test_generate_returns_numpy_array(self):
        """Test that generate() returns a numpy array."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq = carrier.generate(1000)
        assert isinstance(iq, np.ndarray)

    def test_generate_correct_length(self):
        """Test that generate() returns correct number of samples."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        for num_samples in [100, 1000, 10000]:
            iq = carrier.generate(num_samples)
            assert len(iq) == num_samples

    def test_generate_correct_dtype(self):
        """Test that generate() returns complex128 array."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq = carrier.generate(1000)
        assert iq.dtype == np.complex128

    def test_generate_reproducible_with_seed(self):
        """Test that same seed produces same signal."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq1 = carrier1.generate(1000)

        carrier2 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq2 = carrier2.generate(1000)

        np.testing.assert_array_equal(iq1, iq2)

    def test_generate_different_without_seed(self):
        """Test that different seeds produce different signals."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq1 = carrier1.generate(1000)

        carrier2 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=43
        )
        iq2 = carrier2.generate(1000)

        assert not np.allclose(iq1, iq2)

    @pytest.mark.parametrize("modulation", ["BPSK", "QPSK", "8PSK", "16QAM", "CW"])
    def test_generate_various_modulations(self, modulation):
        """Test that generate works with various modulation types."""
        carrier = signal_kit.Carrier(
            modulation=modulation,
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq = carrier.generate(1000)
        assert len(iq) == 1000
        assert iq.dtype == np.complex128


class TestCarrierCombination:
    """Test combining multiple carriers."""

    def test_combine_two_carriers(self):
        """Test combining two carrier signals."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.1,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq1 = carrier1.generate(1000)

        carrier2 = signal_kit.Carrier(
            modulation="8PSK",
            bandwidth=0.05,
            center_freq=-0.2,
            snr_db=5.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=43
        )
        iq2 = carrier2.generate(1000)

        # Combine using numpy addition
        combined = iq1 + iq2
        assert isinstance(combined, np.ndarray)
        assert combined.dtype == np.complex128
        assert len(combined) == 1000

    def test_combine_multiple_carriers(self):
        """Test combining more than two carriers."""
        carriers = []
        iqs = []

        for i in range(3):
            carrier = signal_kit.Carrier(
                modulation="QPSK",
                bandwidth=0.05,
                center_freq=-0.2 + i * 0.2,
                snr_db=10.0 - i * 2,
                rolloff=0.35,
                sample_rate_hz=1e6,
                seed=42 + i
            )
            carriers.append(carrier)
            iqs.append(carrier.generate(1000))

        # Combine all carriers
        combined = np.sum(iqs, axis=0)
        assert len(combined) == 1000
        assert combined.dtype == np.complex128


class TestCarrierProperties:
    """Test carrier signal properties."""

    def test_signal_has_complex_content(self):
        """Test that generated signal has both real and imaginary parts."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq = carrier.generate(1000)

        # Check that we have non-zero real and imaginary parts
        assert np.any(np.abs(iq.real) > 0)
        assert np.any(np.abs(iq.imag) > 0)

    def test_frequency_shifted_carrier(self):
        """Test that center_freq parameter shifts the signal."""
        carrier_baseband = signal_kit.Carrier(
            modulation="CW",
            bandwidth=0.01,
            center_freq=0.0,
            snr_db=20.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq_baseband = carrier_baseband.generate(1000)

        carrier_shifted = signal_kit.Carrier(
            modulation="CW",
            bandwidth=0.01,
            center_freq=0.1,  # 100 kHz shift
            snr_db=20.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq_shifted = carrier_shifted.generate(1000)

        # Signals should be different due to frequency shift
        assert not np.allclose(iq_baseband, iq_shifted)

    def test_snr_affects_noise_level(self):
        """Test that SNR parameter affects signal noise content."""
        carrier_low_snr = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=1.0,  # Very low SNR
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq_low = carrier_low_snr.generate(10000)

        carrier_high_snr = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=30.0,  # Very high SNR
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        iq_high = carrier_high_snr.generate(10000)

        # Estimate noise by looking at variance
        # Note: This is a rough test since deterministic signal changes with SNR
        signal_low = iq_low[0]  # Just a rough proxy
        signal_high = iq_high[0]

        # Signals should be different (noise is added)
        assert signal_low != signal_high
