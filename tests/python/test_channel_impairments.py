"""
Tests for the Python bindings of signal-kit Channel with impairments.
"""

import pytest
import numpy as np

# Import will work when run via maturin develop
try:
    import signal_kit
except ImportError:
    pytest.skip("signal-kit not installed, run 'maturin develop' first", allow_module_level=True)


class TestChannelCreation:
    """Test Channel object creation and validation."""

    def test_create_channel_single_carrier(self):
        """Test creating a channel with a single carrier."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        assert channel is not None

    def test_create_channel_multiple_carriers(self):
        """Test creating a channel with multiple carriers."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.1,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        carrier2 = signal_kit.Carrier(
            modulation="16QAM",
            bandwidth=0.15,
            center_freq=-0.15,
            snr_db=15.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=43
        )
        channel = signal_kit.Channel([carrier1, carrier2])
        assert channel is not None
        assert channel.num_carriers() == 2

    def test_create_channel_empty_raises_error(self):
        """Test that creating a channel with no carriers raises error."""
        with pytest.raises(ValueError, match="Channel must have at least one carrier"):
            signal_kit.Channel([])


class TestChannelGeneration:
    """Test signal generation from Channel objects."""

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
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)
        iq = channel.generate(1000)
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
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        for num_samples in [100, 1000, 10000]:
            iq = channel.generate(num_samples)
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
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)
        iq = channel.generate(1000)
        assert iq.dtype == np.complex128

    def test_generate_clean_returns_numpy_array(self):
        """Test that generate_clean() returns a numpy array without noise."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)  # Must set noise floor even for clean generation
        iq = channel.generate_clean(1000)
        assert isinstance(iq, np.ndarray)
        assert len(iq) == 1000
        assert iq.dtype == np.complex128


class TestChannelNoiseFloor:
    """Test channel noise floor configuration."""

    def test_set_noise_floor_db(self):
        """Test setting noise floor in dB."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-90.0)
        # Should not raise an error
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_set_noise_floor_linear(self):
        """Test setting noise floor in linear units."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_linear(0.001)  # -30 dB
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_set_noise_floor_linear_invalid(self):
        """Test that setting invalid noise floor raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        with pytest.raises(ValueError, match="noise_floor_linear must be positive"):
            channel.set_noise_floor_linear(-0.001)

    def test_set_seed(self):
        """Test setting seed for reproducible channel generation."""
        # Create separate carriers for each channel (since they share state in same instance)
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=99  # Fixed seed for carrier
        )
        carrier2 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=99  # Same fixed seed for carrier
        )

        channel1 = signal_kit.Channel([carrier1])
        channel1.set_noise_floor_db(-100.0)
        channel1.set_seed(42)
        iq1 = channel1.generate(1000)

        channel2 = signal_kit.Channel([carrier2])
        channel2.set_noise_floor_db(-100.0)
        channel2.set_seed(42)
        iq2 = channel2.generate(1000)

        # Same seed and same carrier seed should produce same channel noise
        # (though exact match may vary due to independent AWGN generation)
        np.testing.assert_array_almost_equal(iq1, iq2, decimal=5)


class TestChannelImpairments:
    """Test adding and applying channel impairments."""

    def test_add_impairment_digitizer_droop_ad9361_string(self):
        """Test adding digitizer droop impairment using string shorthand."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        # Should not raise an error
        channel.add_impairment("digitizer_droop_ad9361")
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_digitizer_droop_traditional_string(self):
        """Test adding traditional digitizer droop using string shorthand."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        channel.add_impairment("digitizer_droop_traditional")
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_cosine_taper_string(self):
        """Test adding cosine taper digitizer impairment."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        channel.add_impairment("cosine_taper_digitizer")
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_custom_digitizer_droop_dict(self):
        """Test adding custom digitizer droop using dictionary config."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        channel.add_impairment({
            "type": "digitizer_droop",
            "order": 4,
            "cutoff": 0.48
        })
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_frequency_variation_dict(self):
        """Test adding frequency variation impairment using dictionary config."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        channel.add_impairment({
            "type": "frequency_variation",
            "amplitude_db": 1.0,
            "cycles": 3.0,
            "phase_offset": 0.0
        })
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_frequency_variation_no_phase_offset(self):
        """Test frequency variation with optional phase_offset omitted."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        # phase_offset should default to 0.0
        channel.add_impairment({
            "type": "frequency_variation",
            "amplitude_db": 0.5,
            "cycles": 2.0
        })
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_impairment_cosine_taper_custom_dict(self):
        """Test adding custom cosine taper using dictionary config."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        channel.add_impairment({
            "type": "cosine_taper",
            "passband_end": 0.40,
            "stopband_start": 0.45
        })
        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_add_multiple_impairments(self):
        """Test adding multiple impairments sequentially."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)

        # Add multiple impairments
        channel.add_impairment("digitizer_droop_ad9361")
        channel.add_impairment({
            "type": "frequency_variation",
            "amplitude_db": 0.5,
            "cycles": 2.0
        })

        iq = channel.generate(1000)
        assert len(iq) == 1000

    def test_invalid_impairment_string(self):
        """Test that invalid impairment string raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])

        with pytest.raises(ValueError, match="Unknown impairment"):
            channel.add_impairment("invalid_impairment")

    def test_invalid_impairment_dict_missing_type(self):
        """Test that dict without 'type' key raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])

        with pytest.raises(ValueError, match="must have 'type' key"):
            channel.add_impairment({"order": 3})

    def test_invalid_impairment_dict_missing_params(self):
        """Test that dict with missing required parameters raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])

        # Missing 'cutoff' parameter
        with pytest.raises(ValueError, match="requires 'cutoff' parameter"):
            channel.add_impairment({
                "type": "digitizer_droop",
                "order": 3
            })

    def test_invalid_cutoff_out_of_range(self):
        """Test that cutoff outside [0.0, 0.5] raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])

        with pytest.raises(ValueError, match="cutoff must be in range"):
            channel.add_impairment({
                "type": "digitizer_droop",
                "order": 3,
                "cutoff": 0.6  # Out of range
            })

    def test_invalid_cosine_taper_passband_stopband_order(self):
        """Test that stopband_start <= passband_end raises error."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])

        with pytest.raises(ValueError, match="stopband_start must be greater than passband_end"):
            channel.add_impairment({
                "type": "cosine_taper",
                "passband_end": 0.45,
                "stopband_start": 0.40  # Wrong order
            })


class TestChannelWithMultipleCarriersAndImpairments:
    """Test multi-carrier channel simulation with impairments."""

    def test_multi_carrier_with_impairment(self):
        """Test multi-carrier channel with impairment applied to combined signal."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.1,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        carrier2 = signal_kit.Carrier(
            modulation="16QAM",
            bandwidth=0.15,
            center_freq=-0.15,
            snr_db=15.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=43
        )

        channel = signal_kit.Channel([carrier1, carrier2])
        channel.set_noise_floor_db(-85.0)
        channel.add_impairment("digitizer_droop_ad9361")

        iq = channel.generate(10000)
        assert len(iq) == 10000
        assert iq.dtype == np.complex128

    def test_multi_carrier_multiple_impairments(self):
        """Test multi-carrier channel with multiple impairments."""
        carrier1 = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.1,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        carrier2 = signal_kit.Carrier(
            modulation="8PSK",
            bandwidth=0.08,
            center_freq=-0.1,
            snr_db=12.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=43
        )

        channel = signal_kit.Channel([carrier1, carrier2])
        channel.set_noise_floor_db(-85.0)

        # Add multiple impairments
        channel.add_impairment("digitizer_droop_ad9361")
        channel.add_impairment({
            "type": "frequency_variation",
            "amplitude_db": 1.0,
            "cycles": 2.0
        })

        iq = channel.generate(5000)
        assert len(iq) == 5000

    def test_channel_clean_with_impairment_in_queue(self):
        """Test that generate_clean() can be called with impairments queued."""
        carrier = signal_kit.Carrier(
            modulation="QPSK",
            bandwidth=0.1,
            center_freq=0.0,
            snr_db=10.0,
            rolloff=0.35,
            sample_rate_hz=1e6,
            seed=42
        )
        channel = signal_kit.Channel([carrier])
        channel.set_noise_floor_db(-100.0)  # Must set even for clean generation
        channel.add_impairment("digitizer_droop_ad9361")

        # generate_clean() should work (impairments not applied to clean signal)
        iq_clean = channel.generate_clean(1000)
        assert len(iq_clean) == 1000
        assert iq_clean.dtype == np.complex128
