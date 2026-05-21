"""
Tests for the Python bindings of signal-kit FarrowResampler.
"""

import pytest
import numpy as np

try:
    import signal_kit
except ImportError:
    pytest.skip("signal-kit not installed, run 'maturin develop' first", allow_module_level=True)


class TestFarrowResamplerCreation:
    """Test FarrowResampler construction and validation."""

    def test_create_resampler(self):
        resampler = signal_kit.FarrowResampler(1e6, 1.5e6)
        assert resampler is not None

    def test_invalid_input_rate_zero(self):
        with pytest.raises(ValueError, match="input_rate_hz must be positive"):
            signal_kit.FarrowResampler(0.0, 1e6)

    def test_invalid_input_rate_negative(self):
        with pytest.raises(ValueError, match="input_rate_hz must be positive"):
            signal_kit.FarrowResampler(-1.0, 1e6)

    def test_invalid_output_rate_zero(self):
        with pytest.raises(ValueError, match="output_rate_hz must be positive"):
            signal_kit.FarrowResampler(1e6, 0.0)


class TestFarrowResamplerProcess:
    """Test FarrowResampler block-processing behaviour."""

    def test_dc_input_yields_dc_output(self):
        resampler = signal_kit.FarrowResampler(1e6, 1.5e6)
        dc = 0.7 - 0.3j
        input_signal = np.full(200, dc, dtype=np.complex128)
        output = resampler.process(input_signal)
        np.testing.assert_allclose(output[10:].real, dc.real, atol=1e-10)
        np.testing.assert_allclose(output[10:].imag, dc.imag, atol=1e-10)

    def test_output_length_within_one_of_predicted(self):
        input_rate = 1e6
        output_rate = 1.5e6
        input_len = 1024
        resampler = signal_kit.FarrowResampler(input_rate, output_rate)
        input_signal = np.zeros(input_len, dtype=np.complex128)
        output = resampler.process(input_signal)
        expected = round(input_len * output_rate / input_rate)
        assert abs(len(output) - expected) <= 1

    def test_block_split_continuity(self):
        input_rate = 1e6
        output_rate = 1.5e6
        rng = np.random.default_rng(42)
        n = 1024
        input_signal = (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ).astype(np.complex128)

        single = signal_kit.FarrowResampler(input_rate, output_rate)
        full_output = single.process(input_signal)

        split_resampler = signal_kit.FarrowResampler(input_rate, output_rate)
        first_half = split_resampler.process(input_signal[:400])
        second_half = split_resampler.process(input_signal[400:])
        split_output = np.concatenate([first_half, second_half])

        assert full_output.shape == split_output.shape
        np.testing.assert_allclose(full_output, split_output, atol=1e-10)

    def test_reset_clears_state(self):
        resampler = signal_kit.FarrowResampler(1e6, 1.5e6)
        rng = np.random.default_rng(7)
        input_signal = (
            rng.standard_normal(256) + 1j * rng.standard_normal(256)
        ).astype(np.complex128)
        first = resampler.process(input_signal)
        resampler.reset()
        second = resampler.process(input_signal)
        np.testing.assert_array_equal(first, second)
