# =============================================================================
# channel.py - Kanal modelleri
#   Ideal
#   AWGN
#   Multipath + AWGN
# =============================================================================

import numpy as np


def _awgn_noise_power(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    return signal_power / snr_linear


def _add_awgn(signal, noise_power):
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape)
    )
    return signal + noise


def apply_channel(tx_signal, channel_type="Ideal", snr_db=20.0, return_metadata=False):
    if channel_type == "Ideal":
        rx_signal = tx_signal.copy()
        metadata = {
            "channel_type": channel_type,
            "impulse_response": np.array([1.0 + 0.0j], dtype=complex),
            "noise_power": 0.0,
        }
        return (rx_signal, metadata) if return_metadata else rx_signal

    if channel_type == "AWGN":
        noise_power = _awgn_noise_power(tx_signal, snr_db)
        rx_signal = _add_awgn(tx_signal, noise_power)
        metadata = {
            "channel_type": channel_type,
            "impulse_response": np.array([1.0 + 0.0j], dtype=complex),
            "noise_power": float(noise_power),
        }
        return (rx_signal, metadata) if return_metadata else rx_signal

    if channel_type == "Multipath":
        num_paths = 3

        gains = np.random.randn(num_paths) + 1j * np.random.randn(num_paths)
        gains = gains / np.linalg.norm(gains)

        # Causal linear convolution; frame lengthini sabit tutmak icin truncate edilir.
        rx_clean = np.convolve(tx_signal, gains, mode="full")[: tx_signal.shape[0]]
        noise_power = _awgn_noise_power(rx_clean, snr_db)
        rx_signal = _add_awgn(rx_clean, noise_power)
        metadata = {
            "channel_type": channel_type,
            "impulse_response": gains,
            "noise_power": float(noise_power),
        }
        return (rx_signal, metadata) if return_metadata else rx_signal

    raise ValueError(f"Desteklenmeyen kanal turu: {channel_type}")
